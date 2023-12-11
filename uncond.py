import json
import argparse
import shutil
import os, sys
import yaml, copy

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

from utils import latest_n_checkpoints, yaml_interface
from evaluations.fid_score import compute_fid
from core import (
    init_model,
    DDPMPipeline, DDIMPipeline, DEISABODEPipeline,
    DEISABODEScheduler,
    DDPMScheduler, DDIMScheduler,
    DEISABSNODEScheduler
)

# `diffusers`` v0.11 has a bug related to distributed training.
# Please upgrade diffusers to v0.12.1 (known to work) at least.
check_min_version("0.16.1")

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


@yaml_interface(__file__)
def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional DDPM training & sampling script.")
    parser.add_argument('--train', type=str, choices=['ddpm', ], default='ddpm',
                        help='The noising mode for training. Only `ddpm` available.')
    parser.add_argument('--infer', type=str, choices=['ddpm', 'ddim', "deisabode", "deissnode"], default='ddpm',
                        help='The sampler. Choose between `ddpm` or `ddim`')
    parser.add_argument(
        "--sampler_order",
        type=int,
        default=2,
        help="Order of DEIS or gDDIM."
    )
    parser.add_argument(
        "--score_abs_Ls_path",
        type=str,
        default="/proj/gpu_d_98001/proj_diffusion/evaluation/score_abs_Ls.pt",
        help="path of stored average abs score values."
    )
    parser.add_argument(
        "--clip_normalizer_steps",
        type=int,
        default=5,
        help='clipping steps of the normalizer function'
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where all experiment related stuff will be written.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="a name for the experiment being run. a folder with this name will be created."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument('--unet_json',
        type=str, required=False, default='./arch/diffuser_default_unet.json',
        help='A valid JSON that represents UNet architecture'
    )
    parser.add_argument(
        "--pixel_resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--saving_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        '--clip_grad_norm', type=float, default=1.0, help="graident norm clipping; no clipping if <= 0."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        )
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v_prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--clip_sample", action='store_true', help='whether to clip while generating')
    parser.add_argument(
        "--reverse_variance",
        type=str,
        choices=['fixed_small', 'fixed_large', 'recommended'],
        default='recommended',
        required=False,
        help='The variance of the reverse process. fixed_small is \Tilde{\beta}, and fixed_large is \beta'
    )
    parser.add_argument("--diffusion_num_steps", type=int, default=1000)
    parser.add_argument("--diffusion_beta_schedule", type=str, default="linear")
    parser.add_argument("--diffusion_beta_linear_params", type=float, nargs=2, default=[1.e-4, 2.e-2],
                        help='start and end of linear beta schedule.')
    parser.add_argument("--inference_num_steps", type=int, default=1000)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help='Sample this many images in evaluation phase'
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        '--sample_with',
        type=str,
        default=None,
        nargs='+',
        help=(
            'Provide the name of the saved pipeline folder, e.g. pipeline-10 & an optional exp name postfix. \
                Make sure the `--output_dir` and `--exp_name` are pointing to the right location.'
            'If not None, this flag switches the whole script to only-sampling mode. \
                The same `--eval_batch_size` and `--num_samples` arguments are used for sampling.'
        )
    )
    parser.add_argument('--reference_batch_path', type=str, default=None, help='reference batch for FID computation')
    parser.add_argument(
        '--timestep_spacing',
        type=str,
        choices=['leading', 'trailing', 'linspace','quadratic'],
        default='trailing',
        help=(
            'Timestep definition style for inference.'
        )
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether peforming finetuning by starting from a pre-trained unet. Should be the path to the model ckpt \
                that can be loaded with `torch.load()`"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optionally set the random seed."
        ),
    )

    def validate_args(args):
        if args.dataset_name is None and args.train_data_dir is None:
            raise ValueError("You must specify either a dataset name from the hub or a train data directory.")
        
        if args.num_samples > 0 and not args.reference_batch_path:
            raise ValueError("You must specify a reference batch for FID since num_sample > 0")
        
        dargs = argparse.Namespace() # derived arguments
        
        dargs.sched_kwargs = {
            'num_train_timesteps': args.diffusion_num_steps,
            'prediction_type': args.prediction_type
        }
        if args.train == 'ddpm':
            dargs.sched_kwargs.update({
                'beta_schedule': args.diffusion_beta_schedule,
                'beta_start': args.diffusion_beta_linear_params[0],
                'beta_end': args.diffusion_beta_linear_params[1],
                'clip_sample': args.clip_sample,
                "resolution": args.pixel_resolution
            })
        
        dargs.pipe_kwargs = {
            'batch_size': args.eval_batch_size,
            'num_inference_steps': args.inference_num_steps,
            'output_type': "numpy",
            'return_dict': False
        }
        if args.infer == 'ddim':
            # DDIM doesn't work well without this
            dargs.pipe_kwargs.update({'use_clipped_model_output': True})
            dargs.sched_kwargs.pop('resolution', None)
        elif args.infer == 'ddpm':
            # variance only needed in stochastic samplers
            if args.reverse_variance == 'recommended':
                reverse_variance = "fixed_small" if args.inference_num_steps <= 300 else "fixed_large"
                dargs.sched_kwargs.update({'variance_type': reverse_variance})
            else:
                dargs.sched_kwargs.update({'variance_type': args.reverse_variance})

        
        # ODE timestep spacing config
        if "ode" in args.infer:
            dargs.sched_kwargs.update({'timestep_spacing': args.timestep_spacing})
        if "deissn" in args.infer:
            dargs.sched_kwargs.update({'score_abs_path': args.score_abs_Ls_path})
            dargs.sched_kwargs.update({'clip_normalizer_steps': args.clip_normalizer_steps})
    
        dargs.SchedulerClass, dargs.PipelineClass = {
            'ddpm': { 'ddpm': (DDPMScheduler, DDPMPipeline), 
                      'ddim': (DDIMScheduler, DDIMPipeline), 
                      'deisabode': (DEISABODEScheduler, DEISABODEPipeline),
                      'deissnode': (DEISABSNODEScheduler, DEISABODEPipeline),
                    }
        }[args.train][args.infer]

        if isinstance(args.sample_with, str):
            dargs.sample_exp_name = ''
        elif isinstance(args.sample_with, list):
            dargs.sample_exp_name = '' if len(args.sample_with) == 1 else ('-' + args.sample_with[1])
            args.sample_with = args.sample_with[0]

        return args, dargs

    # return the parser instane and a function to validate args
    return parser, validate_args


def main(args, dargs):
    output_dir = os.path.join(args.output_dir, args.exp_name)
    logging_dir = os.path.join(output_dir, args.logger)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_dir=logging_dir,
    )

    ModelClass, model = init_model(args.unet_json,
        sample_size=args.pixel_resolution,
        in_channels=3, out_channels=3
    )
    accelerator.print(f'Loaded {ModelClass.__name__} architecture from {args.unet_json}')

    # Load pre-trained weights if finetuning
    if args.finetune_from_checkpoint is not None:
        assert os.path.exists(args.finetune_from_checkpoint), f'Cannot find the pre-trained checkpoint {args.finetune_from_checkpoint}!'
        pretrained_state_dict = torch.load(args.finetune_from_checkpoint)
        model.load_state_dict(pretrained_state_dict)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay
        )

    noise_scheduler = dargs.SchedulerClass(**dargs.sched_kwargs)
    # This is only for RGFF time embeddings. The model needs to know the total
    # diffusion steps in order to convert discrete 't's to float [0., 1.].
    model.diffusion_steps = dargs.sched_kwargs['num_train_timesteps']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
        key = {
            # add other datasets when tested
            'cifar10': 'img'
        }.get(args.dataset_name, 'image')
    else:
        if os.path.isfile(os.path.join(args.train_data_dir, 'state.json')):
            # Arrow saved datasets always have a 'state.json' file as part of it's protocol
            dataset = load_from_disk(args.train_data_dir)
            accelerator.print(f'Loaded `Arrow` dataset from {args.train_data_dir}')
        else:
            dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
            accelerator.print(f'Loaded `ImageFolder` dataset from {args.train_data_dir}')
        key = "image"
        # See more about loading custom images at https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = Compose(
        [
            Resize(args.pixel_resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.pixel_resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples[key]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=min(8, os.cpu_count() // 2)
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = len(train_dataloader)

    if args.use_ema:
        accelerator.register_for_checkpointing(ema_model)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0] + \
            (("-" + args.sample_with + dargs.sample_exp_name) if args.sample_with else '') # sampling-only specific trackers
        accelerator.init_trackers(run)

        logging_dir = accelerator.get_tracker(args.logger).logging_dir
        unet_json_file = os.path.join(logging_dir, 'arch.json')
        with open(unet_json_file, 'w') as f:
            json.dump(
                model.module.config,
                f, sort_keys=True, indent=4
            )
        args.unet_json = unet_json_file

        # write all config for this exp to the tracker directory
        config_file = os.path.join(logging_dir, 'config.yml')
        with open(config_file, 'w') as f:
            yaml.dump(
                {os.path.basename(__file__): vars(args)}, f,
                default_flow_style=False
            )
        accelerator.print(f'The config file is written at {config_file}')

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            path = None if not os.path.exists(os.path.join(output_dir, path)) else os.path.join(output_dir, path)
        else:
            path = latest_n_checkpoints(output_dir, prefix='checkpoint', last_n=1)
            path = os.path.join(output_dir, path[0]) if len(path) > 0 else None

        if path is None:
            accelerator.print(
                f"Can't load from '{args.resume_from_checkpoint}'. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % num_update_steps_per_epoch
    
    if args.use_ema:
        ema_model.to(accelerator.device)
    
    if args.sample_with:
        assert os.path.exists(os.path.join(output_dir, args.sample_with)), \
            f"could not find saved pipeline {args.sample_with} in {output_dir}"
        first_epoch = int(args.sample_with.split('-')[-1])
        args.num_epochs = first_epoch + 1 # make it believe that only one epoch left

    for epoch in range(first_epoch, args.num_epochs):
        model.train() if not args.sample_with else ...
        
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=((not accelerator.is_main_process) or args.sample_with) or (not sys.stdout.isatty()))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            if args.sample_with: 
                print('Skipping training. Only sampling will be performed.')
                break # if sampling mode, skip training

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residual
            model_output = model(noisy_images, timesteps).sample

            if args.prediction_type == "epsilon":
                loss = F.mse_loss(model_output, noise)  # this could have different weights!
            elif args.prediction_type == "sample" and args.train == "ddpm":
                alpha_t = _extract_into_tensor(
                    noise_scheduler.alphas_cumprod.to(clean_images.device), 
                    timesteps, (clean_images.shape[0],1,1,1)
                )
                snr_weights = alpha_t / (1 - alpha_t)
                loss = snr_weights * F.mse_loss(
                    model_output, clean_images, reduction="none"
                )  # use SNR weighting from distillation paper
                loss = loss.mean()
            elif args.prediction_type == "sample" and args.train == "spd":
                psi_t = noise_scheduler.alphas_cumprod[timesteps]

                freq_space_snr_weights = (psi_t /(1-psi_t)).sqrt() # elementwise
                trans_output = noise_scheduler.transform(model_output)
                trans_clean_images = noise_scheduler.transform(clean_images)
                filtered_output = noise_scheduler.anti_transform(
                    freq_space_snr_weights * trans_output
                )
                filtered_clean = noise_scheduler.anti_transform(
                    freq_space_snr_weights * trans_clean_images
                )
                loss = F.mse_loss(
                    filtered_output, filtered_clean
                )

            # same for uniform and spd
            elif args.prediction_type == "v_prediction":
                v_target = noise_scheduler.get_velocity(
                    clean_images, noise, timesteps
                )

                loss = F.mse_loss(
                    model_output, v_target
                )

            else:
                raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if args.clip_grad_norm > 0.:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.use_ema:
                    ema_model.step(model.parameters())
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # save first, then clear all but last_n checkpoints
                        prev_checkpoints = latest_n_checkpoints(output_dir, prefix='checkpoint', all_but=True, last_n=3)
                        for cp in prev_checkpoints: shutil.rmtree(os.path.join(output_dir, cp), ignore_errors=True)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            if global_step % 1 == 0:
                accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if (epoch % args.saving_epochs == 0 or epoch == args.num_epochs - 1) or args.sample_with:
            noise_scheduler.set_timesteps(dargs.pipe_kwargs["num_inference_steps"])
            
            if accelerator.is_main_process:
                sampled_images = [ ]

            unet = copy.deepcopy(accelerator.unwrap_model(model))
            if args.use_ema:
                ema_model.copy_to(unet.parameters())
            
            if not args.sample_with:
                unet.eval()
                pipeline = dargs.PipelineClass(unet=unet, scheduler=noise_scheduler)
            else:
                sample_with = os.path.join(output_dir, args.sample_with)
                unet = ModelClass.from_pretrained(sample_with, subfolder="unet")
                unet.diffusion_steps = dargs.sched_kwargs['num_train_timesteps']
                
                pipeline = dargs.PipelineClass(unet=unet, scheduler=noise_scheduler).to(accelerator.device)
            
            n_sampled_images = 0
            sampling_bar = tqdm(total=args.num_samples,
                                        disable=(not accelerator.is_main_process) or (not sys.stdout.isatty()))
            
            pipeline.set_progress_bar_config(disable=True)

            generator = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index) if \
                args.seed is not None else None
            while n_sampled_images < args.num_samples:
                images, _ = pipeline(**dargs.pipe_kwargs, generator=generator)

                images = accelerator.gather(torch.from_numpy(images).to(accelerator.device).contiguous())
                images_processed = (images.cpu().numpy() * 255).round().astype("uint8")

                n_sampled_images += images_processed.shape[0]

                if accelerator.is_main_process:
                    sampling_bar.update(
                        # a little check to ensure the manual updating does not exceed the limit of the bar at the last while iteration
                        images_processed.shape[0] if n_sampled_images <= args.num_samples else (args.num_samples - sampling_bar.n)
                    )
                    sampled_images.append(images_processed) # only rank 0 will hold them
                else:
                    del images_processed # others can delete them
            
            if accelerator.is_main_process:
                if len(sampled_images) > 0: # if sampling is requested at all
                    sampled_images = np.concatenate(sampled_images, 0)[:args.num_samples, ...]

                    # only rank 0 will do the model saving and sample logging
                    if args.logger == "tensorboard":
                        tb_logger = accelerator.get_tracker(args.logger)
                        samples_npz = os.path.join(tb_logger.logging_dir, 
                                                   'samples-{}-pipeline-{}-steps-{}-{}x{}x{}x{}.npz'.format(args.infer, epoch, args.inference_num_steps, *sampled_images.shape))
                        np.savez(samples_npz, sampled_images)
                        accelerator.print(f'Samples saved at {samples_npz}')

                        fid = compute_fid(samples_npz, args.reference_batch_path, batch_size=args.eval_batch_size)
                        sampling_bar.set_postfix({'FID': fid})

                        accelerator.log({'FID': fid}, step=epoch)
                        tb_logger.tracker.add_images(
                            "samples", sampled_images[:min(32, args.eval_batch_size), ...].transpose(0, 3, 1, 2), epoch
                        )

                        if args.sample_with:
                            fid_yaml_file = os.path.join(tb_logger.logging_dir, f'fid-{args.infer}-pipeline-{epoch}-steps-{args.inference_num_steps}-samples-{args.num_samples}.yml')
                            with open(fid_yaml_file, 'w') as yf:
                                yaml.dump({
                                    'samples': samples_npz, 'reference': args.reference_batch_path, 'fid': float(fid)
                                }, yf, default_flow_style=False)
                            accelerator.print(f'FID {float(fid):.2f} recorded at {fid_yaml_file}')
                
                if not args.sample_with: # sampling mode does not save pipeline
                    pipeline.save_pretrained(os.path.join(output_dir, f'pipeline-{epoch}'))
            
            sampling_bar.close()

    accelerator.end_training()


if __name__ == "__main__":
    args, dargs = parse_args()
    main(args, dargs)