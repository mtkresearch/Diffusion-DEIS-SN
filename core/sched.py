import torch
import numpy as np
from typing import Union, List, Optional

from diffusers.configuration_utils import register_to_config
from diffusers import DDPMScheduler as DDPMScheduler_
from diffusers import DDIMScheduler as DDIMScheduler_
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import betas_for_alpha_bar as squaredcos_cap_v2_betas
from diffusers.utils import randn_tensor
from typing import Tuple
import math




def get_alpha_bar_simple_diffusion(
    resolution, tau, 
    logsnr_min=-10, logsnr_max=10
):
    # using the cosine schedule from the blurring diffusion paper
    # this is different to the implementation in iDDPM
    # iDDPM limits beta values and has a small offset
    # Blurring (and simple) diffusion consider in the logSNR space and
    # slightly stretch the curve outwards so that the start and end 
    # are not discontinuities 

    # cosine schedule 
    # note, there is an implicit pi scaling in the arctan
    # resolution scaling from simple diffusion paper
    reference_resolution = 32
    ratio = reference_resolution/resolution
    limit_max = np.arctan(np.exp(-0.5*logsnr_max))
    limit_min = np.arctan(np.exp(-0.5*logsnr_min)) - limit_max
    log_snr = -2*torch.log(torch.tan(limit_min*tau + limit_max)) + 2*np.log(ratio)
    a_bar = torch.sigmoid(log_snr)
    return a_bar

def betas_for_alpha_bar(alpha_bar, max_beta=0.999):

    steps, *_ = alpha_bar.shape
    betas = []
    for t in range(steps):
        alpha_bar_t = alpha_bar[t].item()
        alpha_bar_t_1 = alpha_bar[t-1].item() if t != 0 else 1.
        betas.append(1. - alpha_bar_t / alpha_bar_t_1)
    return torch.clamp(torch.tensor(betas, dtype=torch.float32), max=max_beta)


class DDPMScheduler(DDPMScheduler_):
    @register_to_config
    def __init__(self, 
        num_train_timesteps: int = 1000, 
        beta_start: float = 0.0001, 
        beta_end: float = 0.02, 
        beta_schedule: str = "linear", 
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None, 
        variance_type: str = "fixed_small", 
        clip_sample: bool = True, 
        prediction_type: str = "epsilon", 
        thresholding: bool = False, 
        dynamic_thresholding_ratio: float = 0.995, 
        clip_sample_range: float = 1, 
        sample_max_value: float = 1,
        timestep_spacing: str = 'leading',
        resolution: int = 64,
        scale_snr: bool = True
    ):
        t = torch.linspace(1., num_train_timesteps, num_train_timesteps)[:, None, None, None]
        tau = t / num_train_timesteps
        self.resolution = resolution
        # Mostly copied from superclass
        self.tau = tau # to be used to compute other time related quantities
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "cosine_simple_diffusion":
            # scale snr to match resolution
            # reference resolution is 32x32
            alpha_bar = get_alpha_bar_simple_diffusion(resolution, self.tau)
            self.betas = betas_for_alpha_bar(alpha_bar)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = alpha_bar
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = squaredcos_cap_v2_betas(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif beta_schedule == "spd":
            # unused
            self.betas = squaredcos_cap_v2_betas(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type

        self.cached_generation = []
            

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`Optional[int]`):
                the number of diffusion steps used when generating samples with a pre-trained model. If passed, then
                `timesteps` must be `None`.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps are moved to.
            custom_timesteps (`List[int]`, optional):
                custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If passed, `num_inference_steps`
                must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                    raise ValueError(
                        f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                        f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                        f" maximal {self.config.num_train_timesteps} timesteps."
                    )
            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # 'leading', 'trailing' corresponds to annotation of Table 2 in https://arxiv.org/abs/2305.08891
            # Note this is implemented in later versions of diffusers, and the code is adapted from there
            if self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.int64)
                timesteps -= 1
            
            elif self.config.timestep_spacing == "linspace":
                # TODO: fix dtype later
                timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.int64)[::-1].copy()
            
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        self.alphas_cumprod = self.alphas_cumprod.to(dtype=model_output.dtype, device=model_output.device)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        # self.cached_generation.append(pred_prev_sample[0])
        # if t ==0:
        #     idx = torch.linspace(0,len(self.cached_generation)-1,144).to(int)
        #     generation = torch.stack(self.cached_generation)[idx]
        #     generation = generation.cpu()
        #     out = (generation / 2 + 0.5).clamp(0, 1).unsqueeze(0)
        #     out = (out.cpu().permute(0, 1, 3, 4, 2).squeeze(0).numpy()* 255).round().astype("uint8")
        #     np.savez("./cosine_generation.npz", out)
        #     exit()
        # if not return_dict:
        #     return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)



class DDIMScheduler(DDIMScheduler_):
    # This is useless for now, just in case any customization is needed
    pass