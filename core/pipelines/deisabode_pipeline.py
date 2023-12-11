from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DEISABODEPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        return_intermediates: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            return_intermediates (`bool`, *optional*, defaults to `False`):
                Whether or not to return all steps of the reverse diffusion process, i.e. the intermediates

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        # this also sets Cs values for the sampler
        # try doing outside pipeline, otherwise called every batch
        # It happens in self.scheduler.set_timesteps()

        intermediates = [ ]
        earlier_outputs = []
        score_abs = []
        Ls = []
        for idx, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            
            # for indexing the taus which go 0->1
            time_idx = len(self.scheduler.inference_taus) - 1 - idx

            # 1. predict noise model_output
            with torch.no_grad():
                model_output = self.unet(image, t).sample
                



            # 2. compute previous image: x_t -> x_t-1
            out = self.scheduler.step(
                model_output, earlier_outputs,
                time_idx,
                image, generator=generator
            )
            
            
            image, earlier_outputs = out.prev_sample, out.earlier_outputs
            
            score_est, L = out.score_est, out.L
            abs = torch.abs(score_est).mean().item()
            score_abs.append(abs)
            Ls.append(L)
            if return_intermediates: intermediates.append(image)
        
        # torch.save(
        #         {
        #             "score_abs": torch.tensor(score_abs),
        #             "Ls": torch.stack(Ls).cpu()
        #         }
        #     , "lsunchurch_Ls_score_abs.pt")
        # print("saved")
        if return_intermediates:
            intermediates = torch.stack(intermediates, 0)
            intermediates = (intermediates / 2 + 0.5).clamp(0, 1)
            intermediates = intermediates.cpu().permute(0, 1, 3, 4, 2).detach().numpy()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, intermediates)

        return ImagePipelineOutput(images=image)
