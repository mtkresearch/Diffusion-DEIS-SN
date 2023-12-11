import os
import json
import typing

from .pipelines import *
from .openai_unet import UNetModel, EncoderUNetModel
from diffusers.models import UNet2DModel, UNet2DConditionModel
from diffusers import VQModel
from .sched import DDPMScheduler, DDIMScheduler
# from .euler_ode import EulerODEScheduler                                
from .deis_ab_ode import *



def get_class_and_config(config_path: str):
    # grabs the model class by the right name, and instantiates it
    with open(config_path, 'r') as f:
        class_name = json.load(f)['_class_name']

    Class = eval(class_name)
    class_config = Class.load_config(config_path)
    return Class, class_config


def init_model(config_path: str, **kwargs):
    ModelClass, model_config = get_class_and_config(config_path)
    return ModelClass, ModelClass.from_config(model_config, **kwargs)


def get_vae(config_path: str, **kwargs):
    VaeClass, _ = get_class_and_config(config_path)
    vae_folder = os.path.dirname(config_path)
    vae_model: typing.Union[AutoencoderKL, VQModel] = VaeClass.from_pretrained(vae_folder)
    vae_model.set_transformed_space_parameters(npz_file=kwargs.get('freq_filter_path', None))
    return VaeClass, vae_model


