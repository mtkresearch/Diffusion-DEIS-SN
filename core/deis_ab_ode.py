import torch
import numpy as np
from typing import Tuple, Union, Optional, List

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import register_to_config
from .sched import DDPMScheduler
from tqdm import tqdm

# official deis implementation used as a reference


# use automatic differentiation
def compute_grad(func, x):
    """Compute gradient of a function, vector input and output."""

    x.requires_grad_(True)
    x.retain_grad()
    out = func(x)

    # assert out.shape == x.shape
    v=torch.ones_like(out)

    out.backward(v)

    return x.grad.detach()

def interp_fn(x, xp,fp):
    dev = x.device
    original_dims = len(x.shape)
    while len(x.shape) > 1:
        x = x.squeeze(-1)
    if xp.shape[0] != fp.shape[0]:
        raise ValueError("xp and fp must be the same length.")
    xp = xp.to(dev)
    fp = fp.to(dev)

    # search sorted expects increasing values for xp
    # xp is 1D, fp may be higher dimesion
    # but needs to match xp on the outermost dim
    i = torch.clip(
        torch.searchsorted(xp,x,side="right"),
        min=0, max=len(xp)-1
    )

    df = fp[i] - fp[i-1]
    dx = xp[i] - xp[i-1]
    delta = x - xp[i-1]
    while len(delta.shape) != len(df.shape):
        dx = dx.unsqueeze(-1)
        delta = delta.unsqueeze(-1) 

    f = torch.where(
        (dx==0), 
        fp[i], 
        fp[i-1] + (delta/dx)*df
    )
    while len(f.shape) < original_dims:
        f = f.unsqueeze(-1)
    return f


def rk4(
    x_start, t_start, t_end, n_steps, grad_func
):
    x_start = torch.as_tensor(x_start)
    step_size = (t_end - t_start)/n_steps

    output = torch.zeros(n_steps+1,*x_start.shape)
    output[0] = x_start
    ts = torch.linspace(t_start,t_end,n_steps+1)
    # RK4 integration

    for i in tqdm(range(int(n_steps))):
        t = ts[i]
        y = output[i]

        k1 = grad_func(t, y)
        k2 = grad_func(t+step_size/2, y+step_size/2*k1)
        k3 = grad_func(t+step_size/2, y+step_size/2*k2)
        k4 = grad_func(t+step_size, y+step_size*k3)

        output[i+1] = y+step_size/6*(k1+2*k2+2*k3+k4)
    return output, ts


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class DEISABODESchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """
    prev_sample: torch.FloatTensor
    earlier_outputs: List
    score_est: torch.FloatTensor
    L:torch.FloatTensor


class DEISABODEScheduler(DDPMScheduler):
    @register_to_config
    def __init__(
        self, 
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
        timestep_spacing: str = 'quadratic',
        resolution: int = 32,
        order=2
    ):
        super().__init__(
            num_train_timesteps = num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas, 
            variance_type=variance_type, 
            clip_sample=clip_sample, 
            prediction_type=prediction_type, 
            thresholding=thresholding, 
            dynamic_thresholding_ratio=dynamic_thresholding_ratio, 
            clip_sample_range=clip_sample_range, 
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            resolution=resolution,
        )
        self.schedule = beta_schedule
        assert self.schedule in ["linear", "squaredcos_cap_v2", "cosine_simple_diffusion", "spd"]
        # note that for the VP case considered here, we can analytically
        # get the transition matrix Psi 
        self.order = order

    

    
    def tau2alpha(self, tau):
        """Map tau to alpha continuously."""
        if self.schedule == "cosine_simple_diffusion":
            # includes resolution adjustment
            logsnr_max, logsnr_min = 10, -10
            reference_resolution = 32
            ratio = reference_resolution/self.resolution
            limit_max = np.arctan(np.exp(-0.5*logsnr_max))
            limit_min = np.arctan(np.exp(-0.5*logsnr_min)) - limit_max
            log_snr = -2*torch.log(torch.tan(limit_min*tau + limit_max)) + 2*np.log(ratio)
            alpha = torch.sigmoid(log_snr)
            
        # elif self.schedule == "squaredcos_cap_v2":
            # s = 0.008

            # alpha = torch.cos(
            #     (tau+s)/(1+s) * torch.pi/2
            # )**2

        elif self.schedule in ["linear", "squaredcos_cap_v2"] :
            # discrete approx using linear interp to get alpha
            # just like in official deis implementation
            discrete_alphas = self.alphas_cumprod
            T = len(discrete_alphas)

            # add 1 at tau=0
            discrete_alphas = torch.cat(
                [torch.ones(1), discrete_alphas]
            )
            alphas = interp_fn(
                tau, torch.linspace(0,1, T+1), discrete_alphas
            )
            # alpha = torch.clip(
            #     alphas, # min=1e-7, max=1 # from deis repo
            # )
            # if len(alpha.shape) > 0:
            #     print(tau[::400])
            #     print(alpha[::400])
            #     exit()
            alpha = alphas

        else:
            raise NotImplementedError(f"{self.schedule} is not implemented for {self.__class__}")

        return alpha
            
    def psi_fn(self, tau_prev, tau):
        # simplification that is only valid for VP
        # otherwise needs to numerically solve ODE

        psi = (self.tau2alpha(tau_prev)/self.tau2alpha(tau)).sqrt()

        return psi

    def K_fn(self, tau):
        # sigma in other notation
        K = (1-self.tau2alpha(tau)).sqrt()
        return K
    
    def f_fn(self, tau):
        # f = 1/2 dalpha_dtau 1/alpha
        dalpha_dtau = compute_grad(self.tau2alpha, tau)
        alpha = self.tau2alpha(tau)
        return 0.5*dalpha_dtau/alpha

    def g2_fn(self, tau):
        # simplification only for VP case
        return -2*self.f_fn(tau)

    def eps_integrand(self, taus, tau_prev):
        # returns a series of values to be later summed in 
        # taus range over the integration interval
        # numerical integration over time to get polynomial coeffs
        # print(taus)
        psi = self.psi_fn(tau_prev, taus)
        g2 = self.g2_fn(taus)
        K = torch.clip(self.K_fn(taus), min=1e-8)

        integrand = 0.5*psi*g2/K

        # exit()
        return integrand

    def poly_term(
        self, taus_inter, inference_taus, i,j, order
    ):
        ks = [k for k in range(order+1)]
        ks.remove(j)
        ks = torch.tensor(ks).to(int)


        if len(ks) != 0:
            # calc for different k and then product over k
            terms = (taus_inter.unsqueeze(-1) - inference_taus[i+ks])\
                /(inference_taus[i+j]-inference_taus[i+ks])

            terms = terms.prod(dim=-1)

        else:
            # special case for zero order
            terms = torch.ones_like(taus_inter)
        return terms # should be same length as taus_inter

    def set_Cs(self, inference_taus,num_int_steps=10000):
        """Use numerical integration to get Cs"""

        # note that for order=0 it should be analytically
        # DDIM but this fixes our choice of K
        # for notation see DEIS or gDDIM paper 

        # taus start at 0->1

        # for loop slowge descending i
        Cs = [] # list of lists
        print("Setting Cs...")
        for i in tqdm(range(len(inference_taus)-1, 0, -1)):

            order = np.min([self.order, len(inference_taus)-1-i])
            taus_inter = np.linspace(
                inference_taus[i], inference_taus[i-1],
                num_int_steps, endpoint=False 
                # no endpoint at i-1
                # this handily avoids the discontinuity at tau=0
            )

            taus_inter = torch.tensor(taus_inter)
            dt = (inference_taus[i-1]-inference_taus[i])/num_int_steps
            C = []
            integrand = self.eps_integrand(
                taus_inter, inference_taus[i-1]
            )
            for j in range(order+1): # will be truncated at edges
                poly = self.poly_term(
                    taus_inter, inference_taus,
                    i,j,order
                )

                # adjust for high dimensional f,g
                # time always outermost dimension
                while len(poly.shape) < len(integrand.shape):
                    poly = poly.unsqueeze(-1)

                C.append(
                    (integrand*poly*dt).sum(dim=0) # sum over time interval
                )

            Cs.append(C)
            
        # reverse list to start at i=1
        Cs.reverse()
        self.Cs = Cs



    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        
    ):
        """
        Set timesteps but also calculate C_ij for integration.
        Note that timesteps are shifted by one to match indexing,
        i.e. t=1000 is mapped to 999

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
            elif self.config.timestep_spacing == "quadratic":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.int64)
                # square taus
                timesteps = (timesteps/self.config.num_train_timesteps)**2 * self.config.num_train_timesteps
                # timesteps = timesteps.round()
                timesteps -= 1 # shift to match indices
            
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = torch.from_numpy(timesteps).to(device)
        # print(f"timesteps:\n{self.timesteps.tolist()}")
        # timesteps are reversed
        # inference taus for getting Cs
        # 0 -> 1 not reversed unlike timesteps
        self.inference_taus = torch.cat([(self.timesteps + 1)/self.config.num_train_timesteps, torch.zeros(1)]).flip(dims=(0,))
        if hasattr(self, "Sigma_0"):
            while len(self.inference_taus.shape) <= len(self.Sigma_0.shape):
                self.inference_taus = self.inference_taus.unsqueeze(-1)
        self.set_Cs(self.inference_taus)


    def step(self,
        model_output: torch.FloatTensor,
        earlier_outputs: List,
        time_idx:int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        generator=None
    ) -> Union[DEISABODESchedulerOutput, Tuple]:

        # note set_timesteps() needs to be called before this

        i = time_idx
        t = self.inference_taus[i].to(model_output.device)

        # estimate avg score
        score_est = -model_output/self.K_fn(t)

        # notation from eq 14 in DEIS paper
        psi_transition = self.psi_fn(
            self.inference_taus[i-1],
            self.inference_taus[i]
        )

        linear_term = psi_transition * sample

        # prepend
        model_outputs = [model_output] + earlier_outputs
        assert len(model_outputs) <= self.order + 1

        other_term = []
        for j, output in enumerate(model_outputs):
            # one less C than #inference taus
            other_term.append(
                self.Cs[i-1][j]*model_outputs[j]
            )
        

        other_term = torch.stack(other_term)
        other_term = other_term.sum(dim=0)

        pred_prev_sample = linear_term + other_term

        earlier_outputs = model_outputs
        if len(earlier_outputs) > self.order:
            del earlier_outputs[-1]
        if not return_dict:
            return (pred_prev_sample,earlier_outputs)

        return DEISABODESchedulerOutput(
            prev_sample=pred_prev_sample,
            earlier_outputs=earlier_outputs,
            score_est=score_est,
            L=self.K_fn(t)
        )


class DEISABSNODEScheduler(DEISABODEScheduler):
    @register_to_config
    def __init__(
        self, 
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
        timestep_spacing: str = 'quadratic',
        resolution: int = 32,
        order=2,
        channels: int = 3, # for modelling the data distribution
        score_abs_path: str = None,
        clip_normalizer_steps: int = 5
    ):
        super().__init__(
            num_train_timesteps = num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas, 
            variance_type=variance_type, 
            clip_sample=clip_sample, 
            prediction_type=prediction_type, 
            thresholding=thresholding, 
            dynamic_thresholding_ratio=dynamic_thresholding_ratio, 
            clip_sample_range=clip_sample_range, 
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            resolution=resolution,
            order=order
        )
        self.schedule = beta_schedule
        assert self.schedule in ["linear", "squaredcos_cap_v2", "cosine_simple_diffusion", "spd"]
        # note that for the VP case considered here, we can analytically
        # get the transition matrix Psi 
        self.order = order


        self.T = num_train_timesteps
        
        # print("Finding R...")
        # self.find_R()

        if score_abs_path is None:
            raise ValueError("Need a path to load empirical abs score values")
        score_abs_Ls = torch.load(score_abs_path)
        Ls = score_abs_Ls["Ls"].cpu()
        while len(Ls.shape) > 1:
            Ls = Ls.squeeze(-1)
        score_abs = score_abs_Ls["score_abs"]
        self.ratio = (score_abs.cpu()*Ls.cpu()).flip(dims=(0,))

        # empirically helps...
        # trunc by 0.5% (5/1000 steps)
        # trunc = int(np.round(len(self.ratio) * 0.005))
        self.ratio = self.ratio[clip_normalizer_steps:]
        print(f"truncation steps {clip_normalizer_steps}")
        # pad at tau =0 assume same at point of truncation
        self.ratio = torch.cat(
            [self.ratio[0].repeat(clip_normalizer_steps), self.ratio]
        )

        print("ratio loaded...............")


    

    
    def tau2alpha(self, tau):
        """
        Map tau to alpha continuously.
        This is all in transform space.
        """
        if self.schedule == "cosine_simple_diffusion":
            # includes resolution adjustment
            logsnr_max, logsnr_min = 10, -10
            reference_resolution = 32
            ratio = reference_resolution/self.resolution
            limit_max = np.arctan(np.exp(-0.5*logsnr_max))
            limit_min = np.arctan(np.exp(-0.5*logsnr_min)) - limit_max
            log_snr = -2*torch.log(torch.tan(limit_min*tau + limit_max)) + 2*np.log(ratio)
            alpha = torch.sigmoid(log_snr)
            
        # elif self.schedule == "squaredcos_cap_v2":
            # s = 0.008

            # alpha = torch.cos(
            #     (tau+s)/(1+s) * torch.pi/2
            # )**2

        elif self.schedule in ["linear", "squaredcos_cap_v2"] :
            # discrete approx using linear interp to get alpha
            # just like in official deis implementation
            discrete_alphas = self.alphas_cumprod
            assert self.T  == len(discrete_alphas)

            # add 1 at tau=0
            discrete_alphas = torch.cat(
                [torch.ones(1), discrete_alphas]
            )
            alphas = interp_fn(
                tau, torch.linspace(0,1, self.T+1), discrete_alphas
            )
            # alpha = torch.clip(
            #     alphas, # min=1e-7, max=1 # from deis repo
            # )
            # if len(alpha.shape) > 0:
            #     print(tau[::400])
            #     print(alpha[::400])
            #     exit()
            alpha = alphas


        else:
            raise NotImplementedError(f"{self.schedule} is not implemented for {self.__class__}")

        return alpha









    def L_fn(self, tau):
        # sigma in other notation
        L = (1-self.tau2alpha(tau)).sqrt()
        return L

    def K_fn(self, tau):
        # use ratio

        K = self.L_fn(tau)
        K = K/self.ratio_fn(tau)
        return K



    def ratio_fn(self, tau):
        # ratio should be generated using trailing 
        ratio = interp_fn(
                tau, torch.linspace(0,1, len(self.ratio)), self.ratio
        )
        return ratio
    

    def eps_integrand(self, taus, tau_prev):
        # returns a series of values to be later summed in 
        # taus range over the integration interval
        # numerical integration over time to get polynomial coeffs
        # print(taus)
        psi = self.psi_fn(tau_prev, taus)
        g2 = self.g2_fn(taus)
        K = torch.clip(self.K_fn(taus), min=1e-8)

        integrand = 0.5*psi*g2/K

        # exit()
        return integrand

    def poly_term(
        self, taus_inter, inference_taus, i,j, order
    ):
        ks = [k for k in range(order+1)]
        ks.remove(j)
        ks = torch.tensor(ks).to(int)


        if len(ks) != 0:
            # calc for different k and then product over k
            terms = (taus_inter.unsqueeze(-1) - inference_taus[i+ks])\
                /(inference_taus[i+j]-inference_taus[i+ks])

            terms = terms.prod(dim=1)

        else:
            # special case for zero order
            terms = torch.ones_like(taus_inter)
        return terms # should be same length as taus_inter

    def set_Cs(self, inference_taus,num_int_steps=10000):
        """Use numerical integration to get Cs"""

        # note that for order=0 it should be analytically
        # DDIM but this fixes our choice of K
        # for notation see DEIS or gDDIM paper 

        # taus start at 0->1

        # for loop slowge descending i
        # print("Setting Cs....")
        Cs = [] # list of lists
        for i in range(len(inference_taus)-1, 0, -1):

            order = np.min([self.order, len(inference_taus)-1-i])
            taus_inter = np.linspace(
                inference_taus[i], inference_taus[i-1],
                num_int_steps, endpoint=False 
                # no endpoint at i-1
                # this handily avoids the discontinuity at tau=0
            )

            taus_inter = torch.tensor(taus_inter)
            dt = (inference_taus[i-1]-inference_taus[i])/num_int_steps
            C = []

            integrand = self.eps_integrand(
                taus_inter, inference_taus[i-1]
            )
            for j in range(order+1): # will be truncated at edges
                poly = self.poly_term(
                    taus_inter, inference_taus,
                    i,j,order
                )

                # adjust for high dimensional f,g
                # time always outermost dimension
                while len(poly.shape) < len(integrand.shape):
                    poly = poly.unsqueeze(-1)

                C.append(
                    (integrand*poly*dt).sum(dim=0) # sum over time interval
                )

            Cs.append(C)
            
        # reverse list to start at i=1
        Cs.reverse()
        self.Cs = Cs



    def step(self,
        model_output: torch.FloatTensor,
        earlier_outputs: List,
        time_idx:int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        generator=None
    ) -> Union[DEISABODESchedulerOutput, Tuple]:

        # note set_timesteps() needs to be called before this

        i = time_idx
        t = self.inference_taus[i].to(model_output.device)
        t_prev = self.inference_taus[i-1].to(model_output.device)

        # score function normalisation
        score_est = -model_output/self.L_fn(t)
        eps = -self.K_fn(t)*score_est

        # notation from eq 14 in DEIS paper
        psi_transition = self.psi_fn(
            t_prev,
            t
        )

        linear_term = psi_transition * sample

        # prepend
        epses = [eps] + earlier_outputs
        assert len(epses) <= self.order + 1

        other_term = []
        for j, output in enumerate(epses):
            # one less C than #inference taus
            other_term.append(
                self.Cs[i-1][j]*epses[j]
            )
        

        other_term = torch.stack(other_term)
        other_term = other_term.sum(dim=0)

        pred_prev_sample = linear_term + other_term

        earlier_outputs = epses
        if len(earlier_outputs) > self.order:
            del earlier_outputs[-1]
        if not return_dict:
            return (pred_prev_sample,earlier_outputs)

        return DEISABODESchedulerOutput(
            prev_sample=pred_prev_sample,
            earlier_outputs=earlier_outputs,
            score_est=score_est,
            L=self.L_fn(t)
        )

