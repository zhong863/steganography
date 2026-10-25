"""SAMPLING ONLY."""
import torch
from .dpm_solver_plusplus import NoiseScheduleVP, model_wrapper, DPM_Solver

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               steps,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               t_start=None,
               t_end=None,
               DPMencode=False,
               DPMdecode=False,
               order=3,
               width=None,
               height=None,
               ref=False,
               top=None, 
               left=None, 
               bottom=None, 
               right=None,
               segmentation_map=None,
               param=None,
               target_height=None, 
               target_width=None,
               center_row_rm=None,
               center_col_rm=None,
               tau_a=0.4,
               tau_b=0.8,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {steps}')

        device = self.model.betas.device
        if x_T is None:
            x = torch.randn(size, device=device)
        else:
            x = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
     
        if DPMencode:
            # x_T is not a list
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode: self.model.apply_model(x, t, c, encode=DPMencode),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                # if u want to set guidance scale = 1 when inversion, you can use conditional embedding as uc
                # unconditional_condition=conditioning,
                guidance_scale=unconditional_guidance_scale,
            )

            dpm_solver = DPM_Solver(model_fn, ns)
            data = self.low_order_sample(x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=DPMencode)
            
            for step in range(order, steps + 1):
                data = dpm_solver.sample_one_step(data, step, steps, order=order, DPMencode=DPMencode)   
                     
            return data['x'].to(device), None
        else:
            model_fn_gen = model_wrapper(
                lambda x, t, c, DPMencode: self.model.apply_model(x, t, c, encode=DPMencode),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )
            dpm_solver_gen = DPM_Solver(model_fn_gen, ns)
            gen = self.low_order_sample(x, dpm_solver_gen, steps, order, t_start, t_end, device,
                                           DPMencode=DPMencode)

            for step in range(order, steps + 1):
                gen = dpm_solver_gen.sample_one_step(gen, step, steps, order=order, DPMencode=DPMencode,)

            return gen['x'].to(device), None

            
    
    def low_order_sample(self, x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=False):
        
        t_0 = 1. / dpm_solver.noise_schedule.total_N if t_end is None else t_end
        t_T = dpm_solver.noise_schedule.T if t_start is None else t_start

        assert steps >= order
        timesteps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=t_T, t_0=t_0, N=steps, device=device, DPMencode=DPMencode)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            # x_theta_0
            model_prev_list = [dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode)]

            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = dpm_solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                           solver_type='dpmsolver', DPMencode=DPMencode)
                # x_theta_1
                model_prev_list.append(dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode))
                t_prev_list.append(vec_t)
        
        return {'x': x, 'model_prev_list': model_prev_list, 't_prev_list': t_prev_list, 'timesteps':timesteps}
    