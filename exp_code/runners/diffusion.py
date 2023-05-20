import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import noise_estimation_loss, noise_estimation_kd_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
from accelerate import Accelerator


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.build_model()

    def build_model(self):
        args, config = self.args, self.config
        model = Model(config)
        if args.load_pruned_model is not None:
            print("Loading pruned model from {}".format(args.load_pruned_model))
            states = torch.load(args.load_pruned_model, map_location='cpu')

            if isinstance(states, torch.nn.Module): # a simple pruned model 
                model = torch.load(args.load_pruned_model, map_location='cpu')
            elif isinstance(states, list): # pruned model and training states
                model = states[0]
                if args.use_ema and self.config.model.ema:
                    print("Loading EMA")
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                    self.ema_helper = ema_helper
                else:
                    self.ema_helper = None
            else:
                raise NotImplementedError
            self.model = model
        elif args.restore_from is not None and os.path.isfile(args.restore_from):
            ckpt = args.restore_from
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(
                ckpt,
                map_location='cpu',
            )
            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                model = model.to(self.device)
                model.load_state_dict(states[0], strict=True) 

            if args.use_ema and self.config.model.ema:
                print("Loading EMA")
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        
        elif not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                ckpt = os.path.join(self.args.log_path, "ckpt.pth")
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            else:
                ckpt = os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    )
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            print("Loading checkpoint {}".format(ckpt))

            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                model = model.to(self.device)
                model.load_state_dict(states[0], strict=True)

            if args.use_ema and self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == "CELEBA":
                name = 'celeba'
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.device)
            if isinstance(states, (list,tuple)):
                model.load_state_dict(states[0])
            else:
                model.load_state_dict(states)
            model.to(self.device)
        self.model = model

    def train(self, kd=False):
        accelerator = self.accelerator
        device = accelerator.device

        if kd:
            teacher = Model(self.config)
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CELEBA':
                name = "celeba"
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading teacher from {}".format(ckpt))
            states = torch.load(ckpt, map_location='cpu')
            if isinstance(states, list):
                teacher.load_state_dict(states[0])
            else:
                teacher.load_state_dict()
            teacher.to(accelerator.device)
            self.teacher = teacher.eval()
            
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = self.model
        optimizer = get_optimizer(self.config, model.parameters())
        if args.use_ema and self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            if isinstance(states[0], torch.nn.Module):
                self.model = model = states[0]
                optimizer = get_optimizer(self.config, model.parameters()) # rebuild the optimizer
            else:
                model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        self.model = model

        if ema_helper is not None:
            ema_helper.to(accelerator.device)

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            if ema_helper is not None:
                ema_helper.store(unwrapped_model.parameters())
                ema_helper.copy_to(unwrapped_model.parameters())

            with torch.no_grad():
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=accelerator.device,
                )
                x = self.sample_image(x, unwrapped_model)
                x = inverse_data_transform(config, x)
                grid = tvu.make_grid(x)
                tvu.save_image(grid, os.path.join(args.log_path, 'vis', 'Init.png'))
                #tb_logger.add_image('Before Training', grid, global_step=0)
            
            if args.use_ema:
                ema_helper.restore(unwrapped_model.parameters())

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0    
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                if kd:
                    loss = noise_estimation_kd_loss(model, teacher, x, t, e, b)
                else:
                    loss = noise_estimation_loss(model, x, t, e, b)

                #tb_logger.add_scalar("loss", loss, global_step=step)
                if step % 100 == 0:
                    logging.info(
                        f"step: {step} (Ep={epoch}/{self.config.training.n_epochs}, Iter={i}/{len(train_loader)}), loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )

                optimizer.zero_grad()
                accelerator.backward(loss)
                try:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()

                if args.use_ema and self.config.model.ema:
                    ema_helper.update(model)
                
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.eval()

                        unwrapped_model.zero_grad()
                        states = [
                            unwrapped_model,
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if args.use_ema and self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        torch.save(
                            states,
                            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                        )
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    
                data_start = time.time()

            accelerator.wait_for_everyone()
            # Sampling for visualization
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.eval()
                if ema_helper is not None:
                    ema_helper.store(unwrapped_model.parameters())
                    ema_helper.copy_to(unwrapped_model.parameters())

                with torch.no_grad():
                    n = config.sampling.batch_size
                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=accelerator.device,
                    )
                    x = self.sample_image(x, model)
                    x = inverse_data_transform(config, x)
                    grid = tvu.make_grid(x)
                    tvu.save_image(grid, os.path.join(args.log_path, 'vis', 'epoch-{}.png'.format(epoch)))
                
                if args.use_ema:
                    ema_helper.restore(unwrapped_model.parameters())
        accelerator.end_training()
            
    def sample(self):
        accelerator = self.accelerator
        model = self.model 
        model.to(accelerator.device)
        model.eval()
        
        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        args = self.args
        accelerator = self.accelerator
        import torch
        torch.manual_seed(accelerator.process_index+args.seed)
        import random
        random.seed(accelerator.process_index+args.seed)
        import numpy as np
        np.random.seed(accelerator.process_index+args.seed)

        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        
        #os.makedirs(os.path.join(self.args.image_folder, '{}'.format(accelerator.process_index)), exist_ok=True)
        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=accelerator.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
