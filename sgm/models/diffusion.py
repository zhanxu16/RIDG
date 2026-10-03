import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img, tools_scale, tools_scale2, append_dims)
# from ..modules.learning.metrics import img_metrics, avg_img_metrics
import os
import numpy as np
from PIL import Image
import rasterio
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


class DiffusionEngine(pl.LightningModule):
    def __init__(
            self,
            network_config,
            denoiser_config,
            first_stage_config,
            conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            network_wrapper: Union[None, str] = None,
            ckpt_path: Union[None, str] = None,
            use_ema: bool = False,
            ema_decay_rate: float = 0.9999,
            scale_factor: float = 1.0,
            disable_first_stage_autocast=False,
            input_key: str = "jpg",
            log_keys: Union[List, None] = None,
            no_cond_log: bool = False,
            compile_model: bool = False,
            en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def init_from_ckpt(
            self,
            path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        # elif path.endswith("safetensors"):
        #     sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.decode(
                    z[n * n_samples: (n + 1) * n_samples]
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples: (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                    (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 8,
            sample: bool = True,
            ucg_keys: List[str] = None,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log


class ResidualDiffusionEngine(DiffusionEngine):
    def __init__(self, sigma_st_config, to_rgb_config, scale_01_config=None, ideal_sampler_config=None, mean_key="mu",
                 use_flash_attn2=False, compile_model=False, image_metrics="metrics", metric_norm_mode="auto", count_train_time=False,
                 count_sample_time=False, *args, **kwargs):
        if compile_model:
            os.environ["USE_COMPILE"] = "1"
        else:
            os.environ["USE_COMPILE"] = "0"
        if use_flash_attn2:
            os.environ["USE_FLASH_2"] = "1"
        else:
            os.environ["USE_FLASH_2"] = "0"

        # 两种数据集的评价指标计算方式不一样，这里通过一个不大优美的方法来区分开
        assert image_metrics in ["metrics", "evaluator"], "image_metrics should be either metrics or evaluator"
        if image_metrics == "metrics":
            from sgm.modules.learning.metrics import img_metrics, avg_img_metrics
            self.img_metrics = img_metrics
            self.avg_metrics = avg_img_metrics()
        elif image_metrics == "evaluator":
            from sgm.modules.learning.evaluator import img_metrics, avg_img_metrics
            self.img_metrics = img_metrics
            self.avg_metrics = avg_img_metrics()
        super().__init__(compile_model=compile_model, *args, **kwargs)
        self.mean_key = mean_key
        self.sigma2st = instantiate_from_config(sigma_st_config)
        self.scale_01 = instantiate_from_config(
            default(scale_01_config, {"target": "sgm.util.scale_01_from_minus1_1"})
        )
        if ideal_sampler_config is not None:
            self.ideal_sampler = instantiate_from_config(ideal_sampler_config)
        else:
            self.ideal_sampler = None
        # assert self.sampler.has("set_sigma2st"), "The sampler does not have set_sigma2st function, maybe you should use the residual sampler."
        try:
            self.sampler.set_sigma2st(self.sigma2st)
        except:
            raise NotImplementedError(
                "The sampler does not have set_sigma2st function, maybe you should use the residual sampler.")
        self.to_rgb_func = instantiate_from_config(to_rgb_config)
        self.count_sample_time = count_sample_time
        self.metric_norm_mode = metric_norm_mode
        self.count_train_time = count_train_time
        if self.count_train_time:
            self.train_start_event = None
            self.train_end_event = None
            self.train_time = 0.0
            self.train_count = 0

    def on_train_batch_start(self, *args, **kwargs):
        super().on_train_batch_start(*args, **kwargs)
        if self.count_train_time:
            self.train_start_event = torch.cuda.Event(enable_timing=True)
            self.train_end_event = torch.cuda.Event(enable_timing=True)
            self.train_start_event.record()

    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        if self.count_train_time:
            self.train_end_event.record()
            torch.cuda.synchronize()
            train_time = self.train_start_event.elapsed_time(self.train_end_event)
            self.train_count += 1
            self.train_time += train_time
            avg_train_time = self.train_time / self.train_count
            self.log_dict({"train_time": avg_train_time}, sync_dist=True, on_step=True, on_epoch=False)
            # print(f"Train time: {avg_train_time} per batch.")

    def get_input(self, batch, key):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[key]

    def forward(self, x, mu, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, self.sigma2st, x, mu, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch, self.input_key)
        x = self.encode_first_stage(x)
        mu = self.get_input(batch, self.mean_key)
        mu = self.encode_first_stage(mu)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, mu, batch)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            mu: torch.Tensor,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            return_intermediate: bool = False,
            return_denoised: bool = False,
            ideal_sample=False,
            **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c, st: self.denoiser(
            self.model, input, sigma, c, st, **kwargs
        )
        if ideal_sample:
            samples = self.ideal_sampler(randn, mu, return_intermediate=return_intermediate,
                                         return_denoised=return_denoised)
        else:
            samples = self.sampler(denoiser, randn, mu, cond, uc=uc, return_intermediate=return_intermediate,
                                   return_denoised=return_denoised)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                    (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:
                        # image cond
                        xc = x[:n, ...]
                    else:
                        xc = x
                        # raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = self.to_rgb_func(xc)
        return log

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 8,
            sample: bool = True,
            ucg_keys: List[str] = None,
            return_intermediate: bool = False,
            return_denoised: bool = False,
            return_add_mu: bool = False,
            return_add_noise: bool = False,
            return_cond: bool = False,
            return_reconstrcution: bool = False,
            return_ideal_samples: bool = False,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        mu = mu.to(self.device)[:N]
        log["inputs"] = self.to_rgb_func(x.clone().detach())
        log["mean"] = self.to_rgb_func(mu.clone().detach())

        if return_add_mu or return_add_noise:
            sigmas = self.sampler.discretization(
                self.sampler.num_steps, device=self.device
            )
            mus = [tools_scale(x.clone().detach())] if return_add_mu else None
            noises = [tools_scale(x.clone().detach())] if return_add_noise else None

            for i in reversed(self.sampler.get_sigma_gen(self.sampler.num_steps)):
                sigma = sigmas[i]
                st = self.sigma2st(sigma)
                if return_add_mu:
                    _ = x + (1 - st) / st * mu
                    mus.append(tools_scale(_.detach()))
                if return_add_noise:
                    _ = x + (1 - st) / st * mu + torch.randn_like(x) * sigma
                    noises.append(tools_scale(_.detach()))

            if return_add_mu:
                log["mu_shifting"] = self._get_denoise_row_from_list(mus, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
            if return_add_noise:
                log["mu_noise_shifting"] = self._get_denoise_row_from_list(noises,
                                                                           to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

        z = self.encode_first_stage(x)
        z_mu = self.encode_first_stage(mu)
        if return_reconstrcution:
            log["reconstructions"] = self.to_rgb_func(self.decode_first_stage(z.clone().detach()))
        if return_cond:
            log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples, others = self.sample(
                    c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate,
                    return_denoised=return_denoised, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = self.to_rgb_func(samples)
            if return_intermediate:
                log["intermediate"] = self._get_denoise_row_from_list(others['intermediates'],
                                                                      to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

            if return_denoised:
                log["denoised"] = self._get_denoise_row_from_list(others['denoiseds'],
                                                                  to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

        # if return_ideal_samples:
        #     assert self.ideal_sampler is not None, "Ideal sampler is not defined"
        #     with self.ema_scope("Plotting"):
        #         samples, others = self.sample(
        #             c, z_mu, shape=z.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate, return_denoised=return_denoised, ideal_sample=True, **sampling_kwargs
        #         )
        #     samples = self.decode_first_stage(samples)
        #     log["ideal_samples"] = samples
        #     if return_intermediate:
        #         log["ideal_intermediate"] = self._get_denoise_row_from_list(others['intermediates']) * 2.0 - 1.0

        #     if return_denoised:
        #         log["ideal_denoised"] = self._get_denoise_row_from_list(others['denoiseds']) * 2.0 - 1.0

        return log

    def _get_denoise_row_from_list(self, samples, desc='', to_rgb_func=None):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        if to_rgb_func != None:
            denoise_grid = to_rgb_func(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def _metric_to_01(self, x):
        """Convert image tensors to [0,1] for metrics without per-image min-max normalization.

        This avoids the evaluation bug where self.scale_01 can collapse two
        different tensors into the same normalized image for some Old cases.
        Supported input ranges: [-1,1], [0,1], and [0,255].
        """
        y = x.detach().float()
        if float(y.min()) < -0.05:
            y = (y + 1.0) / 2.0
        if float(y.max()) > 2.0:
            y = y / 255.0
        return y.clamp(0.0, 1.0)

    def _metric_batch_string(self, batch, key, default=""):
        if not isinstance(batch, dict) or key not in batch:
            return default
        value = batch[key]
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            value = value[0]
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return str(value.detach().cpu().item())
            return default
        return str(value)

    def _metric_norm_mode_for_batch(self, batch=None):
        """Choose metric normalization without changing training/sampling.

        - Sen2_MTC_New keeps the original project protocol: self.scale_01.
        - Sen2_MTC_Old uses fixed range conversion to avoid the Old-only
          scale_01 collapse bug observed in several RGB/JPG cases.

        You may override auto mode from YAML by adding to model.params:
            metric_norm_mode: scale01   # New / original EMRDM protocol
            metric_norm_mode: fixed01   # Old RGB/JPG protocol
        """
        mode = str(getattr(self, "metric_norm_mode", "auto")).lower()
        if mode in ["scale01", "fixed01"]:
            return mode

        dataset_name = self._metric_batch_string(batch, "dataset_name", "").lower()
        case_id = self._metric_batch_string(batch, "case_id", "")
        image_path = self._metric_batch_string(batch, getattr(self, "image_path_key", "image_path"), "")
        image_path = image_path or self._metric_batch_string(batch, "image_path", "")
        probe = " ".join([dataset_name, case_id, image_path]).lower()

        if "old" in probe:
            return "fixed01"
        if "new" in probe:
            return "scale01"
        # Sen2_MTC_New case ids in this project use tileXXX__...; Old uses bare MGRS ids.
        if case_id.startswith("tile") or "__" in case_id:
            return "scale01"
        return "fixed01"

    def _metric_tensor_for_metrics(self, x, batch=None):
        mode = self._metric_norm_mode_for_batch(batch)
        if mode == "scale01":
            return self.scale_01(x.detach().float())
        return self._metric_to_01(x)

    @torch.no_grad()
    def shared_test_step(self, batch, batch_idx=None):
        target = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )

        sampling_kwargs = {}
        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        if self.count_sample_time:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with self.ema_scope("Plotting"):
            samples, _ = self.sample(
                c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)
        if self.count_sample_time:
            end_event.record()
            torch.cuda.synchronize()
            sample_time = start_event.elapsed_time(end_event)
            self.log_dict({"sample_time": sample_time}, sync_dist=True, on_step=True, on_epoch=False)

        for i in range(samples.shape[0]):
            _target_raw = target[i, ...].clone().detach()
            _samples_raw = samples[i, ...].clone().detach()
            _target = self._metric_tensor_for_metrics(_target_raw, batch=batch)
            _samples = self._metric_tensor_for_metrics(_samples_raw, batch=batch)

            scale01_target = self.scale_01(_target_raw)
            scale01_samples = self.scale_01(_samples_raw)
            scale01_diff = (scale01_target - scale01_samples).abs()

            diff = (_target - _samples).abs()
            diff_mean = float(diff.mean().item())
            diff_max = float(diff.max().item())
            is_zero_diff = bool(diff_max == 0.0)

            case_id = f"test_{len(getattr(self, 'checked_test_metrics', [])):06d}"
            if isinstance(batch, dict):
                for key in ["case_id", "image_path", "path", "paths"]:
                    if key in batch:
                        value = batch[key]
                        if isinstance(value, (list, tuple)):
                            value = value[i] if len(value) > i else value[0]
                        elif isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item() if value.numel() == 1 else case_id
                        case_id = os.path.splitext(os.path.basename(str(value)))[0]
                        break

            if is_zero_diff:
                print(
                    "[WARN TEST ZERO DIFF]",
                    "case=", case_id,
                    "batch_idx=", batch_idx,
                    "i=", i,
                    "target_minmax=", float(_target.min()), float(_target.max()),
                    "sample_minmax=", float(_samples.min()), float(_samples.max()),
                )

            metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            checked_metrics = dict(metrics)
            checked_metrics["case_id"] = case_id
            checked_metrics["diff_mean"] = diff_mean
            checked_metrics["diff_max"] = diff_max
            checked_metrics["is_zero_diff"] = is_zero_diff
            checked_metrics["scale01_diff_mean"] = float(scale01_diff.mean().item())
            checked_metrics["scale01_diff_max"] = float(scale01_diff.max().item())
            checked_metrics["metric_norm_mode"] = self._metric_norm_mode_for_batch(batch)
            if not hasattr(self, "checked_test_metrics"):
                self.checked_test_metrics = []
            self.checked_test_metrics.append(checked_metrics)

            self.log_dict(metrics, sync_dist=True, batch_size=1, on_epoch=True)
            _mu = self._metric_tensor_for_metrics(mu[i, ...].clone().detach(), batch=batch)
            raw_metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_mu.unsqueeze(0))
            raw_metrics = {"raw_" + k: v for k, v in raw_metrics.items()}
            self.log_dict(raw_metrics, sync_dist=True, batch_size=1, on_epoch=True)
            self.avg_metrics.add(metrics)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        self.shared_test_step(batch=batch, batch_idx=batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.shared_test_step(batch=batch, batch_idx=batch_idx)

    # def on_train_start(self):
    #     # flops, params = thop.profile(self.model.diffusion_model, inputs=(torch.randn([1,28,256,256],device=self.device),\
    #     #     torch.randn(1,device=self.device),))
    #     # flops, params = thop.clever_format([flops, params], "%.3f")
    #     # print(flops, params)
    #     model_fwd = lambda: self.model.diffusion_model(torch.randn([1,28,256,256],device=self.device),\
    #         torch.randn(1,device=self.device))
    #     fwd_flops = measure_flops(self.model.diffusion_model,model_fwd)
    #     print(fwd_flops)

    @torch.no_grad()
    def on_predict_epoch_start(self, *args, **kwargs):
        self.all_pred_metrics = []

    @torch.no_grad()
    def on_predict_epoch_end(self, *args, **kwargs):
        metrics = {}
        for metric in self.all_pred_metrics:
            for k, v in metric.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        pd.DataFrame(metrics).to_csv(self.logger.save_dir + "/metrics.csv")

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        mu = self.get_input(batch, self.mean_key)
        assert mu.shape[0] == 1, "batch size should be 1."
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )

        sampling_kwargs = {}
        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        with self.ema_scope("Plotting"):
            samples, _ = self.sample(
                c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)

        path = self.logger.save_dir + "/sample/"
        os.makedirs(path, exist_ok=True)
        image_path = self.get_input(batch, "image_path")[0]
        image_path = image_path.split("/")[-1]
        _, image_path_extension = os.path.splitext(image_path)
        target = self.get_input(batch, self.input_key)
        target_raw_for_metric = target.clone().detach()
        samples_raw_for_metric = samples.clone().detach()
        target_path = image_path.replace(image_path_extension, "_target.png")
        target = self.scale_01(target)
        samples = self.scale_01(samples)
        target_rgb = np.moveaxis((self.to_rgb_func(target)[0] * 255).cpu().numpy().astype(np.uint8), 0, -1)
        Image.fromarray(target_rgb).save(path + target_path)
        rgb_path = image_path.replace(image_path_extension, ".png")
        rgb = np.moveaxis((self.to_rgb_func(samples)[0] * 255).cpu().numpy().astype(np.uint8), 0, -1)
        Image.fromarray(rgb).save(path + rgb_path)
        sample = samples[0].cpu().numpy()
        if image_path_extension == ".tif":
            with rasterio.open(path + image_path, 'w', driver='GTiff', height=sample.shape[1], width=sample.shape[2],
                               count=sample.shape[0], dtype=sample.dtype) as dst:
                dst.write(sample)
        mu_path = image_path.replace(image_path_extension, "_mu.png")
        mu = self.scale_01(mu)
        mu_rgb = np.moveaxis((self.to_rgb_func(mu)[0] * 255).cpu().numpy().astype(np.uint8), 0, -1)
        Image.fromarray(mu_rgb).save(path + mu_path)
        # calculate the metrics. Keep metric tensors isolated from visualization tensors.
        target_metric = self._metric_tensor_for_metrics(target_raw_for_metric, batch=batch)
        samples_metric = self._metric_tensor_for_metrics(samples_raw_for_metric, batch=batch)
        metric_diff = (target_metric - samples_metric).abs()
        if metric_diff.max().item() == 0:
            print(
                "[WARN METRIC ZERO DIFF]",
                "case=", image_path,
                "target_minmax=", target_metric.min().item(), target_metric.max().item(),
                "sample_minmax=", samples_metric.min().item(), samples_metric.max().item(),
            )
        metrics = self.img_metrics(target=target_metric, pred=samples_metric)
        metrics["image_path"] = image_path
        metrics["case_id"] = os.path.splitext(os.path.basename(image_path))[0]
        metrics["dataset_name"] = "unknown_dataset"
        metrics["metric_norm_mode"] = self._metric_norm_mode_for_batch(batch)
        self.all_pred_metrics.append(metrics)

    @torch.no_grad()
    def on_test_epoch_start(self, *args, **kwargs):
        self.avg_metrics.reset()
        self.checked_test_metrics = []

    @torch.no_grad()
    def on_test_epoch_end(self, *args, **kwargs):
        avg_metrics = self.avg_metrics.value()
        final_metrics = {}
        for k, v in avg_metrics.items():
            final_metrics["final_" + k] = v
        self.log_dict(final_metrics, sync_dist=True, on_epoch=True)

        if hasattr(self, "checked_test_metrics") and len(self.checked_test_metrics) > 0:
            pd.DataFrame(self.checked_test_metrics).to_csv(
                os.path.join(self.logger.save_dir, "per_image_test_metrics_checked.csv"),
                index=False,
            )

    @torch.no_grad()
    def on_validation_epoch_start(self, *args, **kwargs):
        self.avg_metrics.reset()

    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs):
        avg_metrics = self.avg_metrics.value()
        final_metrics = {}
        for k, v in avg_metrics.items():
            final_metrics["final_" + k] = v
        self.log_dict(final_metrics, sync_dist=True, on_epoch=True)


class TemporalResidualDiffusionEngine(ResidualDiffusionEngine):

    def __init__(
            self,
            mask_key=None,
            image_path_key="image_path",
            save_paper_vis: bool = True,
            vis_root: str = "vis_raw",
            vis_method_name: str = "ours",
            vis_color_mode: str = "auto",
            vis_save_legacy_sample: bool = True,
            vis_save_aux: bool = True,
            vis_save_attn: bool = True,
            vis_save_condition_slices: bool = False,
            vis_save_enhanced: bool = True,
            vis_enhance_low: float = 1.0,
            vis_enhance_high: float = 99.0,
            vis_mrh_mode: str = "abs_percentile",
            vis_debug_mrh: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_key = mask_key
        self.image_path_key = image_path_key

        # Paper-figure export controls. This path is independent of ImageLogger/log_images.
        # It writes case-wise files under: <logdir>/<vis_root>/<dataset_name>/<case_id>/.
        self.save_paper_vis = bool(save_paper_vis)
        self.vis_root = str(vis_root)
        self.vis_method_name = str(vis_method_name)
        self.vis_color_mode = str(vis_color_mode)  # auto | rgb | to_rgb
        self.vis_save_legacy_sample = bool(vis_save_legacy_sample)
        self.vis_save_aux = bool(vis_save_aux)
        self.vis_save_attn = bool(vis_save_attn)
        self.vis_save_condition_slices = bool(vis_save_condition_slices)
        self.vis_save_enhanced = bool(vis_save_enhanced)
        self.vis_enhance_low = float(vis_enhance_low)
        self.vis_enhance_high = float(vis_enhance_high)
        self.vis_mrh_mode = str(vis_mrh_mode)
        self.vis_debug_mrh = bool(vis_debug_mrh)

    def _get_temporal_denoise_row_from_list(self, samples, desc='', to_rgb_func=None):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b t c h w -> b t n c h w')
        denoise_grid = rearrange(denoise_grid, 'b t n c h w -> (b t n) c h w')
        if to_rgb_func != None:
            denoise_grid = to_rgb_func(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 8,
            sample: bool = True,
            ucg_keys: List[str] = None,
            return_intermediate: bool = False,
            return_denoised: bool = False,
            return_add_mu: bool = False,
            return_add_noise: bool = False,
            return_cond: bool = False,
            return_reconstrcution: bool = False,
            return_mask: bool = False,
            return_attn: bool = False,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)
        if return_mask:
            mask = self.get_input(batch, self.mask_key)
            if mask is not None:
                for i in range(mask.shape[1]):
                    log[f"mask_timestep{i}"] = mask[:N, i, ...].unsqueeze(dim=1) * 2.0 - 1.0

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        mu = mu.to(self.device)[:N]
        log["inputs"] = self.to_rgb_func(x.clone().detach())
        for i in range(mu.shape[1]):
            log[f"mean_timestep{i}"] = self.to_rgb_func(mu[:, i, ...].clone().detach())

        if return_add_mu or return_add_noise:
            sigmas = self.sampler.discretization(
                self.sampler.num_steps, device=self.device
            )
            for index in range(mu.shape[1]):
                _mu = mu[:, index, ...]
                mus = [tools_scale(x.clone().detach())] if return_add_mu else None
                noises = [tools_scale(x.clone().detach())] if return_add_noise else None

                for i in reversed(self.sampler.get_sigma_gen(self.sampler.num_steps)):
                    sigma = sigmas[i]
                    st = self.sigma2st(sigma)
                    if return_add_mu:
                        _ = x + (1 - st) / st * _mu
                        mus.append(tools_scale(_.detach()))
                    if return_add_noise:
                        _ = x + (1 - st) / st * _mu + torch.randn_like(x) * sigma
                        noises.append(tools_scale(_.detach()))

                if return_add_mu:
                    log[f"mu_shifting_timestep{index}"] = self._get_denoise_row_from_list(mus,
                                                                                          to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
                if return_add_noise:
                    log[f"mu_noise_shifting_timestep{index}"] = self._get_denoise_row_from_list(noises,
                                                                                                to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

        z = self.encode_first_stage(x)
        z_mu = self.encode_first_stage(mu)
        if return_reconstrcution:
            log["reconstructions"] = self.to_rgb_func(self.decode_first_stage(z.clone().detach()))
        if return_cond:
            log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            sampling_kwargs = {}
            with self.ema_scope("Plotting"):
                samples, others = self.sample(
                    c, z_mu, batch, shape=z_mu.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate,
                    return_denoised=return_denoised, return_attn=return_attn, **sampling_kwargs
                )
                samples = self.decode_first_stage(samples)
            log["samples"] = self.to_rgb_func(self.scale_01(samples) * 2.0 - 1.0)
            # log['samples'] = self.to_rgb_func(samples)
            if return_intermediate:
                log["intermediate"] = self._get_temporal_denoise_row_from_list(others['intermediates'],
                                                                               to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

            if return_denoised:
                log["denoised"] = self._get_denoise_row_from_list(others['denoiseds'],
                                                                  to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

            if return_attn:
                attns = others['attns']
                # n_heads, batch_size, t, h, w
                # 只看最后一步的attn
                attn = attns[-1]
                attn = attn.view(-1, attn.shape[2], 1, attn.shape[3], attn.shape[4])
                log["attn"] = self._get_denoise_row_from_list(attn, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0

        return log

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                    (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:
                        # image cond
                        xc = x[:n, ...]
                    elif x.dim() == 5:
                        # a list of images
                        xc = [x[:, i, ...] for i in range(x.shape[1])]
                    else:
                        xc = x
                        # raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                if isinstance(xc, list):
                    for i, _xc in enumerate(xc):
                        log[embedder.input_key + f"_timestep{i}"] = self.to_rgb_func(_xc)
                else:
                    log[embedder.input_key] = self.to_rgb_func(xc)
        return log

    @torch.no_grad()
    def shared_test_step(self, batch, batch_idx=None):
        target = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )
        sampling_kwargs = {}

        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        if self.count_sample_time:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with self.ema_scope("Plotting"):
            samples, _ = self.sample(
                c, z_mu, batch, shape=z_mu.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)
        if self.count_sample_time:
            end_event.record()
            torch.cuda.synchronize()
            sample_time = start_event.elapsed_time(end_event)
            self.log_dict({"sample_time": sample_time}, sync_dist=True, on_step=True, on_epoch=False)

        for i in range(samples.shape[0]):
            _target_raw = target[i, ...].clone().detach()
            _samples_raw = samples[i, ...].clone().detach()
            _target = self._metric_tensor_for_metrics(_target_raw, batch=batch)
            _samples = self._metric_tensor_for_metrics(_samples_raw, batch=batch)

            scale01_target = self.scale_01(_target_raw)
            scale01_samples = self.scale_01(_samples_raw)
            scale01_diff = (scale01_target - scale01_samples).abs()

            diff = (_target - _samples).abs()
            diff_mean = float(diff.mean().item())
            diff_max = float(diff.max().item())
            is_zero_diff = bool(diff_max == 0.0)

            case_id = f"test_{len(getattr(self, 'checked_test_metrics', [])):06d}"
            if isinstance(batch, dict):
                for key in ["case_id", self.image_path_key, "image_path", "path", "paths"]:
                    if key in batch:
                        value = batch[key]
                        if isinstance(value, (list, tuple)):
                            value = value[i] if len(value) > i else value[0]
                        elif isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item() if value.numel() == 1 else case_id
                        case_id = os.path.splitext(os.path.basename(str(value)))[0]
                        break

            if is_zero_diff:
                print(
                    "[WARN TEST ZERO DIFF]",
                    "case=", case_id,
                    "batch_idx=", batch_idx,
                    "i=", i,
                    "target_minmax=", float(_target.min()), float(_target.max()),
                    "sample_minmax=", float(_samples.min()), float(_samples.max()),
                )

            metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            checked_metrics = dict(metrics)
            checked_metrics["case_id"] = case_id
            checked_metrics["diff_mean"] = diff_mean
            checked_metrics["diff_max"] = diff_max
            checked_metrics["is_zero_diff"] = is_zero_diff
            checked_metrics["scale01_diff_mean"] = float(scale01_diff.mean().item())
            checked_metrics["scale01_diff_max"] = float(scale01_diff.max().item())
            checked_metrics["metric_norm_mode"] = self._metric_norm_mode_for_batch(batch)
            if not hasattr(self, "checked_test_metrics"):
                self.checked_test_metrics = []
            self.checked_test_metrics.append(checked_metrics)

            self.log_dict(metrics, sync_dist=True, batch_size=1, on_epoch=True)
            self.avg_metrics.add(metrics)
        # raw_metrics = img_metrics(target=target, pred=mu)
        # raw_metrics = {"raw_" + k:v for k, v in raw_metrics.items()}
        # self.log_dict(raw_metrics, sync_dist=True)

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            mu: torch.Tensor,
            batch: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            return_intermediate: bool = False,
            return_denoised: bool = False,
            return_attn: bool = False,
            ideal_sample=False,
            **kwargs,
    ):
        sampling_kwargs = {k: self.get_input(batch, k) for k in self.loss_fn.batch2model_keys}
        # skip_index = self.loss_fn.get_skip_index(batch)
        randn = torch.randn(batch_size, *shape).to(self.device)
        denoiser = lambda input, sigma, c, st, return_attn=False: self.denoiser(
            self.model, input, sigma, c, st, return_attn, **sampling_kwargs, **kwargs
        )
        samples = self.sampler(denoiser, randn, mu, cond, uc=uc, return_intermediate=return_intermediate,
                               return_denoised=return_denoised, return_attn=return_attn)
        return samples

    # ---------------------------------------------------------------------
    # Paper visualization helpers
    # ---------------------------------------------------------------------
    def _paper_vis_scalar(self, value, default="unknown"):
        """Return a clean Python string from a collated batch field."""
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            return self._paper_vis_scalar(value[0], default=default)
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return str(value.detach().cpu().item())
            return default
        return str(value)

    def _paper_vis_get_string(self, batch, key, default="unknown"):
        if key not in batch:
            return default
        return self._paper_vis_scalar(batch[key], default=default)

    def _paper_vis_dataset_name(self, batch):
        name = self._paper_vis_get_string(batch, "dataset_name", default="unknown_dataset")
        # Protect against accidentally using module/file names in the dataset field.
        name = name.replace(".py", "")
        if name in ["unknown", "none", "None", ""]:
            name = "unknown_dataset"
        return name

    def _paper_vis_case_id(self, batch):
        case_id = self._paper_vis_get_string(batch, "case_id", default="")
        if case_id:
            return os.path.splitext(os.path.basename(case_id))[0]

        # Fall back to image_path_key, then path, then image_path.
        for key in [self.image_path_key, "path", "paths", "image_path"]:
            if key in batch:
                v = self._paper_vis_scalar(batch[key], default="")
                if v:
                    return os.path.splitext(os.path.basename(v))[0]
        return "case_%06d" % int(getattr(self, "global_step", 0))

    def _paper_vis_use_to_rgb(self, dataset_name):
        mode = str(getattr(self, "vis_color_mode", "auto")).lower()
        if mode == "to_rgb":
            return True
        if mode == "rgb":
            return False
        # auto: Old is ordinary RGB/JPG and must not go through Sentinel-specific to_rgb_func.
        # New also stores RGB first-three channels in the current dataloader, so direct RGB is safe.
        # Keep this branch explicit for future extension.
        return False

    def _save_rgb_tensor_png(self, tensor, path, normalize=False, use_to_rgb=False):
        """Save natural RGB tensors robustly for both Sen2_MTC_New and Sen2_MTC_Old.

        For gt/cloudy/prediction: normalize=False. The function converts [-1,1]
        or [0,1] RGB tensors to uint8 PNG. It does NOT use self.to_rgb_func unless
        use_to_rgb=True is explicitly requested.

        For MRH/attention/error-like maps: normalize=True.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        x = tensor.detach().float()

        # Remove batch dimension if present.
        if x.dim() == 4:
            x = x[0]
        if x.dim() != 3:
            raise RuntimeError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(x.shape)} for {path}")

        # Optional original project RGB conversion, mainly for legacy Sentinel-2 configs.
        if use_to_rgb and (not normalize):
            y = x.unsqueeze(0).to(self.device)
            y = self.scale_01(y)
            y = self.to_rgb_func(y)[0].detach().float().cpu()
            if y.dim() == 3 and y.shape[0] >= 3:
                y = y[:3]
            elif y.dim() == 3 and y.shape[0] == 1:
                y = y.repeat(3, 1, 1)
            y = y.clamp(0.0, 1.0)
            arr = (y.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(arr).save(path)
            return

        x = x.cpu()
        if x.shape[0] >= 3:
            x = x[:3]
        elif x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        else:
            raise RuntimeError(f"Unsupported channel number {x.shape[0]} for {path}")

        if normalize:
            xmin = x.amin(dim=(1, 2), keepdim=True)
            xmax = x.amax(dim=(1, 2), keepdim=True)
            x = (x - xmin) / (xmax - xmin + 1e-6)
        else:
            # Most project dataloaders return natural images in [-1,1].
            if float(x.min()) < -0.05:
                x = (x + 1.0) / 2.0
            # Defensive support for [0,255].
            if float(x.max()) > 2.0:
                x = x / 255.0

        x = x.clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(path)

    def _save_rgb_tensor_png_enhanced(self, tensor, path, low=None, high=None, use_to_rgb=False):
        """Save a contrast-enhanced copy for visual inspection only.

        This file is NOT meant for metric computation. It is useful for Old where
        original 8-bit RGB images can be low-contrast. The original gt.png / ours.png
        remain unenhanced and should be used for metrics.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        low = self.vis_enhance_low if low is None else float(low)
        high = self.vis_enhance_high if high is None else float(high)

        x = tensor.detach().float()
        if x.dim() == 4:
            x = x[0]
        if x.dim() != 3:
            raise RuntimeError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(x.shape)} for {path}")

        if use_to_rgb:
            y = x.unsqueeze(0).to(self.device)
            y = self.scale_01(y)
            x = self.to_rgb_func(y)[0].detach().float().cpu()
        else:
            x = x.cpu()
            if x.shape[0] >= 3:
                x = x[:3]
            elif x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            else:
                raise RuntimeError(f"Unsupported channel number {x.shape[0]} for {path}")
            if float(x.min()) < -0.05:
                x = (x + 1.0) / 2.0
            if float(x.max()) > 2.0:
                x = x / 255.0

        x = x.clamp(0.0, 1.0)
        flat = x.flatten(1)
        lo = torch.quantile(flat, low / 100.0, dim=1).view(-1, 1, 1)
        hi = torch.quantile(flat, high / 100.0, dim=1).view(-1, 1, 1)
        x = ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    def _save_residual_tensor_png(self, tensor, path, mode=None):
        """Save MRH/residual tensors with residual-aware normalization.

        mode="abs_percentile" shows residual magnitude using robust percentiles.
        mode="signed" centers zero residual at 0.5 and shows positive/negative signs.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = self.vis_mrh_mode if mode is None else mode
        x = tensor.detach().float().cpu()

        # Accept [B,C,H,W] / [C,H,W]. Temporal handling is done outside.
        if x.dim() == 4:
            x = x[0]
        if x.dim() != 3:
            raise RuntimeError(f"Expected residual [C,H,W], got {tuple(x.shape)} for {path}")

        if x.shape[0] >= 3:
            x = x[:3]
        elif x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        else:
            raise RuntimeError(f"Unsupported residual channel number {x.shape[0]} for {path}")

        if mode == "signed":
            scale = torch.quantile(x.abs().flatten(), 0.99).clamp(min=1e-6)
            x = (x / scale).clamp(-1.0, 1.0) * 0.5 + 0.5
        elif mode == "abs_percentile":
            x = x.abs()
            lo = torch.quantile(x.flatten(), 0.01)
            hi = torch.quantile(x.flatten(), 0.99).clamp(min=lo + 1e-6)
            x = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        elif mode == "minmax":
            xmin = x.amin(dim=(1, 2), keepdim=True)
            xmax = x.amax(dim=(1, 2), keepdim=True)
            x = ((x - xmin) / (xmax - xmin + 1e-6)).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported vis_mrh_mode={mode}")

        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    def _save_temporal_residual_tensor(self, tensor, case_dir, prefix):
        x = tensor.detach()
        if x.dim() == 5:
            x = x[0]  # [T,C,H,W]
        if x.dim() == 4:
            # Usually [T,C,H,W]. If B dimension survived with B=1, this also works as one image.
            for t in range(x.shape[0]):
                self._save_residual_tensor_png(x[t], os.path.join(case_dir, f"{prefix}_t{t}.png"))
        elif x.dim() == 3:
            self._save_residual_tensor_png(x, os.path.join(case_dir, f"{prefix}.png"))
        else:
            raise RuntimeError(f"Unsupported residual tensor shape {tuple(x.shape)} for {prefix}")

    def _write_tensor_stats(self, tensor, path, name):
        try:
            x = tensor.detach().float().cpu()
            row = {
                "name": name,
                "shape": str(tuple(x.shape)),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
                "abs_mean": float(x.abs().mean()),
            }
            pd.DataFrame([row]).to_csv(path, index=False)
        except Exception as e:
            print(f"[WARN] failed to write tensor stats for {name}: {e}")

    def _save_gray_tensor_png(self, tensor, path, normalize=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        x = tensor.detach().float().cpu()
        while x.dim() > 2:
            x = x[0]
        if normalize:
            x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        x = x.clamp(0.0, 1.0)
        arr = (x.numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)

    def _save_temporal_rgb_tensor(self, tensor, case_dir, prefix, normalize=False, use_to_rgb=False):
        x = tensor.detach()
        if x.dim() == 5:
            x = x[0]  # [T,C,H,W]
        if x.dim() == 4:
            for t in range(x.shape[0]):
                self._save_rgb_tensor_png(
                    x[t],
                    os.path.join(case_dir, f"{prefix}_t{t}.png"),
                    normalize=normalize,
                    use_to_rgb=use_to_rgb,
                )
        elif x.dim() == 3:
            self._save_rgb_tensor_png(
                x,
                os.path.join(case_dir, f"{prefix}.png"),
                normalize=normalize,
                use_to_rgb=use_to_rgb,
            )
        else:
            raise RuntimeError(f"Unsupported temporal tensor shape {tuple(x.shape)} for {prefix}")

    def _save_attention_maps(self, others, case_dir, spatial_size=None):
        if not isinstance(others, dict) or "attns" not in others:
            return
        attns = others.get("attns")
        if not attns:
            return
        attn = attns[-1].detach().float().cpu()

        # Expected common shape: [heads, B, T, H, W]. Other variants are handled defensively.
        if attn.dim() == 5:
            # Average over heads and take B=0 -> [T,H,W]
            if attn.shape[1] == 1:
                attn = attn.mean(dim=0)[0]
            else:
                attn = attn[:, 0].mean(dim=0)
        elif attn.dim() == 4:
            # [B,T,H,W] or [heads,T,H,W]
            attn = attn[0]
        elif attn.dim() == 3:
            pass
        else:
            return

        if attn.dim() != 3:
            return

        for t in range(attn.shape[0]):
            a = attn[t]
            if spatial_size is not None and tuple(a.shape[-2:]) != tuple(spatial_size):
                a = torch.nn.functional.interpolate(
                    a[None, None],
                    size=spatial_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            self._save_gray_tensor_png(a, os.path.join(case_dir, f"attn_t{t}.png"), normalize=True)

    def _save_paper_case(self, batch, target, mu, samples, others, c):
        dataset_name = self._paper_vis_dataset_name(batch)
        case_id = self._paper_vis_case_id(batch)
        case_dir = os.path.join(self.logger.save_dir, self.vis_root, dataset_name, case_id)
        os.makedirs(case_dir, exist_ok=True)

        use_to_rgb = self._paper_vis_use_to_rgb(dataset_name)

        # Natural images: direct RGB inverse normalization for both Old and New by default.
        # These unenhanced PNGs should be used for metrics.
        # Use exactly the same normalization domain as quantitative metrics.
        target_metric = self._metric_tensor_for_metrics(
            target.clone().detach(),
            batch=batch,
        )
        samples_metric = self._metric_tensor_for_metrics(
            samples.clone().detach(),
            batch=batch,
        )

        self._save_rgb_tensor_png(
            target_metric[0],
            os.path.join(case_dir, "gt.png"),
            normalize=False,
            use_to_rgb=False,
        )

        self._save_rgb_tensor_png(
            samples_metric[0],
            os.path.join(case_dir, f"{self.vis_method_name}.png"),
            normalize=False,
            use_to_rgb=False,
        )
        self._save_temporal_rgb_tensor(mu, case_dir, "cloudy", normalize=False, use_to_rgb=use_to_rgb)

        # Optional contrast-enhanced copies for human inspection / paper layout.
        # Never use *_vis.png for metrics.
        if self.vis_save_enhanced:
            self._save_rgb_tensor_png_enhanced(target[0], os.path.join(case_dir, "gt_vis.png"), use_to_rgb=use_to_rgb)
            self._save_rgb_tensor_png_enhanced(samples[0], os.path.join(case_dir, f"{self.vis_method_name}_vis.png"),
                                               use_to_rgb=use_to_rgb)
            if mu.dim() == 5:
                for t in range(mu.shape[1]):
                    self._save_rgb_tensor_png_enhanced(mu[0, t], os.path.join(case_dir, f"cloudy_t{t}_vis.png"),
                                                       use_to_rgb=use_to_rgb)
            elif mu.dim() == 4:
                self._save_rgb_tensor_png_enhanced(mu[0], os.path.join(case_dir, "cloudy_t0_vis.png"),
                                                   use_to_rgb=use_to_rgb)

        if self.vis_save_aux:
            if "soft_prior" in batch:
                self._save_temporal_rgb_tensor(batch["soft_prior"], case_dir, "soft_prior", normalize=False,
                                               use_to_rgb=use_to_rgb)
            if "mrh" in batch:
                self._save_temporal_residual_tensor(batch["mrh"], case_dir, "mrh")
                if self.vis_debug_mrh:
                    self._write_tensor_stats(batch["mrh"], os.path.join(case_dir, "mrh_stats.csv"), "batch_mrh")
            elif isinstance(c, dict) and "concat" in c and isinstance(c["concat"], torch.Tensor):
                # Current condition-only MRH layout: cloud RGBIR(4) + hint RGBIR(4).
                cond = c["concat"].detach()
                if cond.dim() == 5 and cond.shape[2] >= 7:
                    mrh_from_cond = cond[:, :, 4:7]
                    self._save_temporal_residual_tensor(mrh_from_cond, case_dir, "mrh")
                    if self.vis_debug_mrh:
                        self._write_tensor_stats(mrh_from_cond, os.path.join(case_dir, "mrh_stats.csv"), "cond_4_7")

        if self.vis_save_condition_slices and isinstance(c, dict) and "concat" in c and isinstance(c["concat"],
                                                                                                   torch.Tensor):
            cond = c["concat"].detach()
            if cond.dim() == 5:
                if cond.shape[2] >= 3:
                    self._save_temporal_rgb_tensor(cond[:, :, :3], case_dir, "cond_rgb", normalize=False,
                                                   use_to_rgb=use_to_rgb)
                if cond.shape[2] >= 4:
                    self._save_temporal_rgb_tensor(cond[:, :, 3:4], case_dir, "cond_ir", normalize=True,
                                                   use_to_rgb=False)

        if self.vis_save_attn:
            h, w = target.shape[-2], target.shape[-1]
            self._save_attention_maps(others, case_dir, spatial_size=(h, w))

        # Small metadata file for later debugging and method-output alignment.
        meta = {
            "dataset_name": dataset_name,
            "case_id": case_id,
            "method_name": self.vis_method_name,
            "source_path": self._paper_vis_get_string(batch, self.image_path_key, default=""),
        }
        try:
            pd.DataFrame([meta]).to_csv(os.path.join(case_dir, "meta.csv"), index=False)
        except Exception:
            pass
        return case_dir, case_id

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        # Supports case-wise paper visualization for Sen2_MTC_New and Sen2_MTC_Old.
        # Use batch_size=1 for clean case folders. The function keeps legacy sample/ output optional.
        mu = self.get_input(batch, self.mean_key)
        if mu.shape[0] != 1:
            raise AssertionError("batch size should be 1 for paper visualization export.")

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )

        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        with self.ema_scope("Plotting"):
            samples, others = self.sample(
                c,
                z_mu,
                batch,
                shape=z_mu.shape[1:],
                uc=uc,
                batch_size=N,
                return_attn=bool(self.vis_save_attn),
            )
            samples = self.decode_first_stage(samples)

        target = self.get_input(batch, self.input_key)

        if self.save_paper_vis:
            self._save_paper_case(batch, target, mu, samples, others, c)

        # Optional legacy output compatible with older scripts. Use direct RGB to avoid Old-dataset color failure.
        image_path = self._paper_vis_get_string(batch, self.image_path_key,
                                                default=self._paper_vis_case_id(batch) + ".png")
        image_name = os.path.basename(image_path)
        image_stem, image_ext = os.path.splitext(image_name)
        if image_ext == "":
            image_ext = ".png"

        if self.vis_save_legacy_sample:
            path = os.path.join(self.logger.save_dir, "sample")
            os.makedirs(path, exist_ok=True)
            dataset_name = self._paper_vis_dataset_name(batch)
            use_to_rgb = self._paper_vis_use_to_rgb(dataset_name)
            self._save_rgb_tensor_png(target[0], os.path.join(path, f"{image_stem}_target.png"), normalize=False,
                                      use_to_rgb=use_to_rgb)
            self._save_rgb_tensor_png(samples[0], os.path.join(path, f"{image_stem}.png"), normalize=False,
                                      use_to_rgb=use_to_rgb)
            if mu.dim() == 5:
                for t in range(mu.shape[1]):
                    self._save_rgb_tensor_png(mu[0, t], os.path.join(path, f"{image_stem}_timestep{t}.png"),
                                              normalize=False, use_to_rgb=use_to_rgb)

            # Keep original GeoTIFF output for tif cases when possible.
            if image_ext.lower() in [".tif", ".tiff"]:
                try:
                    sample_np = self.scale_01(samples)[0].detach().cpu().numpy()
                    with rasterio.open(
                            os.path.join(path, image_name),
                            "w",
                            driver="GTiff",
                            height=sample_np.shape[1],
                            width=sample_np.shape[2],
                            count=sample_np.shape[0],
                            dtype=sample_np.dtype,
                    ) as dst:
                        dst.write(sample_np)
                except Exception as e:
                    print(f"[WARN] failed to write legacy GeoTIFF for {image_name}: {e}")

        # Calculate metrics on [0,1], consistent with shared_test_step.
        # Metric tensors are cloned/detached to prevent any visualization branch from
        # contaminating or aliasing the tensors used for quantitative evaluation.
        target_metric = self._metric_tensor_for_metrics(target.clone().detach(), batch=batch)
        samples_metric = self._metric_tensor_for_metrics(samples.clone().detach(), batch=batch)
        case_id = self._paper_vis_case_id(batch)
        dataset_name = self._paper_vis_dataset_name(batch)
        metric_diff = (target_metric - samples_metric).abs()
        if metric_diff.max().item() == 0:
            print(
                "[WARN METRIC ZERO DIFF]",
                "case=", case_id,
                "target_minmax=", target_metric.min().item(), target_metric.max().item(),
                "sample_minmax=", samples_metric.min().item(), samples_metric.max().item(),
            )
        metrics = self.img_metrics(target=target_metric, pred=samples_metric)
        metrics["image_path"] = image_name
        metrics["case_id"] = case_id
        metrics["dataset_name"] = dataset_name
        metrics["metric_norm_mode"] = self._metric_norm_mode_for_batch(batch)
        self.all_pred_metrics.append(metrics)