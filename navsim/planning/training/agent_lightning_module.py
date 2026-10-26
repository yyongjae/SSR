import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import os
import torch
from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        features, targets = batch

        input_target = self.agent.config.input_target if hasattr(self.agent.config, 'input_target') else False
        if input_target:
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)

        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, Tensor):
            loss_dict = {"traj_loss": loss_dict}
            
        total_loss = 0.0
        # agent.config.log_sync_dist = False drops the per-micro-batch all-reduce (default True = unchanged)
        sync_dist = bool(getattr(getattr(self.agent, "config", None), "log_sync_dist", True))
        for loss_key, loss_value in loss_dict.items():
            self.log(f"{logging_prefix}/{loss_key}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=sync_dist)
            if 'acc' in loss_key:
                continue
            total_loss = total_loss + loss_value
        return total_loss
    
    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        if os.environ.get("KM_NANDBG"):                       # km 10/24: NaN hunt (off unless set)
            self._km_batch = batch
            self._km_rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()
    
    def backward(self, loss):
        # print('set detect anomaly')
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        if os.environ.get("KM_NANDBG"):                       # first non-finite gradient: dump the inputs and the
            bad = [n for n, q in self.named_parameters()      # weights (not stepped yet) for an offline replay, then stop
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad or not torch.isfinite(loss.detach()) or os.environ.get("KM_NANDBG_FORCE"):   # FORCE: test the dump
                out = os.path.join(os.environ["KM_NANDBG"], f"nan_rank{self.global_rank}_step{self.global_step}.pt")
                print(f"KM_NANDBG rank {self.global_rank} step {self.global_step} loss {float(loss)}: "
                      f"{len(bad)} non-finite grads: {bad}", flush=True)
                torch.save(dict(batch=self._km_batch, sd=self.state_dict(), rng=self._km_rng, bad=bad,
                                loss=loss.detach().cpu(), step=self.global_step), out)   # sd holds a non-tensor entry
                raise RuntimeError(f"KM_NANDBG non-finite grad ({len(bad)} params, e.g. {bad[:5]}) -> {out}")
