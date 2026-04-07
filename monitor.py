"""
Vigil — core implementation.

Per-module normalized weight divergence, discrete trajectory curvature,
z-scored dual-channel detection, deepest-early-cluster attribution.

Copyright (c) 2026 Hannah Nine / Bendex Geometry LLC
Licensed under the Bendex Source Available License (see LICENSE).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BendexConfig:
    """All tunable parameters for the Vigil monitor."""

    def __init__(
        self,
        warmup_steps: int = 60,
        z_threshold: float = 2.5,
        persist: int = 2,
        max_lag: int = 6,
        eps: float = 1e-8,
    ):
        self.warmup_steps = warmup_steps
        self.z_threshold = z_threshold
        self.persist = persist
        self.max_lag = max_lag
        self.eps = eps


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _module_divergence(
    params: Dict[str, torch.Tensor],
    ref: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total and per-module normalized L2 weight divergence.

    div(M) = sum_i ||w_i(t) - w_i_ref||_2 / sqrt(sum_i n_i)

    Returns (total_divergence, {module_name: divergence}).
    """
    total = 0.0
    per_module: Dict[str, float] = {}

    for name, tensor in params.items():
        if name not in ref:
            continue
        diff = (tensor - ref[name]).float()
        l2 = diff.norm().item()
        n = diff.numel()
        norm_div = l2 / (math.sqrt(n) + eps)
        per_module[name] = norm_div
        total += norm_div

    return total, per_module


def _grad_energy(model: nn.Module, monitored: List[str], eps: float = 1e-8) -> float:
    """Compute normalized total gradient energy across monitored parameters."""
    total_sq = 0.0
    total_n = 0
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in monitored) and param.grad is not None:
            total_sq += param.grad.float().pow(2).sum().item()
            total_n += param.grad.numel()
    if total_n == 0:
        return 0.0
    return total_sq / (total_n + eps)


def _zscore(value: float, buf: deque, eps: float = 1e-8) -> float:
    if len(buf) < 2:
        return 0.0
    mu = sum(buf) / len(buf)
    var = sum((x - mu) ** 2 for x in buf) / len(buf)
    sigma = math.sqrt(var) + eps
    return (value - mu) / sigma


def _first_persistent(seq: List[float], threshold: float, persist: int) -> Optional[int]:
    """
    Return the earliest index i such that seq[i], seq[i+1], ..., seq[i+persist-1]
    all exceed threshold. Returns None if no such index exists.
    """
    count = 0
    start = None
    for i, v in enumerate(seq):
        if v > threshold:
            if count == 0:
                start = i
            count += 1
            if count >= persist:
                return start
        else:
            count = 0
            start = None
    return None


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class BendexMonitor:
    """
    Real-time neural network training stability monitor.

    Usage::

        monitor = BendexMonitor(model)

        for step, batch in enumerate(dataloader):
            monitor.observe(step)          # call BEFORE forward pass
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(monitor.events)              # list of detected instability events
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[BendexConfig] = None,
        monitored_modules: Optional[List[str]] = None,
        device: str = 'cpu',
    ):
        self.model = model
        self.config = config or BendexConfig()
        self.device = device

        if monitored_modules is None:
            self.monitored = [
                name for name, _ in model.named_modules()
                if name and '.' not in name
            ]
        else:
            self.monitored = monitored_modules

        self._ref: Dict[str, torch.Tensor] = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }

        self._div_raw: List[float] = []
        self._div_hist: List[float] = []
        self._grad_hist: List[float] = []
        self._kappa_hist: List[float] = []

        self._module_div_hist: Dict[str, List[float]] = {m: [] for m in self.monitored}

        self._div_baseline: deque = deque(maxlen=self.config.warmup_steps)
        self._grad_baseline: deque = deque(maxlen=self.config.warmup_steps)
        self._kappa_baseline: deque = deque(maxlen=self.config.warmup_steps)

        self._step = 0
        self._warmup_done = False
        self._intervened = False
        self._quarantine_until = -1

        self.events: List[Dict] = []

    def observe(self, step: int) -> Optional[Dict]:
        """
        Call once per training step, BEFORE the forward pass.
        Returns an event dict if instability is detected, else None.
        """
        self._step = step

        current_params = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
        }
        total_div, per_module = _module_divergence(
            current_params, self._ref, self.config.eps
        )
        grad_e = _grad_energy(self.model, self.monitored, self.config.eps)

        self._div_raw.append(total_div)

        for m in self.monitored:
            val = sum(v for k, v in per_module.items()
                      if k.startswith(m + '.') or k == m)
            self._module_div_hist[m].append(val)

        if len(self._div_raw) >= 3:
            kappa = abs(
                self._div_raw[-1]
                - 2 * self._div_raw[-2]
                + self._div_raw[-3]
            )
        else:
            kappa = 0.0

        if len(self._div_raw) >= 2:
            step_change = abs(self._div_raw[-1] - self._div_raw[-2])
        else:
            step_change = 0.0

        if step < self.config.warmup_steps:
            self._div_baseline.append(step_change)
            self._grad_baseline.append(grad_e)
            self._kappa_baseline.append(kappa)
            self._div_hist.append(0.0)
            self._grad_hist.append(0.0)
            self._kappa_hist.append(0.0)
            return None

        if not self._warmup_done:
            self._warmup_done = True

        z_div   = _zscore(step_change, self._div_baseline,   self.config.eps)
        z_grad  = _zscore(grad_e,      self._grad_baseline,  self.config.eps)
        z_kappa = _zscore(kappa,        self._kappa_baseline, self.config.eps)

        self._div_hist.append(z_div)
        self._grad_hist.append(z_grad)
        self._kappa_hist.append(z_kappa)

        if self._intervened or step < self._quarantine_until:
            return None

        event = self._check_detection(step)
        if event:
            self.events.append(event)
        return event

    def reset_reference(self) -> None:
        """Update the reference checkpoint to current model weights."""
        self._ref = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }

    def _check_detection(self, step: int) -> Optional[Dict]:
        Z = self.config.z_threshold
        P = self.config.persist

        and_trigger = (
            _first_persistent(self._div_hist,   Z, P) is not None
            and _first_persistent(self._grad_hist, Z, P) is not None
        )
        kappa_trigger = _first_persistent(self._kappa_hist, Z, P) is not None

        if not (and_trigger or kappa_trigger):
            return None

        mode = 'kappa' if kappa_trigger else 'and'
        suspect = self._attribute(step)

        return {
            'step':             step,
            'mode':             mode,
            'suspect_module':   suspect,
            'z_div':            self._div_hist[-1],
            'z_grad':           self._grad_hist[-1],
            'z_kappa':          self._kappa_hist[-1],
            'total_divergence': self._div_raw[-1],
        }

    def _attribute(self, step: int) -> Optional[str]:
        """
        Deepest-early-cluster attribution.

        From all modules whose per-module divergence z-score first crossed
        threshold within max_lag steps of the earliest crossing, select
        the deepest (last in ordered module list).
        """
        Z   = self.config.z_threshold
        P   = self.config.persist
        lag = self.config.max_lag

        crossings: Dict[str, int] = {}
        for m in self.monitored:
            hist = self._module_div_hist[m]
            if len(hist) < self.config.warmup_steps:
                continue
            warmup = hist[:self.config.warmup_steps]
            mu  = sum(warmup) / len(warmup)
            var = sum((x - mu) ** 2 for x in warmup) / len(warmup)
            sigma  = math.sqrt(var) + self.config.eps
            z_hist = [(v - mu) / sigma for v in hist[self.config.warmup_steps:]]
            idx = _first_persistent(z_hist, Z, P)
            if idx is not None:
                crossings[m] = idx

        if not crossings:
            return self.monitored[-1] if self.monitored else None

        earliest = min(crossings.values())
        cluster  = [m for m, c in crossings.items() if c <= earliest + lag]

        for m in reversed(self.monitored):
            if m in cluster:
                return m

        return cluster[-1] if cluster else None
