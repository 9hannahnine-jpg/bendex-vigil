"""
Vigil — automated three-phase training correction.

This module implements the automated intervention system:
  Phase 1 — Freeze + LR reduction
  Phase 2 — Escalation
  Phase 3 — Permanent quarantine

Copyright (c) 2026 Hannah Nine / Bendex Geometry LLC
Licensed under the Bendex Source Available License (see LICENSE).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BendexIntervention:
    """
    Automated intervention system for Vigil.

    Attach to a BendexMonitor and call step() after each observe() call.
    If an event is detected, intervention fires automatically.

    Usage::

        monitor = BendexMonitor(model)
        intervention = BendexIntervention(model, optimizer)

        for step, batch in enumerate(dataloader):
            event = monitor.observe(step)
            intervention.step(event, step)
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_factor: float = 0.15,
        grad_clip: float = 0.8,
        lr_cap_steps: int = 50,
        quarantine_steps: int = 9999,
        escalation_min_steps: int = 20,
        escalation_lr_factor: float = 0.20,
        logit_penalty: float = 0.0012,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_factor = lr_factor
        self.grad_clip = grad_clip
        self.lr_cap_steps = lr_cap_steps
        self.quarantine_steps = quarantine_steps
        self.escalation_min_steps = escalation_min_steps
        self.escalation_lr_factor = escalation_lr_factor
        self.logit_penalty = logit_penalty

        self._intervened = False
        self._intervened_at: Optional[int] = None
        self._quarantine_until = -1
        self._lr_cap_until = -1
        self._base_lr: Optional[float] = None
        self._escalated = False
        self._frozen_params: List[str] = []

        self.intervention_log: List[Dict] = []

    def step(self, event: Optional[Dict], step: int) -> None:
        """Call after monitor.observe(). Fires intervention if event is not None."""
        if event is not None and not self._intervened and step >= self._quarantine_until:
            self._primary_intervention(event, step)

        # Escalation check
        if (
            self._intervened
            and not self._escalated
            and self._intervened_at is not None
            and step >= self._intervened_at + self.escalation_min_steps
        ):
            self._check_escalation(step)

        # Apply LR cap if active
        if self._lr_cap_until > 0 and step <= self._lr_cap_until and self._base_lr:
            for pg in self.optimizer.param_groups:
                pg["lr"] = min(pg["lr"], self._base_lr * self.lr_factor)

    def logit_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Returns logit variance penalty term.
        Add to task loss after primary intervention fires:
            loss = task_loss + monitor.intervention.logit_loss(logits)
        """
        if not self._intervened:
            return torch.tensor(0.0, device=logits.device)
        return self.logit_penalty * logits.float().pow(2).mean()

    # ------------------------------------------------------------------

    def _primary_intervention(self, event: Dict, step: int) -> None:
        self._intervened = True
        self._intervened_at = step
        self._quarantine_until = step + self.quarantine_steps

        # Save base LR
        self._base_lr = self.optimizer.param_groups[0]["lr"]

        # Freeze all monitored parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param.requires_grad_(False)
            self._frozen_params.append(name)

        # Reduce LR
        for pg in self.optimizer.param_groups:
            pg["lr"] *= self.lr_factor

        # Set LR cap
        self._lr_cap_until = step + self.lr_cap_steps

        self.intervention_log.append({
            "phase": 1,
            "step": step,
            "event": event,
            "frozen_params": len(self._frozen_params),
            "lr_after": self.optimizer.param_groups[0]["lr"],
        })

    def _check_escalation(self, step: int) -> None:
        self._escalated = True

        # Further reduce LR
        for pg in self.optimizer.param_groups:
            pg["lr"] *= self.escalation_lr_factor

        self.intervention_log.append({
            "phase": 2,
            "step": step,
            "lr_after": self.optimizer.param_groups[0]["lr"],
        })

    def apply_grad_clip(self) -> None:
        """Call after loss.backward() and before optimizer.step()."""
        if self._intervened:
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grad_clip,
            )
