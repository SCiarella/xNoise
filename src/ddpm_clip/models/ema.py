"""
Exponential Moving Average (EMA) for model weights.

This module provides a utility class for maintaining an exponential moving average
of model parameters, which is useful for stabilizing generation quality in diffusion models.
"""

import torch
import torch.nn as nn
from typing import Dict


class EMA:
    """
    Exponential Moving Average for model weights.

    This class maintains a shadow copy of model parameters that is updated using
    exponential moving average. This can improve generation quality and stability.

    Args:
        model: The PyTorch model to track
        decay: The decay factor for the moving average (default: 0.9999)

    Example:
        >>> model = MyModel()
        >>> ema = EMA(model, decay=0.9999)
        >>> # During training
        >>> loss.backward()
        >>> optimizer.step()
        >>> ema.update()
        >>> # For generation
        >>> ema.apply_shadow()
        >>> generate(model)
        >>> ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize the EMA tracker.

        Args:
            model: The PyTorch model to track
            decay: The decay factor for the moving average (default: 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register()

    def register(self):
        """Register all trainable parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update the EMA parameters with current model parameters.

        This should be called after each optimizer step during training.
        The formula is: shadow = decay * shadow + (1 - decay) * param
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay
                               ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Replace model weights with EMA weights.

        This backs up the current model parameters and replaces them with
        the EMA shadow parameters. Call restore() to revert.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model weights.

        This reverts the model parameters to their state before apply_shadow()
        was called.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
