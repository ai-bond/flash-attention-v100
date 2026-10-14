# ======================================================================================
# Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch
import torch.nn as nn

from flash_attn_v100.cross_entropy import cross_entropy_loss

class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index     = -100,
        reduction        = "mean",
        label_smoothing  = 0.0,
        logit_scale      = 1.0,
        lse_square_scale = 0.0,
        return_z_loss    = False,
    ):
        """
        Arguments:
            ignore_index: int. If labels == ignore_index, loss for that row is 0.0
            reduction: "mean" | "none" | "sum"
            label_smoothing: float in [0, 1)
            logit_scale: float > 0. Scale for logits before softmax
            lse_square_scale: float. If > 0, adds lse_square_scale * lse^2 (z-loss)
            return_z_loss: bool. If True, additionally returns z_loss (for logging only)

        Note: Module requires 'weight' tensor in forward because it fuses final linear projection with the loss calculation to save memory
        """
        super().__init__()
        if reduction not in ("mean", "none", "sum"):
            raise NotImplementedError("Only support reduction = 'mean', 'none', or 'sum'")

        self.ignore_index     = int(ignore_index)
        self.reduction        = reduction
        self.label_smoothing  = float(label_smoothing)
        self.logit_scale      = float(logit_scale)
        self.lse_square_scale = float(lse_square_scale)
        self.return_z_loss    = bool(return_z_loss)

        self.reduction_map = {"none": 0, "mean": 1, "sum": 2}

    def forward(
        self,
        input:  torch.Tensor,         # Hidden  [.., D]
        weight: torch.Tensor = None,  # Weights [V,  D]
        target: torch.Tensor = None,  # Labels  [...  ]
    ):
        """
        Returns:
            loss   : scalar or tensor depending on reduction
            z_loss : same shape as loss if self.return_z_loss == True
        """

        if weight is None or target is None:
            raise TypeError(
                "CrossEntropyLoss requires 3 arguments: (hidden_states, lm_head_weight, labels) "
                "You called it with 2 arguments (likely logits and labels)."
                "\n\nYou must pass the hidden states and the classifier weights separately "
                "to avoid materializing the full logits matrix."
                "\n\nExample:\n"
                "  hidden = model.body(input_ids)\n"
                "  weight = model.lm_head.weight\n"
                "  loss = loss_fn(hidden, weight, labels)"
            )

        assert input.is_cuda and weight.is_cuda and target.is_cuda, "Only support CUDA tensors"

        loss, z_loss = cross_entropy_loss(
            input            = input,
            weight           = weight,
            labels           = target,
            reduction        = self.reduction,
            label_smoothing  = self.label_smoothing,
            logit_scale      = self.logit_scale,
            lse_square_scale = self.lse_square_scale,
            ignore_index     = self.ignore_index,
        )

        if not self.return_z_loss:
            return loss
        return loss, z_loss
