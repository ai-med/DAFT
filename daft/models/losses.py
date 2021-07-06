# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
from typing import Optional

import torch
from torch import nn

__all__ = ["CoxphLoss"]


def safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min, _ = torch.min(x, dim=0)
    c = torch.zeros(x_min.shape, device=x.device)
    norm = torch.where(x_min < 0, -x_min, c)
    return x + norm


def logsumexp_masked(
    risk_scores: torch.Tensor, mask: torch.Tensor, dim: int = 0, keepdim: Optional[bool] = None
) -> torch.Tensor:
    """Compute logsumexp across `dim` for entries where `mask` is true."""
    assert risk_scores.dim() == mask.dim(), "risk_scores and mask must have same rank"

    mask_f = mask.type_as(risk_scores)
    risk_scores_masked = risk_scores * mask_f
    # for numerical stability, substract the maximum value
    # before taking the exponential
    amax, _ = torch.max(risk_scores_masked, dim=dim, keepdim=True)
    risk_scores_shift = risk_scores_masked - amax

    exp_masked = risk_scores_shift.exp() * mask_f
    exp_sum = exp_masked.sum(dim, keepdim=True)
    output = exp_sum.log() + amax
    if not keepdim:
        output.squeeze_(dim=dim)
    return output


class CoxphLoss(nn.Module):
    def forward(self, predictions: torch.Tensor, event: torch.Tensor, riskset: torch.Tensor) -> torch.Tensor:
        """Negative partial log-likelihood of Cox's proportional
        hazards model.

        Args:
            predictions (torch.Tensor):
                The predicted outputs. Must be a rank 2 tensor.
            event (torch.Tensor):
                Binary vector where 1 indicates an event 0 censoring.
            riskset (torch.Tensor):
                Boolean matrix where the `i`-th row denotes the
                risk set of the `i`-th instance, i.e. the indices `j`
                for which the observer time `y_j >= y_i`.

        Returns:
            loss (torch.Tensor):
                Scalar loss.

        References:
            .. [1] Faraggi, D., & Simon, R. (1995).
            A neural network model for survival data. Statistics in Medicine,
            14(1), 73–82. https://doi.org/10.1002/sim.4780140108
        """
        if predictions is None or predictions.dim() != 2:
            raise ValueError("predictions must be a 2D tensor.")
        if predictions.size()[1] != 1:
            raise ValueError("last dimension of predictions ({}) must be 1.".format(predictions.size()[1]))
        if event is None:
            raise ValueError("event must not be None.")
        if predictions.dim() != event.dim():
            raise ValueError(
                "Rank of predictions ({}) must equal rank of event ({})".format(predictions.dim(), event.dim())
            )
        if event.size()[1] != 1:
            raise ValueError("last dimension event ({}) must be 1.".format(event.size()[1]))
        if riskset is None:
            raise ValueError("riskset must not be None.")

        event = event.type_as(predictions)
        riskset = riskset.type_as(predictions)
        predictions = safe_normalize(predictions)

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        pred_t = predictions.t()

        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, dim=1, keepdim=True)
        assert rr.size() == predictions.size()

        losses = event * (rr - predictions)
        loss = torch.mean(losses)

        return loss
