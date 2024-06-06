from __future__ import annotations

import nnet.abstract
import nnet.modules
import nnet.loss


def get_3plus_type(
    conditional: bool, depth: bool
) -> type[nnet.abstract.AbstractUNetModel]:
    """Get the 3+ U-Net model type."""
    if conditional:
        if depth:
            return UNet3PlusConitionalWithDepth
        return UNet3PlusConditional
    if depth:
        return UNet3PlusWithDepth
    return UNet3Plus


class UNet3Plus(nnet.abstract.AbstractUNetModel):
    """A 3+ U-Net model."""

    model_class = nnet.modules.UNet3Plus

    def get_loss(self, outputs, targets, ignore_index):
        targets = targets[0]
        return nnet.loss.tversky_loss(outputs, targets, ignore_index)


class UNet3PlusConditional(nnet.abstract.AbstractUNetModel):
    """A 3+ U-Net model with conditional connections."""

    model_class = nnet.modules.ConditionalUNet3Plus

    def get_loss(self, outputs, targets, ignore_index):
        targets = targets[0]
        return nnet.loss.tversky_loss(outputs, targets, ignore_index)


class UNet3PlusConitionalWithDepth(nnet.abstract.AbstractUNetModel):
    """A 3+ U-Net model with conditional connections and depth."""

    model_class = nnet.modules.ConditionalDepthUNet3Plus

    def get_loss(self, outputs, targets, ignore_index):
        # outputs = outputs[0]
        # targets = targets[0]
        # return nnet.loss.tversky_loss(outputs, targets, ignore_index)

        return nnet.loss.seg_depth_comb_loss(outputs, targets, ignore_index)


class UNet3PlusWithDepth(nnet.abstract.AbstractUNetModel):
    """A 3+ U-Net model with conditional connections and depth."""

    model_class = nnet.modules.DepthUNet3Plus

    def get_loss(self, outputs, targets, ignore_index):
        # outputs = outputs[0]
        # targets = targets[0]
        # return nnet.loss.tversky_loss(outputs, targets, ignore_index)

        return nnet.loss.seg_depth_comb_loss(outputs, targets, ignore_index)
