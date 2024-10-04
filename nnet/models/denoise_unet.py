import torch

import datasets.datatypes
import nnet
import nnet.modules
import nnet.protocols
import nnet.loss
import evaluate


class DenoiseUNetModel(nnet.protocols.DenoiseModelProtocol):
    """A U-Net model."""

    encoder_map: list[list[int]]
    decoder_map: list[list[int]]
    hidden_size: int
    num_classes: int
    ignore_index: int
    learning_rate: float
    device: torch.device
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        learning_rate: float,
        device: torch.device = torch.device("cuda"),
        dropout: float = 0.0,
        step: int = 0,
        **network_kwargs,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device
        self.step = step
        self.ignore_index = ignore_index
        self.dropout = dropout
        self.network_kwargs = network_kwargs

        self.network = nnet.modules.SemantSegUNet(
            input_channels=self.num_classes,
            use_linear_bridge=False,
            **network_kwargs,
            num_classes=self.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer, step_size=163, gamma=0.96
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=362 * 100,
            eta_min=learning_rate * 0.3,
        )

    def train_one_step(
        self,
        inputs: datasets.datatypes.SegInputs,
        gt_segmentation: torch.Tensor,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        """Train the model on one step."""
        self.network.train()
        self.optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(inputs, gt_segmentation)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        return loss.detach().cpu(), accuracy

    def validate(
        self,
        inputs: datasets.datatypes.SegInputs,
        gt_segmentation: torch.Tensor,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        """Validate the model."""
        self.network.eval()

        loss, accuracy = self._calc_loss_acc(inputs, gt_segmentation)

        return loss.detach().cpu(), accuracy

    def _calc_loss_acc(
        self,
        inputs: datasets.datatypes.SegInputs,
        gt_segmentation: torch.Tensor,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        inputs.to_device(self.device)
        gt_segmentation = gt_segmentation.to(self.device)

        seg_outputs = self.network(inputs)

        label_outputs = torch.argmax(seg_outputs.logits, dim=1)

        label_outputs[gt_segmentation.squeeze(1) == self.ignore_index] = (
            self.ignore_index
        )

        accuracy = evaluate.mean_dice(
            label_outputs.detach().cpu(),
            gt_segmentation.detach().cpu(),
            n_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        denoise_loss = nnet.loss.tversky_loss(
            seg_outputs.logits,
            gt_segmentation,
            ignore_index=self.ignore_index,
            alpha=0.4,
            beta=0.6,
        )

        loss = nnet.loss.CombinedLoss()
        loss.add(nnet.protocols.LossType.DENOISE, denoise_loss)

        return loss, accuracy

    def predict(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the model."""
        self.network.eval()

        logits = logits.to(self.device).float()
        probs = probs.to(self.device).float() if probs is not None else None
        locations = locations.to(self.device).float() if locations is not None else None

        seg_outputs = self.network(
            datasets.datatypes.SegInputs(logits, probs, locations)
        )

        labels = torch.argmax(seg_outputs.logits, dim=1).detach().cpu().unsqueeze(1)

        return seg_outputs.logits.detach().cpu(), labels

    def save(self, path: str):
        """Save the model to a file."""
        checkpoint = {
            "network_kwargs": self.network_kwargs,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "ignore_index": self.ignore_index,
            "device": self.device.type,
            "step": self.step,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def restore(cls, path: str):
        """Restore the model from a file."""
        checkpoint = torch.load(path)
        model = cls(
            num_classes=checkpoint["num_classes"],
            learning_rate=checkpoint["learning_rate"],
            device=torch.device(checkpoint["device"]),
            ignore_index=checkpoint["ignore_index"],
            step=checkpoint["step"],
            **checkpoint["network_kwargs"],
        )
        model.network.load_state_dict(checkpoint["network"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        model.scheduler.load_state_dict(checkpoint["scheduler"])
        return model

    def load(self, path: str):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        return self
