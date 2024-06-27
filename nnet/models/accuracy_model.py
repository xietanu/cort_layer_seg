import torch

import datasets.datatypes
import nnet
import nnet.modules
import nnet.protocols
import nnet.loss
import evaluate


class AccModel(nnet.protocols.AccuracyModelProtocol):
    """A U-Net model."""

    encoder_map: list[list[int]]
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

        self.network = nnet.modules.ImgRegressor(
            input_channels=self.num_classes,
            uses_accuracy=False,
            **network_kwargs,
            num_classes=self.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer, step_size=163, gamma=0.96
        # )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=163 * 50
        )

    def train_one_step(
        self,
        logits: torch.Tensor,
        gt_segmentations: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Train the model on one step."""
        self.network.train()
        self.optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(logits,gt_segmentations, probs, locations)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        return loss.item(), accuracy

    def validate(
        self,
        logits: torch.Tensor,
        gt_segmentations: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Validate the model."""
        self.network.eval()

        loss, accuracy = self._calc_loss_acc(logits, gt_segmentations, probs, locations)

        return loss.item(), accuracy

    def _calc_loss_acc(
        self,
        logits: torch.Tensor,
        gt_segmentations: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        logits = logits.to(self.device).float()
        probs = probs.to(self.device).float() if probs is not None else None
        locations = locations.to(self.device).float() if locations is not None else None

        gts = []
        for lgts, gt in zip(logits, gt_segmentations):
            seg = torch.argmax(lgts, dim=0)
            f1 = evaluate.f1_score(seg, gt, self.num_classes, self.ignore_index)
            gts.append(torch.tensor(f1))

        gts = torch.stack(gts)


        seg_outputs = self.network(logits, probs, locations)

        label_outputs = torch.argmax(seg_outputs, dim=1)

        label_outputs[gt_segmentations.squeeze(1) == self.ignore_index] = (
            self.ignore_index
        )

        accuracy = torch.

        loss = nnet.loss.tversky_loss(
            seg_outputs,
            gt_segmentation,
            ignore_index=self.ignore_index,
        )

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

        seg_outputs = self.network(logits, probs, locations)

        labels = torch.argmax(seg_outputs, dim=1).detach().cpu().unsqueeze(1)

        return seg_outputs.detach().cpu(), labels

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
