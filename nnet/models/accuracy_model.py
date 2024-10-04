import torch

import datasets.datatypes
import nnet
import nnet.modules
import nnet.protocols
import nnet.loss
import evaluate


class AccuracyModel(nnet.protocols.AccuracyModelProtocol):
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
            **network_kwargs,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer, step_size=163, gamma=0.96
        # )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=166 * 10 * 2
        )

    def train_one_step(
        self,
        logits: torch.Tensor,
        gt_segmentations: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        """Train the model on one step."""
        self.network.train()
        self.optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(logits, gt_segmentations, probs, locations)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        return loss.detach().cpu(), accuracy

    def validate(
        self,
        logits: torch.Tensor,
        gt_segmentations: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        """Validate the model."""
        self.network.eval()

        loss, accuracy = self._calc_loss_acc(logits, gt_segmentations, probs, locations)

        return loss.detach().cpu(), accuracy

    def _calc_loss_acc(
        self,
        logits: torch.Tensor,
        gt_segmentations: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        logits = logits.to(self.device).float()
        probs = probs.to(self.device).float() if probs is not None else None
        locations = locations.to(self.device).float() if locations is not None else None
        gt_segmentations = gt_segmentations.to(self.device)

        gts = []
        for lgts, gt in zip(logits, gt_segmentations):
            seg = torch.argmax(lgts, dim=0)
            f1 = evaluate.f1_score(
                seg[None, :, :], gt[None, :, :], self.num_classes, self.ignore_index
            )
            gts.append(torch.tensor(f1, device=self.device))

        gt_acc = torch.stack(gts)

        pred_acc = self.network(logits, probs, locations)

        acc_loss = torch.nn.functional.mse_loss(pred_acc.squeeze(), gt_acc.squeeze())
        accuracy = torch.mean(torch.abs(pred_acc - gt_acc)).item()

        loss = nnet.loss.CombinedLoss()
        loss.add(nnet.protocols.LossType.ACCURACY_ESTIMATE, acc_loss)

        return loss, accuracy

    def predict(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the model."""
        self.network.eval()

        logits = logits.to(self.device).float()
        probs = probs.to(self.device).float() if probs is not None else None
        locations = locations.to(self.device).float() if locations is not None else None

        return self.network(logits, probs, locations).detach().cpu().squeeze()

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
