import numpy as np
import torch

import nnet
import nnet.modules
import nnet.loss
import evaluate


class UNet3PlusModel(nnet.protocols.ModelProtocol):
    """A U-Net model."""

    input_channels: int
    base_channels: int
    depth: int
    num_classes: int
    dropout: float
    ignore_index: int
    learning_rate: float
    device: torch.device
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        ignore_index: int,
        learning_rate: float = 3e-4,
        device: torch.device = torch.device("cuda"),
        dropout: float = 0.0,
        step: int = 0,
    ):
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.depth = depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device
        self.step = step
        self.ignore_index = ignore_index
        self.dropout = dropout

        self.network = nnet.modules.UNet3Plus(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            depth=self.depth,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )

    def train_one_step(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:
        """Train the model on one step."""
        self.network.train()
        self.optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(inputs, targets)

        loss.backward()

        self.optimizer.step()

        self.step += 1

        return loss.item(), accuracy

    def validate(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:
        """Validate the model."""
        self.network.eval()

        loss, accuracy = self._calc_loss_acc(inputs, targets)

        return loss.item(), accuracy

    def _calc_loss_acc(self, inputs, targets):
        inputs = nnet.util.send_to_device(inputs, self.device)
        targets = nnet.util.send_to_device(targets, self.device)

        if isinstance(targets, tuple):
            targets = targets[0]

        outputs = self.network(inputs.float())

        label_outputs = torch.argmax(outputs, dim=1)

        label_outputs[targets.squeeze() == self.ignore_index] = self.ignore_index

        f1_score = evaluate.f1_score(
            torch.nn.functional.softmax(outputs, dim=1).detach().cpu(),
            targets.detach().cpu(),
            self.ignore_index,
            softmax_based=True,
        )

        # ce_loss = torch.nn.functional.cross_entropy(
        #    flat_outputs, flat_targets, ignore_index=self.ignore_index
        # )
        # dice_loss = nnet.loss.dice_loss(
        #    inputs=outputs,
        #    targets=targets,
        #    ignore_index=self.ignore_index,
        # )
        # loss = ce_loss + dice_loss
        loss = nnet.loss.tversky_loss(
            inputs=outputs,
            targets=targets,
            ignore_index=self.ignore_index,
        )

        return loss, f1_score

    def predict(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> np.ndarray:
        """Predict the model."""
        self.network.eval()

        inputs = nnet.util.send_to_device(inputs, self.device)

        outputs = self.network(inputs.float())

        label_outputs = torch.argmax(outputs, dim=1)

        return label_outputs.detach().cpu().numpy()

    def save(self, path: str):
        """Save the model to a file."""
        checkpoint = {
            "input_channels": self.input_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "ignore_index": self.ignore_index,
            "device": self.device.type,
            "step": self.step,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def restore(cls, path: str):
        """Restore the model from a file."""
        checkpoint = torch.load(path)
        model = cls(
            input_channels=checkpoint["input_channels"],
            base_channels=checkpoint["base_channels"],
            depth=checkpoint["depth"],
            num_classes=checkpoint["num_classes"],
            learning_rate=checkpoint["learning_rate"],
            device=torch.device(checkpoint["device"]),
            ignore_index=checkpoint["ignore_index"],
            step=checkpoint["step"],
        )
        model.network.load_state_dict(checkpoint["network"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        return model
