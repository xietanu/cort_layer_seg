import numpy as np
import torch

import nnet
import nnet.modules
import nnet.loss
import evaluate


class ConditionalUNet3PlusModel(nnet.protocols.ModelProtocol):
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
        embedding_dim: int = 48,
        hidden_embed_dim: int = 48,
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
        self.embedding_dim = embedding_dim
        self.hidden_embed_dim = hidden_embed_dim

        self.network = nnet.modules.ConditionalUNet3Plus(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            depth=self.depth,
            num_classes=self.num_classes,
            dropout=self.dropout,
            embedding_dim=self.embedding_dim,
            hidden_embed_dim=self.hidden_embed_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, mode="min", factor=0.3, patience=20
        # )

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

        # self.scheduler.step(loss)

        return loss.item(), accuracy

    def _calc_loss_acc(self, inputs, targets):
        images, conditions = nnet.util.send_to_device(inputs, self.device)
        targets = nnet.util.send_to_device(targets, self.device)

        if isinstance(targets, tuple):
            targets = targets[0]

        outputs = self.network(images.float(), conditions.float())

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

        images, conditions = nnet.util.send_to_device(inputs, self.device)

        outputs = self.network(images.float(), conditions.float())

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
            # "scheduler": self.scheduler.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_embed_dim": self.hidden_embed_dim,
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
            embedding_dim=checkpoint["embedding_dim"],
            hidden_embed_dim=checkpoint["hidden_embed_dim"],
        )
        model.network.load_state_dict(checkpoint["network"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        # model.scheduler.load_state_dict(checkpoint["scheduler"])
        return model
