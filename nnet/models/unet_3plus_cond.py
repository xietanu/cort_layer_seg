import numpy as np
import torch

import nnet
import nnet.modules
import nnet.loss
import evaluate
import datasets.protocols


TRAIN_CONV = 0
TRAIN_FILM = 1
TRAIN_ALL = 2


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
        conv_lr: float = 3e-4,
        film_lr: float = 3e-4,
        embed_dim: int = 48,
        hidden_embed_dim: int = 48,
        device: torch.device = torch.device("cuda"),
        dropout: float = 0.0,
        step: int = 0,
    ):
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.depth = depth
        self.num_classes = num_classes
        self.conv_lr = conv_lr
        self.film_lr = film_lr
        self.device = device
        self.step = step
        self.ignore_index = ignore_index
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_embed_dim = hidden_embed_dim
        self.num_classes_with_ignore = num_classes + 1

        self.network = nnet.modules.ConditionalUNet3Plus(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            depth=self.depth,
            num_classes=self.num_classes,
            dropout=self.dropout,
            embed_dim=self.embed_dim,
            hidden_embed_dim=self.hidden_embed_dim,
        ).to(self.device)

        self.conv_optimizer = torch.optim.Adam(
            self.network.conv_params(), lr=self.conv_lr
        )
        self.film_optimizer = torch.optim.Adam(
            self.network.film_params(), lr=self.film_lr
        )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, mode="min", factor=0.3, patience=20
        # )
        self._train_mode = TRAIN_ALL

    def train_conv(self):
        for param in self.network.conv_params():
            param.requires_grad = True
        for param in self.network.film_params():
            param.requires_grad = False
        self._train_mode = TRAIN_CONV

    def train_film(self):
        for param in self.network.conv_params():
            param.requires_grad = False
        for param in self.network.film_params():
            param.requires_grad = True
        self._train_mode = TRAIN_FILM

    def train_all(self):
        for param in self.network.conv_params():
            param.requires_grad = True
        for param in self.network.film_params():
            param.requires_grad = True
        self._train_mode = TRAIN_ALL

    def train_one_step(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:
        """Train the model on one step."""
        self.network.train()
        if self._train_mode == TRAIN_CONV or self._train_mode == TRAIN_ALL:
            self.conv_optimizer.zero_grad()
        if self._train_mode == TRAIN_FILM or self._train_mode == TRAIN_ALL:
            self.film_optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(inputs, targets)

        loss.backward()

        if self._train_mode == TRAIN_CONV or self._train_mode == TRAIN_ALL:
            self.conv_optimizer.step()
        if self._train_mode == TRAIN_FILM or self._train_mode == TRAIN_ALL:
            self.film_optimizer.step()

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
            torch.argmax(outputs, dim=1).detach().cpu(),
            targets.detach().cpu(),
            n_classes=self.num_classes,
            ignore_index=self.ignore_index,
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
            "conv_lr": self.conv_lr,
            "film_lr": self.film_lr,
            "ignore_index": self.ignore_index,
            "device": self.device.type,
            "step": self.step,
            "network": self.network.state_dict(),
            "conv_optimizer": self.conv_optimizer.state_dict(),
            "film_optimizer": self.film_optimizer.state_dict(),
            "hidden_embed_dim": self.hidden_embed_dim,
            "embedding_dim": self.embed_dim,
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
            conv_lr=checkpoint["conv_lr"],
            film_lr=checkpoint["film_lr"],
            device=torch.device(checkpoint["device"]),
            ignore_index=checkpoint["ignore_index"],
            step=checkpoint["step"],
            embed_dim=checkpoint["embedding_dim"],
            hidden_embed_dim=checkpoint["hidden_embed_dim"],
        )
        model.network.load_state_dict(checkpoint["network"])
        model.conv_optimizer.load_state_dict(checkpoint["conv_optimizer"])
        model.film_optimizer.load_state_dict(checkpoint["film_optimizer"])

        # model.scheduler.load_state_dict(checkpoint["scheduler"])
        return model
