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

with torch.no_grad():
    DEPTH_MULTI = torch.tensor(np.linspace(0, 1, 256)[None, None, :, None]).to(
        torch.float64
    )


class ConditionalShiftUNet3PlusModel(nnet.protocols.ModelProtocol):
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
        input_shape: tuple[int, int],
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
        self.input_shape = input_shape

        self.network = nnet.modules.ConditionalShiftUNet3Plus(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            depth=self.depth,
            num_classes=self.num_classes,
            dropout=self.dropout,
            embed_dim=self.embed_dim,
            hidden_embed_dim=self.hidden_embed_dim,
            input_shape=self.input_shape,
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

        loss, accuracy = self._calc_loss_acc(inputs, targets, report=True)

        # self.scheduler.step(loss)

        return loss.item(), accuracy

    def _calc_loss_acc(self, inputs, targets, report=False):
        images, conditions = nnet.util.send_to_device(inputs, self.device)
        targets = nnet.util.send_to_device(targets, self.device)

        if isinstance(targets, tuple):
            targets = targets[0]

        outputs, offsets = self.network(images.float(), conditions.float())

        label_outputs = torch.argmax(outputs, dim=1)

        label_outputs[targets.squeeze() == self.ignore_index] = self.ignore_index

        f1_score = evaluate.f1_score(
            torch.argmax(outputs, dim=1).detach().cpu(),
            targets.detach().cpu(),
            n_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        seg_loss = nnet.loss.tversky_loss(
            inputs=outputs,
            targets=targets,
            ignore_index=self.ignore_index,
        )

        with torch.no_grad():
            pred_mask = torch.argmax(outputs, dim=1)[:, None, :, :]
            pred_mask_offset = torch.roll(pred_mask, shifts=-1, dims=2)

            true_mask = targets
            true_mask_offset = torch.roll(true_mask, shifts=-1, dims=2)

            pred_borders = torch.zeros(
                (
                    pred_mask.shape[0],
                    self.num_classes - 1,
                    pred_mask.shape[2],
                    pred_mask.shape[3],
                ),
                device=self.device,
            )
            true_borders = torch.zeros(
                (
                    true_mask.shape[0],
                    self.num_classes - 1,
                    true_mask.shape[2],
                    true_mask.shape[3],
                ),
                device=self.device,
            )
            for i in range(self.num_classes - 1):
                pred_borders[:, i, :, :] = (
                    ((pred_mask == i) & (pred_mask_offset == i + 1))
                    .to(torch.float32)
                    .squeeze()
                )
                true_borders[:, i, :, :] = (
                    ((true_mask == i) & (true_mask_offset == i + 1))
                    .to(torch.float32)
                    .squeeze()
                )

            pred_depth_weighted = pred_borders * DEPTH_MULTI.to(self.device)
            true_depth_weighted = true_borders * DEPTH_MULTI.to(self.device)

            pred_avg_depth = torch.sum(pred_depth_weighted, dim=(2, 3)) / (
                torch.sum(pred_borders, dim=(2, 3)) + 1e-8
            )

            true_avg_depth = torch.sum(true_depth_weighted, dim=(2, 3)) / (
                torch.sum(true_borders, dim=(2, 3)) + 1e-8
            )

            diff = pred_avg_depth - true_avg_depth

        offsets = offsets.to(torch.float64)

        diff_loss = torch.nn.functional.mse_loss(diff, offsets, reduction="sum")

        loss = seg_loss + diff_loss

        if report:
            print(offsets - diff)

        return loss, f1_score

    def predict(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> np.ndarray:
        """Predict the model."""
        self.network.eval()

        images, conditions = nnet.util.send_to_device(inputs, self.device)

        outputs, offsets = self.network(images.float(), conditions.float())

        label_outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        offset_labels = np.roll(label_outputs, shift=-1, axis=1)

        offsets = np.floor(
            offsets.detach().cpu().numpy() * label_outputs.shape[1]
        ).astype(int)

        for i in range(offsets.shape[1]):
            cur_borders = np.zeros_like(label_outputs)
            cur_borders[(label_outputs == i) & (offset_labels == i + 1)] = 1
            offset_borders = np.roll(cur_borders, 1, axis=1)
            for j in range(offsets.shape[0]):
                roll_dir = np.sign(offsets[j, i])
                inst_border = cur_borders[j]
                inst_offset = offset_borders[j]
                for k in range(abs(offsets[j, i])):
                    inst_border = np.roll(inst_border, roll_dir, axis=0)
                    inst_offset = np.roll(inst_offset, roll_dir, axis=0)
                    label_outputs[j, inst_border == 1] = i
                    label_outputs[j, inst_offset == 1] = i + 1

        return label_outputs

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
            "input_shape": self.input_shape,
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
            input_shape=checkpoint["input_shape"],
        )
        model.network.load_state_dict(checkpoint["network"])
        model.conv_optimizer.load_state_dict(checkpoint["conv_optimizer"])
        model.film_optimizer.load_state_dict(checkpoint["film_optimizer"])

        # model.scheduler.load_state_dict(checkpoint["scheduler"])
        return model
