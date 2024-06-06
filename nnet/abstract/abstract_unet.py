import abc
import numpy as np
import torch

import nnet.protocols
import nnet.modules
import nnet.loss
import evaluate


class AbstractUNetModel(nnet.protocols.ModelProtocol, abc.ABC):
    """A U-Net model."""

    model_class: type[torch.nn.Module]

    @abc.abstractmethod
    def get_loss(self, outputs, targets, ignore_index):
        pass

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        step: int = 0,
        device: torch.device = torch.device("cuda"),
        high_lr: float = 1e-4,
        low_lr: float = 1e-5,
        **network_kwargs,
    ):
        self.device = device
        self.step = step
        self.ignore_index = ignore_index
        self.high_lr = high_lr
        self.low_lr = low_lr
        self.num_classes = num_classes
        self.network_kwargs = network_kwargs

        self.network = self.model_class(num_classes=num_classes, **network_kwargs).to(
            self.device
        )

        if len(self.network.film_params()) > 0:
            self.high_lr_optimizer = torch.optim.AdamW(
                self.network.film_params(), lr=self.high_lr
            )
            # self.high_lr_scheduler = (
            #    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #        self.high_lr_optimizer, T_0=100, T_mult=2, eta_min=self.high_lr / 10
            #    )
            # )
            self.high_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.high_lr_optimizer, gamma=0.999
            )
        else:
            self.high_lr_optimizer = None

        self.low_lr_optimizer = torch.optim.AdamW(
            self.network.conv_params(), lr=self.low_lr
        )
        # self.low_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    self.low_lr_optimizer, T_0=100, T_mult=2, eta_min=self.low_lr / 10
        # )
        self.low_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.low_lr_optimizer, gamma=0.999
        )

    def train_one_step(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:
        """Train the model on one step."""
        self.network.train()
        if self.high_lr_optimizer is not None:
            self.high_lr_optimizer.zero_grad()
        self.low_lr_optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(inputs, targets)

        loss.backward()

        if self.high_lr_optimizer is not None:
            self.high_lr_optimizer.step()
            self.high_lr_scheduler.step()
        self.low_lr_optimizer.step()
        self.low_lr_scheduler.step()

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

        images, conditions, positions = inputs

        outputs = self.network(images.float(), conditions.float(), positions.float())

        if isinstance(targets, tuple):
            seg_targets = targets[0]
        else:
            seg_targets = targets

        if isinstance(outputs, tuple):
            seg_outputs = outputs[0]
        else:
            seg_outputs = outputs

        label_outputs = torch.argmax(seg_outputs, dim=1)

        label_outputs[seg_targets.squeeze(1) == self.ignore_index] = self.ignore_index

        f1_score = evaluate.f1_score(
            label_outputs.detach().cpu(),
            seg_targets.detach().cpu(),
            n_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        loss = self.get_loss(outputs, targets, self.ignore_index)

        return loss, f1_score

    def predict(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Use the model to predict outputs."""
        self.network.eval()

        inputs = nnet.util.send_to_device(inputs, self.device)

        if isinstance(inputs, tuple):
            images, conditions, positions = inputs
            outputs = self.network(
                images.float(), conditions.float(), positions.float()
            )
        else:
            outputs = self.network(inputs.float())

        if isinstance(outputs, tuple):
            seg_outputs = outputs[0]
        else:
            seg_outputs = outputs

        label_outputs = torch.argmax(seg_outputs, dim=1).detach().cpu().numpy()

        if isinstance(outputs, tuple):
            return (label_outputs, outputs[1].detach().cpu().numpy())

        return label_outputs

    def save(self, path: str):
        """Save the model to a file."""
        checkpoint = {
            "network_kwargs": self.network_kwargs,
            "num_classes": self.num_classes,
            "high_lr": self.high_lr,
            "low_lr": self.low_lr,
            "ignore_index": self.ignore_index,
            "device": self.device.type,
            "step": self.step,
            "network": self.network.state_dict(),
            "high_lr_optimizer": (
                self.high_lr_optimizer.state_dict()
                if self.high_lr_optimizer is not None
                else 0
            ),
            "low_lr_optimizer": self.low_lr_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def restore(cls, path: str):
        """Restore the model from a file."""
        checkpoint = torch.load(path)
        model = cls(
            num_classes=checkpoint["num_classes"],
            high_lr=checkpoint["high_lr"],
            low_lr=checkpoint["low_lr"],
            device=torch.device(checkpoint["device"]),
            ignore_index=checkpoint["ignore_index"],
            step=checkpoint["step"],
            **checkpoint["network_kwargs"],
        )
        model.network.load_state_dict(checkpoint["network"])
        if model.high_lr_optimizer is not None:
            model.high_lr_optimizer.load_state_dict(checkpoint["high_lr_optimizer"])
        model.low_lr_optimizer.load_state_dict(checkpoint["low_lr_optimizer"])

        return model

    def load(self, path: str):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        if self.high_lr_optimizer is not None:
            self.high_lr_optimizer.load_state_dict(checkpoint["high_lr_optimizer"])
        self.low_lr_optimizer.load_state_dict(checkpoint["low_lr_optimizer"])
        self.step = checkpoint["step"]

        return self
