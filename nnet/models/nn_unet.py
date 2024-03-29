import numpy as np
import torch

import nnet
import nnet.modules
import nnet.loss


class NNUNetModel(nnet.protocols.ModelProtocol):
    """A U-Net model."""

    encoder_map: list[list[int]]
    decoder_map: list[list[int]]
    final_image_size: tuple[int, int]
    hidden_size: int
    num_classes: int
    ignore_index: int
    learning_rate: float
    device: torch.device
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        encoder_map: list[list[int]],
        decoder_map: list[list[int]],
        final_image_size: tuple[int, int],
        hidden_size: int,
        num_classes: int,
        ignore_index: int,
        learning_rate: float = 3e-4,
        device: torch.device = torch.device("cuda"),
        dropout: float = 0.0,
        step: int = 0,
    ):
        self.encoder_map = encoder_map
        self.decoder_map = decoder_map
        self.final_image_size = final_image_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device
        self.step = step
        self.ignore_index = ignore_index
        self.dropout = dropout

        self.network = nnet.modules.NNUnet(
            encoder_map=self.encoder_map,
            decoder_map=self.decoder_map,
            final_image_size=self.final_image_size,
            hidden_size=self.hidden_size,
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

        acc_num = torch.sum(label_outputs == targets.squeeze()) - torch.sum(
            targets == self.ignore_index
        )

        acc_den = torch.sum(targets != self.ignore_index)

        accuracy = (acc_num.float() / acc_den.float()).item()

        flat_targets = targets.flatten().long()
        flat_outputs = outputs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        ce_loss = torch.nn.functional.cross_entropy(
            flat_outputs, flat_targets, ignore_index=self.ignore_index
        )
        dice_loss = nnet.loss.dice_loss(
            inputs=outputs,
            targets=targets,
            ignore_index=self.ignore_index,
        )
        loss = ce_loss + dice_loss

        return loss, accuracy

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
            "encoder_map": self.encoder_map,
            "decoder_map": self.decoder_map,
            "final_image_size": self.final_image_size,
            "hidden_size": self.hidden_size,
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
            encoder_map=checkpoint["encoder_map"],
            decoder_map=checkpoint["decoder_map"],
            final_image_size=checkpoint["final_image_size"],
            hidden_size=checkpoint["hidden_size"],
            num_classes=checkpoint["num_classes"],
            learning_rate=checkpoint["learning_rate"],
            device=torch.device(checkpoint["device"]),
            ignore_index=checkpoint["ignore_index"],
            step=checkpoint["step"],
        )
        model.network.load_state_dict(checkpoint["network"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        return model
