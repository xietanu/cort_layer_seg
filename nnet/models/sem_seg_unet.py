import torch

import datasets.datatypes
import nnet
import nnet.modules
import nnet.protocols
import nnet.loss
import evaluate


class SemantSegUNetModel(nnet.protocols.SegModelProtocol):
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
    denoise_model_name: str

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        learning_rate: float,
        device: torch.device = torch.device("cuda"),
        dropout: float = 0.0,
        step: int = 0,
        denoise_model_name: str | None = None,
        **network_kwargs,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device
        self.step = step
        self.ignore_index = ignore_index
        self.dropout = dropout
        self.network_kwargs = network_kwargs
        self.denoise_model_name = denoise_model_name

        self.network = nnet.modules.SemantSegUNet(
            input_channels=1,
            **network_kwargs,
            num_classes=self.num_classes,
        ).to(self.device)

        self.denoise_model = None

        if self.denoise_model_name is not None:
            self.denoise_model = nnet.models.DenoiseUNetModel.restore(
                f"models/{self.denoise_model_name}.pth"
            )
            self.denoise_model.device = self.device
            self.denoise_model.network.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer, step_size=163, gamma=0.96
        # )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=163 * 100
        )

    def train_one_step(
        self,
        inputs: datasets.datatypes.SegInputs,
        ground_truths: datasets.datatypes.SegGroundTruths,
    ) -> tuple[float, float, float]:
        """Train the model on one step."""
        self.network.train()
        self.optimizer.zero_grad()

        loss, accuracy, acc_err = self._calc_loss_acc(inputs, ground_truths)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        return loss.item(), accuracy, acc_err

    def validate(
        self,
        inputs: datasets.datatypes.SegInputs,
        ground_truths: datasets.datatypes.SegGroundTruths,
    ) -> tuple[float, float, float]:
        """Validate the model."""
        self.network.eval()

        loss, accuracy, acc_err = self._calc_loss_acc(inputs, ground_truths)

        return loss.item(), accuracy, acc_err

    def _calc_loss_acc(
        self,
        inputs: datasets.datatypes.SegInputs,
        ground_truths: datasets.datatypes.SegGroundTruths,
    ) -> tuple[torch.Tensor, float, float]:
        inputs.to_device(self.device)
        ground_truths.to_device(self.device)

        raw_outputs = self.network(
            inputs.input_images.float(),
            (
                inputs.area_probabilities.float()
                if inputs.area_probabilities is not None
                else None
            ),
            inputs.position.float() if inputs.position is not None else None,
        )

        seg_outputs, acc_outputs = raw_outputs[:2]

        label_outputs = torch.argmax(seg_outputs, dim=1)

        label_outputs[ground_truths.segmentation.squeeze(1) == self.ignore_index] = (
            self.ignore_index
        )

        accuracy = evaluate.mean_dice(
            label_outputs.detach().cpu(),
            ground_truths.segmentation.detach().cpu(),
            n_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        with torch.no_grad():
            per_sample_dice = torch.tensor(
                [
                    evaluate.f1_score(
                        label_outputs[None, i],
                        ground_truths.segmentation[None, i],
                        self.num_classes,
                        self.ignore_index,
                    )
                    for i in range(len(ground_truths.segmentation))
                ],
                device=self.device,
            )

        if len(raw_outputs) == 3:
            loss = nnet.loss.seg_acc_depth_loss(
                raw_outputs, ground_truths, per_sample_dice, self.ignore_index
            )
        else:
            loss = nnet.loss.seg_acc_loss(
                raw_outputs, ground_truths, per_sample_dice, self.ignore_index
            )

        with torch.no_grad():
            acc_err = torch.abs(raw_outputs[1] - per_sample_dice).mean()

        return loss, accuracy, acc_err

    def predict(
        self,
        inputs: datasets.datatypes.SegInputs,
        logits: bool = False,
    ) -> datasets.datatypes.Predictions:
        """Predict the model."""
        self.network.eval()

        inputs.to_device(self.device)

        raw_ouputs = self.network(
            inputs.input_images.float(),
            (
                inputs.area_probabilities.float()
                if inputs.area_probabilities is not None
                else None
            ),
            inputs.position.float() if inputs.position is not None else None,
        )

        seg_outputs, acc_outputs = raw_ouputs[:2]

        logits = seg_outputs.detach().cpu()

        labels = torch.argmax(seg_outputs, dim=1).detach().cpu().unsqueeze(1)

        denoised_logits = None
        denoised_labels = None
        if self.denoise_model is not None:
            if (
                "uses_condition" in self.denoise_model.network_kwargs
                and self.denoise_model.network_kwargs["uses_condition"]
            ):
                denoised_logits, denoised_labels = self.denoise_model.predict(
                    logits, inputs.area_probabilities.float(), inputs.position.float()
                )
            else:
                denoised_logits, denoised_labels = self.denoise_model.predict(logits)

        return datasets.datatypes.Predictions(
            segmentation=labels,
            accuracy=acc_outputs.detach().cpu(),
            depth_maps=(raw_ouputs[2].detach().cpu() if len(raw_ouputs) == 3 else None),
            logits=logits,
            denoised_segementation=denoised_labels.detach().cpu(),
            denoised_logits=denoised_logits.detach().cpu(),
        )

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
            "denoise_model_name": self.denoise_model_name,
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
            denoise_model_name=checkpoint["denoise_model_name"],
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
