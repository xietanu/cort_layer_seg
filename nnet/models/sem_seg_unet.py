import matplotlib.pyplot as plt
import numpy as np
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
        accuracy_model_name: str | None = None,
        decay_epochs: int = 50,
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
        self.accuracy_model_name = accuracy_model_name

        self.network = nnet.modules.SemantSegUNet(
            input_channels=1,
            **network_kwargs,
            num_classes=self.num_classes,
        ).to(self.device)

        self.denoise_model = None
        self.accuracy_model = None

        if self.denoise_model_name is not None:
            try:
                self.denoise_model = nnet.models.DenoiseUNetModel.restore(
                    f"models/{self.denoise_model_name}.pth"
                )
                self.denoise_model.device = self.device
                self.denoise_model.network.to(self.device)
            except FileNotFoundError:
                self.denoise_model = None
                print("!!! Denoise model not found !!!")

        if self.accuracy_model_name is not None:
            try:
                self.accuracy_model = nnet.models.AccuracyModel.restore(
                    f"models/{self.accuracy_model_name}.pth"
                )
                self.accuracy_model.device = self.device
                self.accuracy_model.network.to(self.device)
            except FileNotFoundError:
                self.accuracy_model = None
                print("!!! Accuracy model not found !!!")

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer, step_size=163, gamma=0.96
        # )
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(
        #    self.optimizer,
        #    start_factor=1.0,
        #    end_factor=0.1,
        #    total_iters=166 * decay_epochs,
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=171 * decay_epochs,
            eta_min=learning_rate * 0.3,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    self.optimizer,
        #    T_0=5,
        #    T_mult=1,
        #    eta_min=learning_rate * 0.1,
        # )

    def train_one_step(
        self,
        inputs: datasets.datatypes.SegInputs,
        ground_truths: datasets.datatypes.SegGroundTruths | None,
        step: float | None = None,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        """Train the model on one step."""
        self.network.train()
        self.optimizer.zero_grad()

        loss, accuracy = self._calc_loss_acc(inputs, ground_truths)

        loss.backward()

        self.optimizer.step()
        if step is not None:
            # self.scheduler.step(step)
            self.scheduler.step()
            self.step += 1

        return loss.detach().cpu(), accuracy

    def validate(
        self,
        inputs: datasets.datatypes.SegInputs,
        ground_truths: datasets.datatypes.SegGroundTruths | None,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        """Validate the model."""
        self.network.eval()

        loss, accuracy = self._calc_loss_acc(inputs, ground_truths)

        return loss.detach().cpu(), accuracy

    def _calc_loss_acc(
        self,
        inputs: datasets.datatypes.SegInputs,
        ground_truths: datasets.datatypes.SegGroundTruths | None,
    ) -> tuple[nnet.loss.CombinedLoss, float]:
        inputs.to_device(self.device)

        raw_outputs = self.network(inputs, autoencode_only=ground_truths is None)

        if ground_truths is not None:
            label_outputs = torch.argmax(raw_outputs.logits, dim=1)

            ground_truths.to_device(self.device)

            label_outputs[
                ground_truths.segmentation.squeeze(1) == self.ignore_index
            ] = self.ignore_index

            accuracy = evaluate.mean_dice(
                label_outputs.detach().cpu(),
                ground_truths.segmentation.detach().cpu(),
                n_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
        else:
            accuracy = 0

        loss = nnet.loss.CombinedLoss()

        if ground_truths is not None and ground_truths.segmentation is not None:
            segmentation_loss = nnet.loss.tversky_loss(
                raw_outputs.logits, ground_truths.segmentation, self.ignore_index
            )
            loss.add(nnet.protocols.LossType.SEGMENTATION, segmentation_loss)

        if (
            raw_outputs.depth_maps is not None
            and ground_truths is not None
            and ground_truths.depth_maps is not None
            and ground_truths.segmentation is not None
        ):
            depth_loss = nnet.loss.depthmap_loss(
                raw_outputs.depth_maps,
                ground_truths.depth_maps,
                ground_truths.segmentation,
                self.ignore_index,
            )
            loss.add(
                nnet.protocols.LossType.DEPTH,
                depth_loss,
                3,  # / raw_outputs.depth_maps.shape[1],
            )

            consistency_loss = nnet.loss.consistency_loss(
                raw_outputs.logits,
                raw_outputs.depth_maps,
                ground_truths.segmentation,
                self.ignore_index,
            )
            loss.add(nnet.protocols.LossType.CONSISTENCY, consistency_loss, 1 / 10)

        if raw_outputs.autoencoded_imgs is not None:
            if ground_truths is not None and ground_truths.segmentation is not None:
                pred = raw_outputs.autoencoded_imgs[
                    ground_truths.segmentation != self.ignore_index
                ]
                gt = inputs.input_images[
                    ground_truths.segmentation != self.ignore_index
                ]
            else:
                pred = raw_outputs.autoencoded_imgs
                gt = inputs.input_images

            loss_fn = torch.nn.MSELoss()
            autoencoder_loss = loss_fn(
                pred.float(),
                gt.float(),
            )
            loss.add(nnet.protocols.LossType.AUTOENCODE, autoencoder_loss, 10)

        return loss, accuracy

    def predict(
        self,
        inputs: datasets.datatypes.SegInputs,
        logits: bool = False,
    ) -> datasets.datatypes.Predictions:
        """Predict the model."""
        self.network.eval()

        inputs.to_device(self.device)

        raw_ouputs = self.network(
            inputs,
        )

        logits = raw_ouputs.logits.detach().cpu()

        labels = torch.argmax(logits, dim=1).detach().cpu().unsqueeze(1)

        denoised_logits = None
        denoised_labels = None
        if self.denoise_model is not None:
            denoised_logits, denoised_labels = self.denoise_model.predict(
                logits, inputs.area_probabilities.float(), inputs.position.float()
            )
            denoised_logits = denoised_logits.detach().cpu()
            denoised_labels = denoised_labels.detach().cpu()
        acc_preds = torch.zeros((inputs.input_images.shape[0])).to(self.device)

        if self.accuracy_model is not None:
            acc_preds = self.accuracy_model.predict(
                logits, inputs.area_probabilities.float(), inputs.position.float()
            )
            if len(acc_preds.shape) == 0:
                acc_preds = torch.stack([acc_preds]).to(self.device)

        return datasets.datatypes.Predictions(
            segmentation=labels,
            accuracy=acc_preds.detach().cpu(),
            depth_maps=(
                raw_ouputs.depth_maps.detach().cpu()
                if raw_ouputs.depth_maps is not None
                else None
            ),
            logits=logits,
            denoised_segementation=denoised_labels,
            denoised_logits=denoised_logits,
            autoencoded_img=(
                raw_ouputs.autoencoded_imgs.detach().cpu()
                if raw_ouputs.autoencoded_imgs is not None
                else None
            ),
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
            "accuracy_model_name": self.accuracy_model_name,
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
            accuracy_model_name=checkpoint["accuracy_model_name"],
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
