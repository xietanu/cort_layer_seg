import nnet.protocols
import nnet.models


def create_model_from_config(config: dict) -> nnet.protocols.SegModelProtocol:
    """Create a model from a config."""
    # if config["base_model"] != "unet3+":
    #    raise ValueError(f"Unsupported model: {config['base_model']}")

    if config["conditional"]:
        config["nnet_config"]["uses_condition"] = True
    if config["positional"]:
        config["nnet_config"]["uses_position"] = True

    if config["base_model"] == "unet3+":
        model = nnet.models.UNet3PlusModel(
            input_channels=1,
            num_classes=8,
            ignore_index=8,
            high_lr=config["high_lr"],
            low_lr=config["low_lr"],
            **config["nnet_config"],
        )
    elif config["base_model"] == "unet":
        model = nnet.models.SemantSegUNetModel(
            num_classes=8,
            ignore_index=8,
            denoise_model_name=(
                config["denoise_model_name"] if "denoise_model_name" in config else None
            ),
            learning_rate=config["lr"],
            **config["nnet_config"],
        )
    else:
        raise ValueError(f"Unsupported model: {config['base_model']}")

    return model
