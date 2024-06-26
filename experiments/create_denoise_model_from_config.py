import nnet.protocols
import nnet.models


def create_denoise_model_from_config(
    config: dict,
) -> nnet.protocols.DenoiseModelProtocol:
    """Create a model from a config."""
    if config["conditional"]:
        config["nnet_config"]["uses_condition"] = True
    if config["positional"]:
        config["nnet_config"]["uses_position"] = True

    model = nnet.models.DenoiseUNetModel(
        num_classes=8,
        ignore_index=8,
        learning_rate=config["lr"],
        **config["nnet_config"],
    )

    return model
