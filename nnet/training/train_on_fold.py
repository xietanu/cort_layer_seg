import nnet.protocols
import nnet.training
import datasets


def train_on_fold(
    model: nnet.protocols.SegModelProtocol,
    fold: datasets.Fold,
    n_epochs: int = -1,
    end_after: int = -1,
    save_name: str | None = None,
    use_random: bool = False,
    use_gaussian: bool = False,
    use_synth: bool = False,
):
    """Train a model on a fold_data."""
    return nnet.training.train_seg_model(
        model=model,
        train_dataloader=fold.train_dataloader,
        val_dataloader=fold.val_dataloader,
        random_dataloader=fold.random_dataloader if use_random else None,
        gaussian_dataloader=fold.gaussian_dataloader if use_gaussian else None,
        syn_dataloader=fold.synth_dataloader if use_synth else None,
        n_epochs=n_epochs,
        end_after=end_after,
        save_name=save_name,
    )
