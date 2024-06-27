import nnet.protocols
import nnet.training
import datasets


def train_on_fold(
    model: nnet.protocols.SegModelProtocol,
    fold: datasets.Fold,
    n_epochs: int = 100,
    save_name: str | None = None,
):
    """Train a model on a fold_data."""
    return nnet.training.train_seg_model(
        model=model,
        train_dataloader=fold.train_dataloader,
        val_dataloader=fold.val_dataloader,
        n_epochs=n_epochs,
        save_name=save_name,
    )
