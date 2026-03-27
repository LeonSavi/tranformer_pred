from lightning.pytorch.callbacks import Callback

class TrainingHistoryLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss_epoch')
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('val_loss')
        if loss is not None:
            self.val_losses.append(loss.item())

    @property
    def history(self):
        # align to shorter list (validation may have one extra from sanity check)
        n = min(len(self.train_losses), len(self.val_losses))
        return {
            'epoch': list(range(1, n + 1)),
            'train_loss': self.train_losses[:n],
            'val_loss': self.val_losses[:n],
        }