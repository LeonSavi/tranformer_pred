from lightning.pytorch.callbacks import EarlyStopping, Callback

class TrainingHistoryLogger(Callback):
    def __init__(self):
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    def on_train_epoch_end(self, trainer, pl_module):
        self.history['epoch'].append(trainer.current_epoch + 1)
        self.history['train_loss'].append(
            trainer.callback_metrics.get('train_loss_epoch', float('nan')).item()
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get('val_loss', float('nan'))
        if len(self.history['val_loss']) < len(self.history['epoch']):
            self.history['val_loss'].append(val.item())