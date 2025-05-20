"""
Early stopping is a regularization technique used to 
prevent overfitting by halting the training process 
when the model's performance on a validation set stops 
improving. 
"""

class EarlyStopping:
    """This class will stop training 
    early if the validation loss 
    doesn't improve by at least `min_delta` 
    for `patience` number of epochs."""

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True