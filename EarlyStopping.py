#!/usr/bin/python

import torch

class EarlyStopping_loss(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, wandb=None, name=""):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0 #  Counter which checks for early stopping
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e11 # Nawid - Set a very high initial best loss
        self.name = name
        self.wandb = wandb
        self.epoch_counter = 0 # Counter which counts the epochs
        self.checkpoint_counter = 0 # Counter which saves after 10 successful decreases in value

    def __call__(self, val_loss, model,optimizer):

        score = val_loss
        self.epoch_counter +=1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
        elif score >= self.best_score: # Nawid - Inverse signs to take into minimising loss instead of maximising accuracy
            self.counter += 1
            print(f'EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'{self.name} has stopped')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased/improved for {self.name}  ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        current_checkpoint = {
            'epoch': self.epoch_counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'lr':optimizer.state_dict()['param_groups'][0]['lr']} # Saves the epoch number, model state dict, optimizer state, loss and learning rate


        save_dir = self.wandb.run.dir
        torch.save(current_checkpoint, save_dir + "/" + self.name + ".pt")
        self.wandb.save(save_dir + "/" + self.name + ".pt")
        #torch.save(model.state_dict(), save_dir + "/" + 'Epoch_no_{}_'.format(self.epoch_counter)+self.name + ".pt")
        #self.wandb.save(save_dir + "/" + self.name + ".pt")
        self.val_loss_min = val_loss

        self.checkpoint_counter += 1

        if self.checkpoint_counter % 10 == 0: # Saves the optimiser and the state model
            checkpoint = {
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'loss': val_loss,
              'lr': optimizer.state_dict()['param_groups'][0]['lr']}

            torch.save(checkpoint, save_dir + "/" + 'Epoch_no_{}_'.format(self.epoch_counter)+self.name + "_checkpoint.pt")
            self.wandb.save(save_dir + "/" + 'Epoch_no_{}_'.format(self.epoch_counter)+self.name + "_checkpoint.pt")
