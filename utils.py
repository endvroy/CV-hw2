import os
import torch
import statistics


class TrainLogger:
    def __init__(self):
        self.epoches = {}
        self.vals = {}

    def log(self, epoch, batch, loss):
        if epoch not in self.epoches:
            self.epoches[epoch] = {}
        self.epoches[epoch][batch] = loss

    def log_val(self, epoch, loss, accuracy, preds):
        self.vals[epoch] = [loss, accuracy, preds]

    def n_epoches(self):
        return len(self.epoches)

    def epoch_losses(self):
        losses = []
        for epoch in range(1, self.n_epoches() + 1):
            loss = statistics.mean(self.epoches[epoch].values())
            losses.append(loss)
        return losses

    def state_dict(self):
        return {'epoches': self.epoches,
                'vals': self.vals}

    def load_state_dict(self, state_dict):
        self.epoches = state_dict['epoches']
        self.vals = state_dict['vals']


def save_checkpoint(model, optimizer, checkpoint_path, epoch, train_logger):
    os.makedirs(checkpoint_path, exist_ok=True)
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'train_logger': train_logger.state_dict()}
    fname = f'epoch_{epoch}.pth'
    fpath = os.path.join(checkpoint_path, fname)
    torch.save(state, fpath)
    print(f'checkpoint saved to {fpath}')


def load_state(checkpoint_path):
    state = torch.load(checkpoint_path)
    return state


def load_model(model, state):
    model.load_state_dict(state['model'])


def load_optimizer(optimizer, state):
    optimizer.load_state_dict(state['optimizer'])


def load_train_logger(state):
    train_logger = TrainLogger()
    train_logger.load_state_dict(state['train_logger'])
    return train_logger


def load_epoch(state):
    return state['epoch']


def load_checkpoint(model, optimizer, checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    train_logger = TrainLogger()
    train_logger.load_state_dict(state['train_logger'])
    return state['epoch'], train_logger
