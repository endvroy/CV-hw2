import os
import operator
import torch
import torch.nn.functional as F
import torch.optim as optim
import data_loader
from model import Net
from tqdm import tqdm

params = {
    'data_path': 'data/nyucvfall2019',
    'batch_size': 256,
    'lr': 1e-3,
    'momentum': 1e-2,
    'log_interval': 10,
    'epoches': 20
}


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
            loss = max(self.epoches[epoch].items(), key=operator.itemgetter(0))[1]
            losses.append(loss)
        return losses

    def state_dict(self):
        return {'epoches': self.epoches,
                'vals': self.vals}

    def load_state_dict(self, state_dict):
        self.epoches = state_dict['epoches']
        self.vals = state_dict['vals']


def train_epoch(model, optimizer, train_loader, epoch, train_logger):
    model.train()
    for batch_idx, (inp, target) in enumerate(train_loader):
        inp, target = inp.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(inp)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % params['log_interval'] == 0:
            train_logger.log(epoch, batch_idx, loss.item())
            print('Epoch {}, Batch {}: [{}/{} ({:.0f}%)]    Loss: {:.6f}'.format(
                epoch,
                batch_idx,
                batch_idx * len(inp),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))


def validate(model, val_loader, epoch, train_logger):
    print(f'Validating epoch {epoch}:')
    model.eval()
    val_loss = 0
    correct = 0
    for inp, target in tqdm(val_loader):
        inp, target = inp.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(inp)
        loss = F.nll_loss(output, target).item()  # sum up batch loss
        val_loss += loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct.item() / len(val_loader.dataset)
    train_logger.log_val(epoch, val_loss, accuracy, pred.tolist())
    print('Validation loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss,
        correct,
        len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


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


def load_checkpoint(model, optimizer, checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    train_logger.load_state_dict(state['train_logger'])
    return state['epoch'], train_logger


def train(model, optimizer, epoch, epoches_to_train, save_path, train_logger):
    for epoch_i in range(1, epoches_to_train + 1):
        train_epoch(model, optimizer, train_loader, epoch + epoch_i, train_logger)
        validate(model, val_loader, epoch + epoch_i, train_logger)
        save_checkpoint(model, optimizer, save_path, epoch + epoch_i, train_logger)
        print('')


if __name__ == '__main__':
    train_loader = data_loader.load_train(params['data_path'], params['batch_size'])
    val_loader = data_loader.load_val(params['data_path'], params['batch_size'])
    print('train and val data loaded')

    model = Net()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    epoch = 0
    epoches_to_train = params['epoches']
    train_logger = TrainLogger()
    epoch, train_logger = load_checkpoint(model, optimizer, 'checkpoints/1/epoch_3.pth')
    epoches_to_train = 10
    train(model, optimizer, epoch, epoches_to_train, 'checkpoints/1', train_logger)
