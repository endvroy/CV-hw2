import torch
import torch.nn.functional as F
import torch.optim as optim
import data_loader
from model import Net
import os
from tqdm import tqdm

params = {
    'data_path': 'data/nyucvfall2019',
    'batch_size': 256,
    'lr': 1e-3,
    'momentum': 1e-2,
    'log_interval': 2,
    'epoches': 30
}


def train_epoch(model, optimizer, train_loader, epoch):
    model.train()
    for batch_idx, (inp, target) in enumerate(train_loader):
        inp, target = inp.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(inp)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % params['log_interval'] == 0:
            print('Epoch {}, Batch {}: [{}/{} ({:.0f}%)]    Loss: {:.6f}'.format(
                epoch,
                batch_idx,
                batch_idx * len(inp),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))


def validate(model, val_loader, epoch):
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
    print('Validation loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss,
        correct,
        len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def save_checkpoint(model, optimizer, checkpoint_path, epoch, continued=False):
    os.makedirs(checkpoint_path, exist_ok=continued)
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    fname = f'epoch_{epoch}.pth'
    torch.save(state, os.path.join(checkpoint_path, fname))


def load_check_point(model, optimizer, checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch']


def train(model, optimizer, epoch, epoches_to_train, save_path, continued=False):
    for epoch_i in range(1, epoches_to_train + 1):
        train_epoch(model, optimizer, train_loader, epoch + epoch_i)
        validate(model, val_loader, epoch + epoch_i)
        save_checkpoint(model, optimizer, save_path, epoch + epoch_i, continued)


if __name__ == '__main__':
    train_loader = data_loader.load_train(params['data_path'], params['batch_size'])
    val_loader = data_loader.load_val(params['data_path'], params['batch_size'])
    model = Net()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    # epoch = 0
    # epoches_to_train = 10
    epoch = load_check_point(model, optimizer, 'checkpoints_t/epoch_10.pth')
    epoches_to_train = params['epoches']
    train(model, optimizer, epoch, epoches_to_train, 'checkpoints_t', continued=True)
