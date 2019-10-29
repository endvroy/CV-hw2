import matplotlib.pyplot as plt
from utils import load_state, load_train_logger

if __name__ == '__main__':
    state = load_state('checkpoints/stn7/epoch_20.pth')
    train_logger = load_train_logger(state)
    losses = train_logger.epoch_losses()
    epoches = range(1, train_logger.n_epoches() + 1)
    plt.plot(epoches, losses)
    plt.xticks(epoches)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.show()
