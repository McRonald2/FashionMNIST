from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.nn.functional import relu, selu, leaky_relu
from torch import tanh
from utils import *
from cnn_model import FashionConvNet
from hyperopt import hp, tpe, fmin, STATUS_OK, space_eval
from torchtest import assert_vars_change


def train_fashion_mnist(model, batch_size, lr, momentum, num_epochs, optimization):
    """
    A function that trains a neural network with certain hyper-parameters.

    :param model: model's object (FashionConvNet in our case).
    :param batch_size: batch size of data to train on.
    :param lr: the learning rate for the optimizer algorithm.
    :param momentum: the momentum scalar for the optimizer algorithm (in case of SGD).
    :param num_epochs: number of epochs to run over all data points.
    :param optimization: optimization algorithm object.
    :return: Cross-Entropy loss.
    """

    model = model.to(device=torch.device("cpu"))  # run on cpu
    loader_train, loader_val, _ = load_FashionMNIST(batch_size=batch_size, ROOT='./data')  # load data.

    if optimization == SGD:
        optimizer = optimization(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optimization(model.parameters(), lr=lr)

    model.train()  # turn model to training mode.
    loss = None
    for e in range(num_epochs):
        for batch_idx, (X_tr, y_tr) in enumerate(loader_train):

            X_tr = X_tr.to(device=torch.device("cpu"), dtype=torch.float32)
            y_tr = y_tr.to(device=torch.device("cpu"), dtype=torch.long)

            # Assert variables change during the training process
            assert_vars_change(
                model=model,
                loss_fn=F.cross_entropy,
                optim=optimizer,
                batch=[X_tr, y_tr], device="cpu"
            )

            scores = model(X_tr)
            loss = F.cross_entropy(scores, y_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_every = 10
            if batch_idx % print_every == 0:
                print('Iteration %d, loss = %.4f' % (batch_idx, loss.item()))
                check_accuracy(loader_val, model)

    return loss


def train(hp_config, save_checkpoint=False):
    """
    A function to train over different hyper-parameters to find the best combination.

    :param hp_config: hyper-parameters search space.
    :param save_checkpoint: True for saving a checkpoint of the model for later inference task.
    :return: Cross-Entropy loss.
    """
    batch_size = int(hp_config['batch_size'])
    lr = hp_config['learning_rate']
    momentum = hp_config['momentum']
    channel_1 = hp_config['channel_1']
    channel_2 = hp_config['channel_2']
    num_epochs = hp_config['num_epochs']
    optimization = hp_config['optimization']
    activation = hp_config['activation']

    print('batch_size = %d, learning = %.4f, momentum = %.4f, channel_1 = %d , channel_2 = %d \n, num_epochs = %d, '
          'optimization = %s, activation = %s, \n' % (batch_size, lr, momentum, channel_1, channel_2, num_epochs,
                                                      str(optimization), str(activation)))

    model = FashionConvNet(1, channel_1, channel_2, 10, activation)
    loss = train_fashion_mnist(model, batch_size, lr, momentum, num_epochs, optimization)

    if save_checkpoint:
        state = {
            'model': FashionConvNet(1, channel_1, channel_2, 10, activation),
            'state_dict': model.state_dict(),
            'loss': loss,
            'epoch': num_epochs
        }
        torch.save(state, './data/checkpoint/CnnClassifier.pt')

    return {'loss': loss.item(), 'status': STATUS_OK}


def hyper_params_space():
    """
    A function to set a hyper-parameter search space.
    :return: A dictionary of hyper-parameters search spaces.
    """
    hp_config = {
        'batch_size': hp.choice('batch_size', range(32, 128, 32)),
        'learning_rate': hp.choice('lr', (0.01, 0.007, 0.005, 0.001)),
        'momentum': hp.uniform('momentum', 0.1, 0.5),
        'num_epochs': hp.choice('num_epochs', range(2, 6, 1)),
        'channel_1': hp.choice('channel_1', range(32, 128, 32)),
        'channel_2': hp.choice('channel_2', range(8, 32, 8)),
        'optimization': hp.choice('optimization', [SGD, Adam]),
        'activation': hp.choice('activation', [relu, selu, leaky_relu, tanh])
    }
    return hp_config


def find_best_params():
    """
    A function to find the best hyper-parameters combination that produce the lowest loss.
    :return: A dictionary of the best hyper-parameters.
    """
    # Load hyper-parameters search space
    hp_config = hyper_params_space()

    # Find and store the best hyper-parameters
    best_hp = fmin(
        fn=train,
        space=hp_config,
        algo=tpe.suggest,
        max_evals=20
    )

    return space_eval(hp_config, best_hp)


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=torch.device("cpu"), dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=torch.device("cpu"), dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)\n' % (num_correct, num_samples, 100 * acc))


def train_save_best_model():
    print('Tuning hyper parameters...')
    best_hp = find_best_params()
    print('Best hyper-params: ', best_hp)
    print('Training and Saving best model...')
    train_dict = train(best_hp, save_checkpoint=True)
    print('Checkpoint saved.\nBest loss: %d' % (train_dict['loss']))


# best_hp = find_best_params()
# train_save_best_model()

best_hp = {'batch_size': 64,
           'learning_rate': 0.005,
           'momentum': 0.3964,
           'channel_1': 64,
           'channel_2': 24,
           'optimization': Adam,
           'activation': selu,
           'num_epochs': 2}

train_dict = train(best_hp, save_checkpoint=True)

