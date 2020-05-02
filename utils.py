from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler


def calculate_mean_std(x_train):
    """
    A function that calculates the mean and the standard deviation per color channel in an image.
    It is used for normalizing the image tensors later.

    :param x_train: torch.tensor
    :return: mean and std per channel of images as a tuples.
    """
    per_image_mean = torch.mean(x_train, dim=(0))
    per_image_std = torch.std(x_train, dim=(0))
    channel_mean = torch.mean(per_image_mean, dim=(0, 1)) / 255  # 0.2860
    channel_std = torch.std(per_image_std, dim=(0, 1)) / 255  # 0.1071

    return channel_mean, channel_std


def load_FashionMNIST(batch_size, ROOT):

    """
    A function that loads the Fashion-MNIST dataset.
    :param batch_size: The size of each batch of data.
    :param ROOT: The path to download to or load from the dataset.
    :return: data loaders for the training, validating and testing data.
    """
    # Defining the number of training samples.
    NUM_TRAIN = 58000

    # Normalizing: subtracting the mean and dividing by the standard deviation
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.2860,), (0.1071,), )])

    # Load the data. If data do not exist - download it.
    fashion_train = FashionMNIST(root=ROOT, train=True, download=True,
                                 transform=transform)
    loader_train = DataLoader(fashion_train, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    fashion_val = FashionMNIST(root=ROOT, train=True, download=True,
                               transform=transform)
    loader_val = DataLoader(fashion_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))
    fashion_test = FashionMNIST(root=ROOT, train=False, download=True,
                                transform=transform)
    loader_test = DataLoader(fashion_test, batch_size=batch_size)

    return loader_train, loader_val, loader_test


def flatten(x):
    """
    A function that reshapes/flattens n-th dimensional tensor.
    :param x: A tensor of shape (N,1,H,W)
    :return: flattened x of shape (N,HxW)
    """
    N = x.shape[0]  # read in N, H, W
    return x.view(N, -1)  # "flatten" the H * W values into a single vector per image


def test_fashion_convnet():
    """
    A function that tests if the network's output dimensions are right.
    """
    x = torch.zeros((64, 1, 28, 28), dtype=dtype)
    model = FashionConvNet(1, 32, 16, 10, relu)
    scores = model(x)
    print(scores.size())  # should print torch.Size([64, 10])
