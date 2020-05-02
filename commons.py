import io
from utils import *
from torchvision import transforms
from PIL import Image

"""
Helper functions for our deployed service.
"""


def get_model():
    """
    Load out pre-trained model.
    :return: model with trained weights.
    """
    checkpoint = torch.load('./data/checkpoint/CnnClassifier.pt')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def get_tensor(image_bytes):
    """
    Transform image's bytes to a tensor.
    :param image_bytes: image bytes.
    :return: 4-dimensional tensor.
    """
    transform = transforms.Compose([transforms.CenterCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.2860,), (0.1071,), )])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)
