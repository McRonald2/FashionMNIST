from commons import get_tensor, get_model
import json

with open('cat_to_name.json') as f:
    cat_to_name = json.load(f)

model = get_model()


def get_class_name(image_bytes):
    """
    Get image bytes from an image, turn them to a tensor and predict the right class for this tensor.
    :param image_bytes:
    :return: class's id (int) and class's name (string).
    """
    tensor = get_tensor(image_bytes)
    outputs = model.forward(tensor)
    _, prediction = outputs.max(1)
    pred_class = prediction.item()
    class_name = cat_to_name[str(pred_class)]
    return pred_class, class_name
