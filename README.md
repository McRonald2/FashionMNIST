# FashionMNIST
A Pytorch implementaion of training and deploying a CNN model for the FashionMNIST dataset.<br>
Flask was used to deploy the trained model as a REST api.

For a quick implemetation of the web application please run the following in the Terminal: `python web_app.py`.<br>
Upload an image and get the predicted class.

For a quick client test you can Run `test_client.py` and get predicted classes of two image samples from the dataset.<br>
The images are stored in the "data" folder.

## Overview

`utils.py`: A module with helper functions for loading the data and preparing it for training.

`cnn_model.py`: A module used to define the costumed Convolutional Neural Network archirtecture and class.

`train.py`: A module used to automate the training process and the hyper-parameter tuning.<br>
<b>note:</b> The `train_save_best_model()` function is wrapping the training, hyper-parameter tuning and saving the best model as a `.pt` file.

## api and web applicaion

`web_app.py`: Create a web application based in REST api.

`app.py`: Create a REST api without the web stuff.

`commons.py`: A module with helper functions for loading the pre-trained model and <br>preparing raw image bytes to an inferable tensor for our model to predict on.

`inference.py`: A module for the inference task. It contains a single function `get_class_name`.

