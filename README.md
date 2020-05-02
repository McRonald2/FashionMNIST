# FashionMNIST
A Pytorch implementaion of training and deploying a CNN model for the FashionMNIST dataset.<br>
Flask was used to deploy the trained model as a REST api.

For a quick implemetation of the web application please run the following in the Terminal: `python web_app.py`.<br>
Upload an image and get the predicted class.

For a quick client test there you can Run `test_client.py` and get predicted classes of two image samples from the dataset.<br>
The images are stored in the "data" folder.

## Overview

`utils.py`: A module with helper functions for loading the data and preparing it for training.<br>
`cnn_model.py`: A module used to define the cutumed Convolutional Neural Network archirtecture.<br>
`train.py`: A module used to automate the traing process and the hyper-parameter tuning.<br>
`utils.py`: A module with helper functions for loading the data and preparing it for training.<br>
