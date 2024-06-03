# Garbage Sorting

This project uses the Kaggle Garbage classification dataset to train deep learning models and compare their performance.
We train 3 models:
- Convolution Neural Network
- ResNet50
- VGG19

and save the model weights. Since CNN model weights are more than 100MB, we load only the pretrained models to the repository.

Using the 2 saved models, we build a FLASK app where we load an image or capture an image with the webcam and classify it into 6 different Garbage categories.

The categories are
- Cardboard
- Paper
- Plastic
- Glass
- Metal
- Trash.


### Contents 
- GarbageSorting.ipynb -- EDA, Training, Saving the model
- Garbage_predict.ipynb -- Load the saved models and predict the classes
- GarbageRecognition Flaskapp


### Installation

- Clone the repository

- Install the required dependencies















-
