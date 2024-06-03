# Garbage Sorting

This project uses the Kaggle Garbage classification dataset to train deep learning models and compare their performance.
We train 3 models:
- Convolution Neural Network
- ResNet50
- VGG19

and save the model weights. Since CNN model weights are more than 100MB, we load only the pretrained models to the repository.

We store the model weights in GarbageRecognition/model folder
Using the 2 saved models, we build a FLASK app where we load an image or capture an image with the webcam and classify it into 6 different Garbage categories.

The categories are
- Cardboard
- Paper
- Plastic
- Glass
- Metal
- Trash.


### Contents 
- `GarbageSorting.ipynb` -- EDA, Training, Saving the model
- `Garbage_predict.ipynb` -- Load the saved models and predict the classes
- `GarbageRecognition` Flaskapp


### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/j-swaminathan/GarbageSorting.git
    cd GarbageSorting
    ```

### Running the Flask App

1. **Navigate to the Flask app directory:**

    ```bash
    cd GarbageRecognition
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask app:**

    ```bash
    python app.py
    ```

5. **Open your browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

### Usage

- **Upload an Image:** Click on the "Choose File" button, select an image, and click "Upload" to classify it.
- **Capture an Image:** Click on the "Start Camera" button to use your webcam, then click on "Capture Image" to take picture and click on "Upload" to classify it.


### License

This project is licensed under the MIT License.

### Acknowledgments

- Kaggle for the Garbage classification dataset.













-
