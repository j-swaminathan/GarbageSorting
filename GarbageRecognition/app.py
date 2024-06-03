import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load your pre-trained models
resnet_model = load_model("model/garbage_resnet_model.h5")
#cnn_model = load_model("model/garbage_cnn_model.h5")
vgg_model = load_model("model/garbage_vgg_model.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Function to make predictions and return the classes
def predict_and_plot(img_path):
    img_array = load_and_preprocess_image(img_path)

    # Make predictions
 #   cnn_pred = cnn_model.predict(img_array)
    resnet_pred = resnet_model.predict(img_array)
    vgg_pred = vgg_model.predict(img_array)

    # Decode predictions
  #  cnn_class = class_names[np.argmax(cnn_pred)]
    resnet_class = class_names[np.argmax(resnet_pred)]
    vgg_class = class_names[np.argmax(vgg_pred)]
    predicted_accuracy_vgg = round(np.max(vgg_pred) * 100, 2)
   # predicted_accuracy_cnn = round(np.max(cnn_pred) * 100, 2)
    predicted_accuracy_resnet = round(np.max(resnet_pred) * 100, 2)

    #print('CNN accuracy is : ', predicted_accuracy_cnn)
    print('Resnet accuracy is : ', predicted_accuracy_resnet)
    print('vgg accuracy is : ', predicted_accuracy_vgg)
    return  resnet_class, vgg_class,  predicted_accuracy_resnet, predicted_accuracy_vgg


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_name = None
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file_name = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(img_path)

            # Debugging: Check if the image file is saved correctly
            if os.path.exists(img_path):
                print(f"Image successfully saved at {img_path}")
            else:
                print(f"Failed to save image at {img_path}")

            resnet_class, vgg_class, predicted_accuracy_resnet, predicted_accuracy_vgg = predict_and_plot(img_path)
            prediction = { 'resnet': resnet_class, 'vgg': vgg_class,  'acc_resnet': predicted_accuracy_resnet, 'acc_vgg': predicted_accuracy_vgg}

        elif 'capture' in request.form:
            print('IM gere')
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "Could not open video device"

            ret, frame = cap.read()
            if ret:
                file_name = 'captured_image.jpg'
                capture_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                cv2.imwrite(capture_path, frame)
                cap.release()
                cv2.destroyAllWindows()

                resnet_class, vgg_class,  predicted_accuracy_resnet, predicted_accuracy_vgg = predict_and_plot(capture_path)
                prediction = { 'resnet': resnet_class, 'vgg': vgg_class,  'acc_resnet': predicted_accuracy_resnet, 'acc_vgg': predicted_accuracy_vgg}
            else:
                cap.release()
                cv2.destroyAllWindows()
                return "Failed to capture image"

    return render_template('index.html', prediction=prediction, file_name=file_name)


if __name__ == '__main__':
    app.run(debug=True)
