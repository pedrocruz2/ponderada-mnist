from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

model_cnn = load_model('pesos.h5')
model_linear = load_model('pesos_linear.h5')

def prepare_image(image, target_size=(28, 28)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file found"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = prepare_image(image)
    
    pred_cnn = model_cnn.predict(image)
    pred_linear = model_linear.predict(image)
    
    response = {
        'cnn_prediction': int(np.argmax(pred_cnn)),
        'linear_prediction': int(np.argmax(pred_linear))
    }
    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
