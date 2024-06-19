from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Carregando os modelos
model_cnn = load_model('./modelos/pesos.h5')
model_linear = load_model('./modelos/pesos_linear.h5')

def prepare_image(image, target_size=(28, 28)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model_type' not in request.form:
        return jsonify({'error': 'No file or model type found'})
    
    file = request.files['file']
    model_type = request.form['model_type']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = prepare_image(image)
    
    if model_type == 'cnn':
        prediction = model_cnn.predict(image)
        model_used = 'CNN'
    elif model_type == 'linear':
        prediction = model_linear.predict(image)
        model_used = 'Linear'
    else:
        return jsonify({'error': 'Invalid model type'})
    
    response = {
        'model_used': model_used,
        'prediction': int(np.argmax(prediction))
    }
    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
