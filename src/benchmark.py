import time
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.layers import Flatten
from keras.optimizers import Adam

# Carregar dataset
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Pré-processamento dos dados
x_treino = x_treino / 255.0
x_teste = x_teste / 255.0
x_treino = np.expand_dims(x_treino, axis=-1)
x_teste = np.expand_dims(x_teste, axis=-1)
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)

# Modelo CNN
def create_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo Linear
def create_linear_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Treinamento e avaliação dos modelos
def benchmark_model(model, x_treino, y_treino, x_teste, y_teste):
    start_time = time.time()
    history = model.fit(x_treino, y_treino, epochs=5, validation_split=0.2)
    training_time = time.time() - start_time
    
    start_time = time.time()
    loss, accuracy = model.evaluate(x_teste, y_teste)
    inference_time = time.time() - start_time
    
    return training_time, accuracy, inference_time

# Benchmark CNN
cnn_model = create_cnn_model()
cnn_training_time, cnn_accuracy, cnn_inference_time = benchmark_model(cnn_model, x_treino, y_treino, x_teste, y_teste)

# Benchmark Linear
linear_model = create_linear_model()
linear_training_time, linear_accuracy, linear_inference_time = benchmark_model(linear_model, x_treino, y_treino, x_teste, y_teste)

# Exibir resultados
print(f"CNN - Tempo de treinamento: {cnn_training_time:.2f} segundos, Acurácia: {cnn_accuracy:.4f}, Tempo de inferência: {cnn_inference_time:.4f} segundos")
print(f"Linear - Tempo de treinamento: {linear_training_time:.2f} segundos, Acurácia: {linear_accuracy:.4f}, Tempo de inferência: {linear_inference_time:.4f} segundos")
