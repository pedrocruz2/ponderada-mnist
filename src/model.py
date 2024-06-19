# Importações necessárias
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np

# Carregando o dataset e separando os dados de treino e de teste
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Transformando os labels em one-hot encoding
y_treino_cat = to_categorical(y_treino)
y_teste_cat = to_categorical(y_teste)

# Normalização dos dados de entrada
x_treino_norm = x_treino / x_treino.max()
x_teste_norm = x_teste / x_teste.max()

# Reshape dos dados de entrada para adicionar o canal de cor
x_treino_norm = x_treino_norm.reshape(len(x_treino_norm), 28, 28, 1)
x_teste_norm = x_teste_norm.reshape(len(x_teste_norm), 28, 28, 1)

# Criação do modelo LeNet5
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compila o modelo
adam = Adam()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Realiza o treinamento do modelo
historico = model.fit(x_treino_norm, y_treino_cat, epochs=5, validation_split=0.2)

# Salva o modelo
model.save('pesos.h5')

# Criação do modelo linear
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Criação do modelo linear
linear_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
adam = Adam()
# Compilação do modelo
linear_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Treinamento do modelo linear
linear_historico = linear_model.fit(x_treino_norm, y_treino_cat, epochs=5, validation_split=0.2)

# Salva o modelo linear
linear_model.save('pesos_linear.h5')

