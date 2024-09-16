# data_preprocessing.py

import os
import cv2 # Biblioteca OpenCV, usada para carregar e processar imagens.
import numpy as np # Biblioteca para manipulação eficiente de arrays numéricos.
from sklearn.model_selection import train_test_split

# Função para garantir que as pastas existem
def verificar_pastas(directory):
    if not os.path.exists(directory):
        print(f"Diretório {directory} não encontrado. Criando pasta...")
        os.makedirs(directory)

# Função para carregar e processar imagens
def carregar_imagens_das_pastas(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

# Função para normalizar as imagens
def normalizar_imagens(images):
    return np.array(images) / 255.0 
    # Normalizar os valores dos pixels das imagens para o intervalo [0, 1].
    # Converte a lista de imagens em um array NumPy (np.array(images)) 
    # e divide os valores dos pixels por 255.0, normalizando-os para o intervalo [0, 1].
    # Os valores dos pixels variam de 0 a 255, e redes neurais treinam melhor quando os dados 
    # estão em um intervalo menor e mais estável (como [0, 1]).

# Função para dividir os dados em treino e validação
def dividir_dados(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42) 
    # random_state=42 garante que a divisão seja replicável. 42 é a resposta para a vida, o universo e tudo mais (Mochileiro).
    # Se o código for executado novamente, a divisão será a mesma).
    # 20% dos dados (especificado por test_size=0.2).
    # 80% dos dados (por padrão).
