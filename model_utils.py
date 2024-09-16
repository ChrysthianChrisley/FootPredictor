# model_utils.py

import os
import glob
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np  
import cv2
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow TF fornece todas as ferramentas e recursos para construir, treinar e ajustar essas modelos. 
# Pode lidar com problemas grandes e complexos e fazer cálculos muito rapidamente.
# Quando o TF constrói um modelo, você pode dar a ela muitos exemplos, e ela aprenderá a fazer tarefas, 
# como identificar objetos em fotos ou prever o clima.

# Keras é uma "interface amigável de controle" do TensorFlow. 
# O Keras simplifica e dá um painel de controle fácil de usar. 

# Caminhos das Pastas
base_dir = os.path.dirname(__file__) # Raiz
pasta_modelos = os.path.join(base_dir, "Modelos_treinados") # Nome da Pasta atual

# Função para carregar o modelo melhor e mais recente
def pegar_melhor_modelo(path): # Pega o melhor modelo salvo com base na menor perda de validação (val_loss)
    models = glob.glob(os.path.join(path, "*.keras"))
    if not models:
        return None, None

    melhor_modelo = None
    melhor_perda = float('inf') 
    # Forma de representar o conceito de infinito em Python. float('inf') > Qualquer número.
    # A melhor perda de validação começa como infinito.
    
    caminho_melhor_modelo = None

    for caminho_modelo in models:
        model = load_model(caminho_modelo)
        
        # Verifica se o arquivo de histórico existe
        local_historico_do_modelo = caminho_modelo.replace('.keras', '_history.npy')
        if os.path.exists(local_historico_do_modelo):
            history = np.load(local_historico_do_modelo, allow_pickle=True).item()
            # Obtém a perda de validação mais baixa
            if 'val_loss' in history:
                last_val_loss = history['val_loss'][-1] 
                if last_val_loss < melhor_perda:
                    melhor_perda = last_val_loss
                    melhor_modelo = model
                    caminho_melhor_modelo = caminho_modelo

    return melhor_modelo, caminho_melhor_modelo

# Função para criar e compilar um novo modelo
def create_model(img_size):     # CNN (Rede Neural Convolucional) CNNs são especialmente eficazes no processamento de dados visuais, como imagens.
    model = Sequential([        # sequential (sequência de camadas) - Parâmetros:
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(), # achatar os dados 
        Dense(128, activation='relu'),
        Dropout(0.5), # 0.5 = 50% para evitar overfitting (quando o modelo memoriza os dados de treino e não generaliza bem para novos dados.)
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Função para treinar o modelo
def train_model(X_train, y_train, X_val, y_val, img_size, model=None):
    if model is None:
        model = create_model(img_size)

    model_name = datetime.now().strftime("Model_%d-%m-%y_%H-%M.keras")
    caminho_modelo = os.path.join(pasta_modelos, model_name)

    checkpoint = ModelCheckpoint(caminho_modelo, monitor='val_loss', save_best_only=True, mode='min')
    # ModelCheckpoint salva o melhor modelo com base na perda de validação mínima.

    # Criar datasets de treinamento e validação
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=base_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=32,
        class_names=['Healthy_Foot', 'Non_Healthy_Foot'],
        shuffle=True,
        subset="training",
        validation_split=0.2,
        seed=123
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=base_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=32,
        class_names=['Healthy_Foot', 'Non_Healthy_Foot'],
        shuffle=True,
        subset="validation",
        validation_split=0.2,
        seed=123
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255) # Aplica a normalização das imagens (escalando valores entre 0 e 1) usando a camada Rescaling(1./255).
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Treinar o modelo
    history = model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=20,
              callbacks=[checkpoint])

    # Salvar o histórico de treinamento
    local_historico_do_modelo = caminho_modelo.replace('.keras', '_history.npy')
    np.save(local_historico_do_modelo, history.history)

    return model

# Função para prever as imagens e exibir resultados
def Fazer_previsao_e_exibir_resultados(model, test_folder, reference_folders):
    test_images = []
    test_filenames = []
    predictions = []

    # Carregar imagens de referência das pastas Healthy_Foot e Non_Healthy_Foot
    reference_images = []
    for folder in reference_folders:
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                reference_images.append(img_resized)
    
    reference_images = np.array(reference_images) / 255.0

    for filename in os.listdir(test_folder):
        img_path = os.path.join(test_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (128, 128))
            test_images.append(img_resized)
            test_filenames.append(filename)
    
    test_images = np.array(test_images) / 255.0
    predictions = model.predict(test_images)

    for i, img in enumerate(test_images):
        # Verificar se a imagem em test_images é similar a qualquer imagem nas pastas de referência
        pe_humano = False
        for ref_img in reference_images:
            similarity = np.mean(np.abs(img - ref_img))  # Calcular a similaridade
            if similarity < 0.1:  # Ajuste o limiar conforme necessário  Para cada imagem de teste, 
                                  # calcula a similaridade média absoluta entre a imagem de teste e as imagens de referência. 
                                  # Se a similaridade for maior que um limiar (0.1), a imagem é reconhecida como um pé.
                pe_humano = True
                break
        
        if not pe_humano: # Ou seja, se a similiradidade for menor que a definida na Linha 151: "em  "if similarity < 0.1" 
            label = "Essa imagem não é um pé."
        else:
            label = "Não Saudável" if predictions[i] > 0.5 else "Saudável"
        
        img_display = (img * 255).astype(np.uint8)
        
        # Exibir a imagem
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.title(f"{test_filenames[i]}: {label}")
        plt.axis('off')
        plt.show()