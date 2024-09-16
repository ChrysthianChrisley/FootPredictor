# main.py

import os
from data_preprocessing import verificar_pastas, carregar_imagens_das_pastas, normalizar_imagens, dividir_dados
from model_utils import pegar_melhor_modelo, train_model, Fazer_previsao_e_exibir_resultados

# Caminhos das pastas
pasta_raiz = os.path.dirname(__file__)
pasta_saudaveis = os.path.join(pasta_raiz, "Healthy_Foot")
pasta_nao_saudaveis = os.path.join(pasta_raiz, "Non_Healthy_Foot")
pasta_para_testar_imagens = os.path.join(pasta_raiz, "Mixed_Foot")
pasta_modelos = os.path.join(pasta_raiz, "Modelos_treinados")

# Parâmetros
definir_tamanho_imagem = (128, 128) # Padroniza as imagens em 128x128 pixels

# Verificar e criar diretórios, se necessário
verificar_pastas(pasta_saudaveis)
verificar_pastas(pasta_nao_saudaveis)
verificar_pastas(pasta_para_testar_imagens)
verificar_pastas(pasta_modelos)

# Carregar as imagens de treino e aplicar rótulos (0 ou 1)
imagens_saudaveis, rotulos_saudaveis = carregar_imagens_das_pastas(pasta_saudaveis, 0) # 0 será o rótulo para Healthy | Saudáveis
imagens_nao_saudaveis, rotulos_nao_saudaveis = carregar_imagens_das_pastas(pasta_nao_saudaveis, 1) # 1 será o rótulo para Non_Healthy | Não Saudáveis

# Preparar os dados para treinamento
X = imagens_saudaveis + imagens_nao_saudaveis
y = rotulos_saudaveis + rotulos_nao_saudaveis
X = normalizar_imagens(X)

# Dividir os dados em treino e validação
X_train, X_val, y_train, y_val = dividir_dados(X, y) # O conjunto de validação será usado para testar o desempenho do modelo após cada iteração de treino.

# Carregar o melhor modelo salvo
modelo, escolher_melhor_modelo = pegar_melhor_modelo(pasta_modelos) # Melhor modelo com menor erro de validação.

if not escolher_melhor_modelo:
    print("\nNão há modelos treinados. Um novo modelo será treinado.\n")
else:
    print(f"\nO melhor modelo ({os.path.basename(escolher_melhor_modelo)}) será utilizado.\n")

# Treinar ou continuar treinando o modelo
modelo = train_model(X_train, y_train, X_val, y_val, definir_tamanho_imagem, model=modelo)

# Prever e exibir as imagens da pasta "Mixed_Foot"
Fazer_previsao_e_exibir_resultados(modelo, pasta_para_testar_imagens, [pasta_saudaveis, pasta_nao_saudaveis])
