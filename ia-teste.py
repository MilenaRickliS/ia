import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

# Dados 
produtos = [
    {
        "id": '1', 
        "nome": "Shampoo Hidratante", 
        "marca": "Marca A", 
        "imagem": "assets/shampoo-hidratante.png", 
        "preco": '29.99', 
        "categoria": "Cabelos", 
        "descricao": "Shampoo hidratante para cabelos secos e danificados, promovendo brilho e maciez.", 
        "avaliacoes": '4.5', 
        "sexo": "neutro", 
        "infantil": "não"
    },
    {
        "id": '2', 
        "nome": "Creme Hidratante Corporal", 
        "marca": "Marca B", 
        "imagem": "assets/creme-hidratante.png", 
        "preco": '49.90', 
        "categoria": "Corpo", 
        "descricao": "Creme hidratante corporal com fórmula enriquecida para pele extra-seca.", 
        "avaliacoes": '4.8', 
        "sexo": "feminino", 
        "infantil": "não"
    },
    {
        "id": '3', 
        "nome": "Protetor Solar FPS 50", 
        "marca": "Marca C", 
        "imagem": "assets/protetor-50.png", 
        "preco": '59.90', 
        "categoria": "Rosto", 
        "descricao": "Protetor solar para o rosto com alta proteção contra raios UVA e UVB.", 
        "avaliacoes": '4.7', 
        "sexo": "neutro", 
        "infantil": "não"
    },
    {
        "id": '4', 
        "nome": "Shampoo Baby", 
        "marca": "Marca D", 
        "imagem": "assets/shampoo-baby.png", 
        "preco": '19.90', 
        "categoria": "Cabelos", 
        "descricao": "Shampoo para bebês, suave e sem lágrimas, ideal para peles sensíveis.", 
        "avaliacoes": '4.9', 
        "sexo": "neutro", 
        "infantil": "sim"
    },
    {
        "id": '5', 
        "nome": "Perfume Floral", 
        "marca": "Marca E", 
        "imagem": "assets/perfume-floral.png", 
        "preco": '139.90', 
        "categoria": "Perfumes", 
        "descricao": "Perfume floral com notas de rosas e jasmins, ideal para o dia a dia.", 
        "avaliacoes": '4.3', 
        "sexo": "feminino", 
        "infantil": "não"
    },
    {
        "id": '6', 
        "nome": "Desodorante Masculino", 
        "marca": "Marca F", 
        "imagem": "assets/desodorante-masc.png", 
        "preco": '29.90', 
        "categoria": "Higiene Pessoal", 
        "descricao": "Desodorante com fragrância masculina de longa duração e proteção contra suor.", 
        "avaliacoes": '4.6', 
        "sexo": "masculino", 
        "infantil": "não"
    }
]

df = pd.DataFrame(produtos)

le = LabelEncoder()
df['sexo_encoded'] = le.fit_transform(df['sexo'])
df['infantil_encoded'] = df['infantil'].map({'não': 0, 'sim': 1})
df['categoria_encoded'] = le.fit_transform(df['categoria'])

df['avaliacoes'] = pd.to_numeric(df['avaliacoes'])
df['preco'] = pd.to_numeric(df['preco'])

X = df[['preco', 'avaliacoes', 'sexo_encoded', 'infantil_encoded']]
y = df['categoria_encoded']

# modelos
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando os modelos 
dt_model.fit(X, y)
rf_model.fit(X, y)

# previsões
dt_preds = dt_model.predict(X)
rf_preds = rf_model.predict(X)

# Calculando a acurácia dos modelos
dt_accuracy = accuracy_score(y, dt_preds)
rf_accuracy = accuracy_score(y, rf_preds)

# Baseline Heurística de Gênero
def recomendacao_heuristica_genero(sexo):
    if sexo == 'feminino':
        return "Perfumes, Corpo"
    elif sexo == 'masculino':
        return "Higiene Pessoal"
    else:
        return "Qualquer categoria"

# Baseline Aleatória de Preço
def recomendacao_aleatoria_preco(preco_max):
    produtos_baratos = df[df['preco'] <= preco_max]
    produto_recomendado = random.choice(produtos_baratos['nome'].tolist())
    return produto_recomendado

# simulação
# usuario_sexo = 'neutro'
# usuario_preco_max = 20.00

def chatbot_recomendacao():
    print("Olá! Bem-vindo(a) ao nosso sistema de recomendação de produtos de beleza. 😊")

    # Perguntar o sexo do usuário
    sexo = input("Qual é o seu sexo? (masculino, feminino, neutro): ").strip().lower()
    while sexo not in ['masculino', 'feminino', 'neutro']:
        print("Opção inválida. Por favor, digite 'masculino', 'feminino' ou 'neutro'.")
        sexo = input("Qual é o seu sexo? (masculino, feminino, neutro): ").strip().lower()

    # Perguntar o preço máximo desejado
    try:
        preco_max = float(input("Qual o preço máximo que você gostaria de pagar por um produto? (ex: 50.00): ").strip())
    except ValueError:
        print("Valor inválido. Definindo preço máximo como R$ 50.00 por padrão.")
        preco_max = 50.0

    # Recomendação com base no gênero
    recomendacao_genero = recomendacao_heuristica_genero(sexo)
    print(f"\nCom base no seu gênero, recomendamos as categorias: {recomendacao_genero}.")

    # Recomendação com base no preço
    try:
        recomendacao_preco = recomendacao_aleatoria_preco(preco_max)
        print(f"Com base no seu preço máximo, recomendamos o produto: {recomendacao_preco}.")
    except IndexError:
        print("Não encontramos produtos abaixo desse preço. Tente aumentar um pouco o valor.")

    print("\nObrigado por usar nosso assistente de beleza! 💄🧴")

# Executar o chatbot
chatbot_recomendacao()


# # Recomendação heurística de gênero
# recomendacao_genero = recomendacao_heuristica_genero(usuario_sexo)
# # Recomendação aleatória de preço
# recomendacao_preco = recomendacao_aleatoria_preco(usuario_preco_max)

# print(f"Recomendação Heurística de Gênero: {recomendacao_genero}")
# print(f"Recomendação Aleatória de Preço: {recomendacao_preco}")

# comparação entre os modelos 
models = ['Árvore de Decisão', 'Floresta Aleatória']
accuracies = [dt_accuracy, rf_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia entre Modelos de Recomendação (Usando Todos os Dados)')
plt.ylim(0, 1)
plt.show()

print(f"Acurácia da Árvore de Decisão: {dt_accuracy:.2f}")
print(f"Acurácia da Floresta Aleatória: {rf_accuracy:.2f}")