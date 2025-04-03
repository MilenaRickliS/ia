import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = [
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

df = pd.DataFrame(data)

df['avaliacoes'] = pd.to_numeric(df['avaliacoes'])
df['preco'] = pd.to_numeric(df['preco'])


le = LabelEncoder()
df['sexo_encoded'] = le.fit_transform(df['sexo'])

def recomendar_produto(categoria, max_preco, min_avaliacao, sexo, infantil):
    # Filter the products based on user input
    filtered_df = df[
        (df['categoria'] == categoria) &
        (df['preco'] <= max_preco) &
        (df['avaliacoes'] >= min_avaliacao) &
        (df['sexo'] == sexo) &
        (df['infantil'] == infantil)
    ]
    
    # If no products match, recommend a generic message
    if filtered_df.empty:
        return "Nenhum produto encontrado com esses critérios. Tente alterar as preferências."

    # Use the trained model to recommend products (For simplicity, we will show the first match)
    produto_recomendado = filtered_df.iloc[0]['nome']
    return f"Produto recomendado: {produto_recomendado}"

# Simulate 
categoria_usuario = 'Cabelos'
max_preco_usuario = 30.00
min_avaliacao_usuario = 4.5
sexo_usuario = 'neutro'
infantil_usuario = 'não'

# Get recommendation
produto = recomendar_produto(categoria_usuario, max_preco_usuario, min_avaliacao_usuario, sexo_usuario, infantil_usuario)
print(produto)
