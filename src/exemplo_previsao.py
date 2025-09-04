import pandas as pd
import joblib
from preprocessamento import preprocessar_dados

# Carregar modelo e colunas
modelo = joblib.load("modelo_imdb.pkl")
colunas_treinadas = joblib.load("colunas_treinadas.pkl")

# Carregar novos dados
dados_novos = pd.DataFrame([{
    'Series_Title': 'The Shawshank Redemption',
    'Released_Year': '1994',
    'Certificate': 'A',
    'Runtime': '142 min',
    'Genre': 'Drama',
    'Overview': 'Two imprisoned men bond over a number of years...',
    'Meta_score': 80.0,
    'Director': 'Frank Darabont',
    'Star1': 'Tim Robbins',
    'Star2': 'Morgan Freeman',
    'Star3': 'Bob Gunton',
    'Star4': 'William Sadler',
    'No_of_Votes': 2343110,
    'Gross': '28,341,469'
}])

df_novos = pd.DataFrame(dados_novos)

# Preprocessar
df_tratado = preprocessar_dados(df_novos)

# Fazer previsão
y_pred = modelo.predict(df_tratado)
print(f"A nota prevista para {df_novos.Series_Title[0]} é: {y_pred[0]:.2f}")
