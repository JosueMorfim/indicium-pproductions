# Previsão de Notas IMDb 🎬

Este projeto foi desenvolvido como parte do desafio da Indicium.  
O objetivo é prever a nota de filmes no IMDb a partir de variáveis como duração, gênero, certificação, número de votos, faturamento e meta score.

## Estrutura do Repositório
- LH_CD_Josue_Burigo_Morfim : Notebook principal com EDA e modelagem.
- modelo_imdb.pkl :  Modelo treinado (Random Forest).
- colunas_treinadas.pkl : Lista de colunas usadas no treino.
- src/preprocessamento.py : Função para tratar dados novos.
- src/exemplo_previsao.py : Exemplo de como carregar e usar o modelo.

## Como instalar ( Windows )
Clone o repositório:
```bash
git clone https://github.com/JosueMorfim/indicium-pproductions.git
cd indicium-pproductions
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt
```
## Exemplo de uso

```bash
python src/exemplo_previsao.py
```
Previsão de saída: A nota prevista para The Shawshank Redemption é: 8.83

