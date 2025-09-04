# Previsão de Notas IMDb 🎬

Este projeto foi desenvolvido como parte do desafio da Indicium.  
O objetivo é prever a nota de filmes no IMDb a partir de variáveis como duração, gênero, certificação, número de votos, faturamento e meta score.

## 📂 Estrutura do Repositório
- `PProductions.ipynb` → Notebook principal com EDA e modelagem.
- `modelo_imdb.pkl` → Modelo treinado (Random Forest).
- `colunas_treinadas.pkl` → Lista de colunas usadas no treino.
- `src/preprocessamento.py` → Função para tratar dados novos.
- `src/previsao_exemplo.py` → Exemplo de como carregar e usar o modelo.

## ⚙️ Como instalar
Clone o repositório:
```bash
git clone https://github.com/seu-usuario/indicium-imdb.git
cd indicium-imdb
