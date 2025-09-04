import pandas as pd
import joblib

def preprocessar_dados(df: pd.DataFrame) -> pd.DataFrame:

    # Criar Runtime_min e remover Runtime, se existir
    if "Runtime" in df.columns:
        df["Runtime_min"] = df["Runtime"].apply(lambda x: str(x).split(" ")[0]).astype(int)
        df = df.drop("Runtime", axis=1)
    
    # Criar main_genre a partir da coluna Genre
    if "Genre" in df.columns:
        df["main_genre"] = df["Genre"].apply(lambda x: str(x).split(",")[0])
    
    # Tratar coluna Certificate
    if "Certificate" in df.columns:
        df["Certificate"] = df["Certificate"].fillna("Unknown")
    
    # Tratar Released_Year
    if "Released_Year" in df.columns:
        df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce").fillna(1995)
    
    # Tratar Gross (remover vírgula e converter)
    if "Gross" in df.columns:
        df["Gross"] = df["Gross"].astype(str).str.replace(",", "", regex=True)
        df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")
    
    # Selecionar apenas as features usadas no treino
    features = ["Meta_score", "No_of_Votes", "Gross", "Runtime_min", "main_genre", "Certificate"]
    df = df[features]
    
    # Aplicar get_dummies (categorias)
    df_encoded = pd.get_dummies(df, columns=["main_genre", "Certificate"], drop_first=True)
    
    # Reindexar para garantir mesmas colunas do treino
    colunas_treinadas = joblib.load("colunas_treinadas.pkl")
    df_final = df_encoded.reindex(columns=colunas_treinadas, fill_value=0)
    
    return df_final
