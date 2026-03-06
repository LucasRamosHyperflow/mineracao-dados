"""
Módulo de Pré-processamento de Dados para Detecção de Fraude.
Responsável por carregar, limpar, transformar e salvar os dados processados.

Autor: Equipe do Projeto (Membro 1 e 2)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_FILE = DATA_DIR / "onlinefraud.csv"

# Constantes do Modelo
RANDOM_STATE = 42
TEST_SIZE = 0.20

def load_data(filepath: Path) -> pd.DataFrame:
    """Carrega o dataset CSV bruto (low_memory=False para dtype único e menos RAM)."""
    if not filepath.exists():
        logger.error(f"Arquivo não encontrado: {filepath}")
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    logger.info(f"Carregando dados de: {filepath}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Dataset carregado. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao ler CSV: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas irrelevantes e ruídos."""
    logger.info("Iniciando limpeza de dados...")
    
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    
    if existing_cols:
        df_cleaned = df.drop(columns=existing_cols)
        logger.info(f"Colunas removidas: {existing_cols}")
    else:
        df_cleaned = df.copy()
        logger.warning("Nenhuma coluna para remover foi encontrada.")

    return df_cleaned

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica One-Hot Encoding na coluna 'type'."""
    logger.info("Aplicando One-Hot Encoding na coluna 'type'...")
    
    if 'type' not in df.columns:
        logger.error("Coluna 'type' não encontrada para encoding.")
        raise KeyError("Coluna 'type' ausente.")

    df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True)
    # float32 reduz uso de RAM pela metade em máquinas com pouca memória
    for col in df_encoded.select_dtypes(include=[np.floating]).columns:
        df_encoded[col] = df_encoded[col].astype(np.float32)
    logger.info(f"Encoding concluído. Total de colunas: {len(df_encoded.columns)} (float32)")
    return df_encoded

def split_and_save(df: pd.DataFrame, target_col: str = 'isFraud'):
    """Divide em Treino/Teste e salva em arquivos CSV separados."""
    logger.info("Dividindo dados em Treino e Teste (Base Completa)...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify garante a mesma proporção de fraudes no treino e no teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    output_files = {
        "X_train": DATA_DIR / "processed_X_train.csv",
        "X_test": DATA_DIR / "processed_X_test.csv",
        "y_train": DATA_DIR / "processed_y_train.csv",
        "y_test": DATA_DIR / "processed_y_test.csv"
    }

    logger.info("Salvando arquivos processados...")
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(output_files["X_train"], index=False)
        X_test.to_csv(output_files["X_test"], index=False)
        y_train.to_csv(output_files["y_train"], index=False)
        y_test.to_csv(output_files["y_test"], index=False)
        
        logger.info("Arquivos salvos com sucesso na pasta data/!")
    except Exception as e:
        logger.error(f"Erro ao salvar arquivos: {e}")
        raise

def main():
    logger.info("--- INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ---")
    
    try:
        df_raw = load_data(RAW_FILE)
    except FileNotFoundError:
        logger.error("Abortando pipeline.")
        return

    df_clean = clean_data(df_raw)
    df_encoded = encode_features(df_clean)
    
    # ATENÇÃO: Pulamos o balanceamento aqui. A base vai inteira para o split!
    split_and_save(df_encoded)
    
    logger.info("--- PIPELINE CONCLUÍDO COM SUCESSO ---")

if __name__ == "__main__":
    main()