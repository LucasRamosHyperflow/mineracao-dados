"""
Módulo de Pré-processamento de Dados para Detecção de Fraude.
Responsável por carregar, limpar, transformar e salvar os dados processados.

Autor: Equipe do Projeto (Membro 1)
Data: 2024
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
# CORREÇÃO AQUI: _name_ (dois underlines)
logger = logging.getLogger(__name__)

# Constantes de Caminhos (Compatível com Linux/Windows)
# Ajuste o caminho conforme onde você está rodando o script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# Certifique-se que o nome do arquivo CSV está correto aqui
RAW_FILE = DATA_DIR / "onlinefraud.csv"

# Constantes do Modelo
RANDOM_STATE = 42
TEST_SIZE = 0.20

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Carrega o dataset CSV bruto.
    """
    if not filepath.exists():
        logger.error(f"Arquivo não encontrado: {filepath}")
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    logger.info(f"Carregando dados de: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset carregado com sucesso. Shape inicial: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao ler CSV: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas irrelevantes e ruídos.
    """
    logger.info("Iniciando limpeza de dados...")
    
    # Colunas de alta cardinalidade (IDs) ou regras pré-existentes
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    
    # Verifica se as colunas existem antes de tentar remover
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    
    if existing_cols:
        df_cleaned = df.drop(columns=existing_cols)
        logger.info(f"Colunas removidas: {existing_cols}")
    else:
        df_cleaned = df.copy()
        logger.warning("Nenhuma coluna para remover foi encontrada.")

    return df_cleaned

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding na coluna 'type'.
    """
    logger.info("Aplicando One-Hot Encoding na coluna 'type'...")
    
    if 'type' not in df.columns:
        logger.error("Coluna 'type' não encontrada para encoding.")
        raise KeyError("Coluna 'type' ausente.")

    # drop_first=True evita a armadilha da multicolinearidade
    df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    logger.info(f"Encoding concluído. Novas colunas: {list(df_encoded.columns)}")
    return df_encoded

def balance_data(df: pd.DataFrame, target_col: str = 'isFraud') -> pd.DataFrame:
    """
    Realiza Undersampling da classe majoritária para balancear o dataset.
    """
    logger.info("Iniciando balanceamento de dados (Undersampling)...")
    
    count_fraud = len(df[df[target_col] == 1])
    count_normal = len(df[df[target_col] == 0])
    
    logger.info(f"Contagem Original -> Fraudes: {count_fraud}, Normais: {count_normal}")

    # Separar classes
    fraud_df = df[df[target_col] == 1]
    normal_df = df[df[target_col] == 0]

    # Amostrar a classe majoritária (Normal)
    # Se houver poucas fraudes, pegamos apenas essa quantidade dos normais
    if count_fraud < count_normal:
        normal_undersampled = normal_df.sample(n=count_fraud, random_state=RANDOM_STATE)
        # Concatenar e embaralhar
        df_balanced = pd.concat([fraud_df, normal_undersampled], axis=0)
    else:
        df_balanced = df.copy() # Se já estiver balanceado ou fraudes forem maioria

    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info(f"Shape após balanceamento: {df_balanced.shape}")
    return df_balanced

def split_and_save(df: pd.DataFrame, target_col: str = 'isFraud'):
    """
    Divide em Treino/Teste e salva em arquivos CSV separados.
    """
    logger.info("Dividindo dados em Treino e Teste...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify garante a mesma proporção de fraudes no treino e no teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Definir caminhos de saída
    output_files = {
        "X_train": DATA_DIR / "processed_X_train.csv",
        "X_test": DATA_DIR / "processed_X_test.csv",
        "y_train": DATA_DIR / "processed_y_train.csv",
        "y_test": DATA_DIR / "processed_y_test.csv"
    }

    logger.info("Salvando arquivos processados...")
    try:
        # Garante que a pasta existe
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(output_files["X_train"], index=False)
        X_test.to_csv(output_files["X_test"], index=False)
        y_train.to_csv(output_files["y_train"], index=False)
        y_test.to_csv(output_files["y_test"], index=False)
        
        logger.info("Arquivos salvos com sucesso na pasta data/!")
        for name, path in output_files.items():
            logger.info(f"{name}: {path}")
            
    except Exception as e:
        logger.error(f"Erro ao salvar arquivos: {e}")
        raise

def main():
    """
    Função principal que orquestra o pipeline.
    """
    logger.info("--- INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ---")
    
    # 1. Carregar
    try:
        df_raw = load_data(RAW_FILE)
    except FileNotFoundError:
        logger.error("Abortando pipeline: arquivo não encontrado.")
        return

    # 2. Limpar
    df_clean = clean_data(df_raw)
    
    # 3. Encoding
    df_encoded = encode_features(df_clean)
    
    # 4. Balancear
    df_balanced = balance_data(df_encoded)
    
    # 5. Salvar
    split_and_save(df_balanced)
    
    logger.info("--- PIPELINE CONCLUÍDO COM SUCESSO ---")

# CORREÇÃO AQUI TAMBÉM: _name_ e _main_ (dois underlines)
if __name__ == "__main__":
    main()