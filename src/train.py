"""
Módulo de Treinamento (Mineração de Dados).
Responsável por treinar o modelo e salvar o artefato final.

Autor: Equipe do Projeto (Membro 2)
Data: 2024
"""

import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ==========================================
# CONFIGURAÇÕES
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" # Pasta para salvar o modelo treinado

def load_processed_data():
    """
    Carrega os arquivos CSV gerados pelo pré-processamento.
    """
    logger.info("Carregando dados processados...")
    try:
        X_train = pd.read_csv(DATA_DIR / "processed_X_train.csv")
        y_train = pd.read_csv(DATA_DIR / "processed_y_train.csv").values.ravel() # ravel() para array 1D
        X_test = pd.read_csv(DATA_DIR / "processed_X_test.csv")
        y_test = pd.read_csv(DATA_DIR / "processed_y_test.csv").values.ravel()
        
        logger.info(f"Dados carregados. Treino: {X_train.shape}, Teste: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Arquivos não encontrados. Rode o preprocess.py primeiro! Erro: {e}")
        raise

def train_model(X_train, y_train):
    """
    Configura e treina o modelo Random Forest.
    """
    logger.info("Iniciando treinamento do modelo Random Forest...")
    
    # CONFIGURAÇÃO DOS HIPERPARÂMETROS (Importante para o relatório)
    # n_estimators=100: Cria 100 árvores de decisão.
    # max_depth=None: As árvores crescem até o fim (cuidado com overfitting, mas ok aqui).
    # n_jobs=-1: Usa todos os núcleos do processador para ser mais rápido.
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' # Ajuda extra contra desbalanceamento
    )
    
    clf.fit(X_train, y_train)
    logger.info("Treinamento concluído.")
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Faz uma avaliação rápida para garantir que o modelo não está quebrado.
    A avaliação visual profunda será feita pelo Membro 4.
    """
    logger.info("Avaliando performance preliminar...")
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    logger.info(f"F1-Score (Equilíbrio Precisão/Recall): {f1:.4f}")
    
    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- Matriz de Confusão Simples ---")
    print(confusion_matrix(y_test, y_pred))

def save_model(clf):
    """
    Salva o modelo treinado para uso futuro.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "random_forest_fraud.pkl"
    
    logger.info(f"Salvando modelo em: {model_path}")
    joblib.dump(clf, model_path)
    logger.info("Modelo salvo com sucesso!")

def main():
    logger.info("--- INICIANDO PIPELINE DE TREINAMENTO ---")
    
    # 1. Carregar
    X_train, y_train, X_test, y_test = load_processed_data()
    
    # 2. Treinar
    model = train_model(X_train, y_train)
    
    # 3. Avaliar (Preliminar)
    evaluate_model(model, X_test, y_test)
    
    # 4. Salvar
    save_model(model)
    
    logger.info("--- PIPELINE DE TREINAMENTO CONCLUÍDO ---")

if __name__ == "__main__":
    main()