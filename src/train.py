import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_processed_data():
    """Carrega os arquivos CSV gerados pelo pré-processamento (tipos enxutos para 32 GB RAM)."""
    logger.info("Carregando dados processados da base completa...")
    try:
        X_train = pd.read_csv(DATA_DIR / "processed_X_train.csv").astype(np.float32)
        y_train = pd.read_csv(DATA_DIR / "processed_y_train.csv").values.ravel()
        X_test = pd.read_csv(DATA_DIR / "processed_X_test.csv").astype(np.float32)
        y_test = pd.read_csv(DATA_DIR / "processed_y_test.csv").values.ravel()
        logger.info(f"Dados carregados (float32). Treino: {X_train.shape}, Teste: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Arquivos não encontrados. Rode o preprocessamento primeiro! Erro: {e}")
        raise

def train_and_evaluate(model_name, clf, X_train, y_train, X_test, y_test):
    """
    Executa a Validação Cruzada (3 folds), treina o modelo final e faz a avaliação no teste.
    """
    logger.info(f"--- Iniciando processamento para: {model_name} ---")
    
    # 1. Validação Cruzada (3 Folds)
    logger.info(f"Executando Validação Cruzada (CV=3) para {model_name}...")
    scoring = ['f1_macro', 'recall', 'precision']
    
    # n_jobs=1 evita múltiplas cópias dos dados em RAM (seguro para 32 GB)
    cv_results = cross_validate(clf, X_train, y_train, cv=3, scoring=scoring, n_jobs=1)
    
    logger.info(f"Resultados CV (Média de 3 folds) - {model_name}:")
    logger.info(f" -> F1-Macro Médio:  {np.mean(cv_results['test_f1_macro']):.4f} (+/- {np.std(cv_results['test_f1_macro']):.4f})")
    logger.info(f" -> Recall Médio:    {np.mean(cv_results['test_recall']):.4f}")
    logger.info(f" -> Precision Média: {np.mean(cv_results['test_precision']):.4f}")

    # 2. Treinamento Final na base de treino completa
    logger.info(f"Treinando o modelo final {model_name}...")
    clf.fit(X_train, y_train)
    
    # 3. Avaliação no conjunto de Teste (Holdout)
    logger.info(f"Avaliando performance no conjunto de Teste ({model_name})...")
    y_pred = clf.predict(X_test)
    f1_macro_test = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n[{model_name}] --- Relatório de Classificação (Teste) ---")
    print(classification_report(y_test, y_pred, target_names=['Lícito', 'Fraude'], zero_division=0))
    print(f"  F1-Macro (teste): {f1_macro_test:.4f}")

    print(f"\n[{model_name}] --- Matriz de Confusão (Teste) ---")
    print(confusion_matrix(y_test, y_pred))

    return clf

def save_model(clf, filename):
    """Salva o modelo treinado em disco."""
    model_path = MODEL_DIR / filename
    joblib.dump(clf, model_path)
    logger.info(f"Modelo salvo com sucesso em: {model_path}\n")

def main():
    logger.info("=== INICIANDO PIPELINE DE TREINAMENTO E VALIDAÇÃO ===")
    
    # Carregar dados
    X_train, y_train, X_test, y_test = load_processed_data()
    
    # Modelos com n_jobs limitado e RF com max_samples para caber em 32 GB RAM
    models = {
        'rf': (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=2,
                class_weight='balanced',
                max_samples=0.25,  # 25% por árvore (reduz pico de memória em bases grandes)
            ),
            "random_forest_fraud.pkl"
        ),
        'lr': (
            "Logistic Regression",
            LogisticRegression(random_state=42, n_jobs=1, class_weight='balanced', max_iter=1000),
            "logistic_regression.pkl"
        )
    }

    # Executa automaticamente para todos os modelos do dicionário
    for key, (name, clf, filename) in models.items():
        # Treina, faz CV (5 folds) e avalia
        trained_model = train_and_evaluate(name, clf, X_train, y_train, X_test, y_test)
        
        # Salva o arquivo .pkl
        save_model(trained_model, filename)

    logger.info("=== PIPELINE DE TREINAMENTO CONCLUÍDO ===")

if __name__ == "__main__":
    main()