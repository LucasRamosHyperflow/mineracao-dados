import gc
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

# Configuração de Estilo
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)

class ModelEvaluator:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.figures_dir = self.base_dir / "reports" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        self.X_train = self.y_train = self.X_test = self.y_test = None

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_train_data(self):
        """Carrega só dados de treino (float32 para economizar RAM)."""
        self.logger.info("Carregando dados de TREINO...")
        try:
            self.X_train = pd.read_csv(self.data_dir / "processed_X_train.csv").astype(np.float32)
            self.y_train = pd.read_csv(self.data_dir / "processed_y_train.csv").values.ravel()
            self.logger.info(f"Treino carregado: {self.X_train.shape}")
        except FileNotFoundError as e:
            self.logger.error(f"Arquivos não encontrados: {e}")
            sys.exit(1)

    def load_test_data(self):
        """Carrega só dados de teste (float32 para economizar RAM)."""
        self.logger.info("Carregando dados de TESTE...")
        try:
            self.X_test = pd.read_csv(self.data_dir / "processed_X_test.csv").astype(np.float32)
            self.y_test = pd.read_csv(self.data_dir / "processed_y_test.csv").values.ravel()
            self.logger.info(f"Teste carregado: {self.X_test.shape}")
        except FileNotFoundError as e:
            self.logger.error(f"Arquivos não encontrados: {e}")
            sys.exit(1)

    def free_train_data(self):
        """Libera dados de treino da memória."""
        self.X_train = None
        self.y_train = None
        gc.collect()
        self.logger.info("Dados de treino liberados da RAM.")

    def _plot_confusion_matrix_one(self, model, model_key, model_name, X, y, subset_name):
        """Gera uma matriz de confusão para um conjunto (treino ou teste)."""
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Lícito', 'Fraude'],
                    yticklabels=['Lícito', 'Fraude'])
        plt.title(f'Matriz de Confusão - {model_name} (conjunto {subset_name})\nF1-Macro = {f1_macro:.4f}')
        plt.ylabel('Realidade')
        plt.xlabel('Predição')
        save_path = self.figures_dir / f"confusion_matrix_{model_key}_{subset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  F1-Macro ({subset_name}): {f1_macro:.4f}")

    def evaluate_model_train(self, model_key, model_name, filename):
        """Gera apenas a matriz de confusão no conjunto de TREINO (requer load_train_data)."""
        if self.X_train is None:
            self.logger.warning("Chame load_train_data() antes.")
            return
        model_path = self.model_dir / filename
        if not model_path.exists():
            self.logger.warning(f"Modelo não encontrado: {model_path}. Pulando {model_name}...")
            return
        self.logger.info(f"Matriz de confusão (treino): {model_name}")
        model = joblib.load(model_path)
        self._plot_confusion_matrix_one(model, model_key, model_name, self.X_train, self.y_train, "treino")

    def evaluate_model_test(self, model_key, model_name, filename):
        """Gera matriz de confusão, importância e ROC no conjunto de TESTE (requer load_test_data)."""
        if self.X_test is None:
            self.logger.warning("Chame load_test_data() antes.")
            return
        model_path = self.model_dir / filename
        if not model_path.exists():
            self.logger.warning(f"Modelo não encontrado: {model_path}. Pulando {model_name}...")
            return
        self.logger.info(f"--- AVALIAÇÃO (teste): {model_name} ---")
        model = joblib.load(model_path)
        self._plot_confusion_matrix_one(model, model_key, model_name, self.X_test, self.y_test, "teste")
        self.plot_feature_importance(model, model_key, model_name)
        self.plot_roc_curve(model, model_key, model_name)
        self.logger.info(f"--- CONCLUÍDO: {model_name} ---")

    def plot_feature_importance(self, model, model_key, model_name):
        self.logger.info(f"Calculando Importância das Variáveis para {model_key}...")
        feature_names = self.X_test.columns
        
        # A extração muda dependendo do algoritmo
        if model_key == 'rf':
            importances = model.feature_importances_
            label = 'Importância (Gini)'
        elif model_key == 'lr':
            # Na regressão logística, pegamos o valor absoluto dos coeficientes
            importances = np.abs(model.coef_[0])
            label = 'Magnitude Absoluta do Coeficiente'
            
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', legend=False, palette='viridis')
        plt.title(f'Top 10 Variáveis Mais Importantes - {model_name}')
        plt.xlabel(label)
        
        save_path = self.figures_dir / f"feature_importance_{model_key}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, model, model_key, model_name):
        self.logger.info(f"Gerando Curva ROC para {model_key}...")
        y_probs = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend(loc="lower right")
        
        save_path = self.figures_dir / f"roc_curve_{model_key}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    models = {
        'rf': ("Random Forest", "random_forest_fraud.pkl"),
        'lr': ("Logistic Regression", "logistic_regression.pkl")
    }

    # Fase 1: só treino na RAM → matriz de confusão (treino)
    evaluator.load_train_data()
    for key, (name, filename) in models.items():
        evaluator.evaluate_model_train(key, name, filename)
    evaluator.free_train_data()

    # Fase 2: só teste na RAM → matriz de confusão (teste) + importância + ROC
    evaluator.load_test_data()
    for key, (name, filename) in models.items():
        evaluator.evaluate_model_test(key, name, filename)