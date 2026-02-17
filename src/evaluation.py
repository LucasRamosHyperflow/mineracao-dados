"""
Módulo de Avaliação e Visualização de Resultados.
Gera gráficos profissionais (Matriz de Confusão, ROC, Feature Importance)
e salva na pasta de relatórios.

Autor: Equipe do Projeto (Membro 4)
"""

import sys
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Configuração de Estilo para Gráficos Acadêmicos
plt.style.use('ggplot') 
sns.set_context("paper", font_scale=1.2)

class ModelEvaluator:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.reports_dir = self.base_dir / "reports"
        self.figures_dir = self.reports_dir / "figures"
        
        # Garante que as pastas existem
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_artifacts(self):
        """Carrega os dados de teste e o modelo treinado."""
        self.logger.info("Carregando artefatos de teste...")
        try:
            self.X_test = pd.read_csv(self.data_dir / "processed_X_test.csv")
            self.y_test = pd.read_csv(self.data_dir / "processed_y_test.csv").values.ravel()
            
            model_path = self.model_dir / "random_forest_fraud.pkl"
            self.model = joblib.load(model_path)
            self.logger.info("Modelo e dados carregados com sucesso.")
        except FileNotFoundError as e:
            self.logger.error(f"Arquivo não encontrado: {e}")
            sys.exit(1)

    def plot_confusion_matrix(self):
        """Gera e salva a Matriz de Confusão."""
        self.logger.info("Gerando Matriz de Confusão...")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Lícito', 'Fraude'],
                    yticklabels=['Lícito', 'Fraude'])
        plt.title('Matriz de Confusão')
        plt.ylabel('Realidade')
        plt.xlabel('Predição do Modelo')
        
        save_path = self.figures_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Salvo em: {save_path}")

    def plot_feature_importance(self):
        """Gera gráfico das variáveis mais importantes para o modelo."""
        self.logger.info("Calculando Importância das Variáveis...")
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns
        
        # Criar DataFrame para ordenar
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', legend=False, palette='viridis')
        plt.title('Top 10 Variáveis Mais Importantes na Detecção')
        plt.xlabel('Importância (Gini)')
        
        save_path = self.figures_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Salvo em: {save_path}")

    def plot_roc_curve(self):
        """Gera a Curva ROC e calcula a área AUC."""
        self.logger.info("Gerando Curva ROC...")
        # Probabilidade de ser classe 1 (Fraude)
        y_probs = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        save_path = self.figures_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Salvo em: {save_path}")

    def run(self):
        self.load_artifacts()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_roc_curve()
        self.logger.info("=== GERAÇÃO DE RELATÓRIOS CONCLUÍDA ===")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()