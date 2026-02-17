# 🕵️‍♂️ Detecção de Fraudes em Sistemas de Pagamentos Online

Este projeto visa desenvolver um modelo de mineração de dados robusto para a detecção precoce de atividades fraudulentas em transações financeiras digitais. O foco principal é a redução de perdas financeiras e o aumento da confiabilidade das plataformas.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Índice
- [Visão Geral](#-visão-geral)
- [Objetivo](#-objetivo)
- [Dataset](#-dataset)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Metodologia](#-metodologia)
- [Resultados Alcançados](#-resultados-alcançados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Autores](#-autores)

---

## 🧐 Visão Geral

*Problema de Pesquisa:* A detecção precoce de atividades fraudulentas em sistemas de pagamentos online visando a redução de perdas financeiras e o aumento da confiança do usuário em plataformas digitais.

A fraude financeira é um problema crescente que custa bilhões anualmente. Este projeto utiliza técnicas avançadas de Machine Learning para identificar padrões suspeitos em tempo hábil, lidando especificamente com o desafio do desbalanceamento de classes (onde fraudes são eventos raros).

## 🎯 Objetivo

Desenvolver um modelo de classificação capaz de categorizar transações como *'Lícitas'* ou *'Fraudulentas'*, baseando-se em variáveis comportamentais e transacionais, tais como:
- Tipo de transação (ex: CASH_OUT, TRANSFER)
- Comportamento de saldo (Zerar a conta de origem)
- Discrepância de valores

*Tarefa de Mineração:* Classificação Supervisionada.

## 💾 Dataset
Utilizamos a base de dados pública do Kaggle:
* *Nome:* Online Payment Fraud Detection
* *Link:* [Kaggle Dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)
* *Instrução:* O arquivo PS_20174392719_1491204439457_log.csv deve ser colocado na pasta data/.

## 🛠 Tecnologias Utilizadas

O projeto foi desenvolvido em *Python* seguindo práticas de Engenharia de Software (OOP e Logging) para garantir compatibilidade com ambientes Linux/Ubuntu.

- *Linguagem:* Python 3.8+
- *Manipulação de Dados:* Pandas, NumPy
- *Visualização:* Matplotlib, Seaborn
- *Machine Learning:* Scikit-learn (Random Forest Classifier)
- *Balanceamento de Dados:* Imbalanced-learn (SMOTE)
- *Persistência de Modelo:* Joblib

## 🚀 Metodologia

O pipeline de dados foi implementado de forma sequencial e automatizada no script src/main.py, cobrindo as três etapas fundamentais:

### 1. Pré-processamento
Tratamento da "sujeira" dos dados e preparação para os algoritmos.
- *Limpeza:* Remoção de colunas de alta cardinalidade (IDs de usuários).
- *Otimização:* Downcasting de tipos numéricos para redução de uso de memória RAM.
- *Encoding:* Aplicação de One-Hot Encoding na variável categórica type.
- *Balanceamento:* Aplicação do algoritmo *SMOTE* (Synthetic Minority Over-sampling Technique) nos dados de treino, gerando fraudes sintéticas para equilibrar as classes 50/50.

### 2. Mineração de Dados (Modelagem)
Treinamento do modelo preditivo.
- *Algoritmo:* Random Forest Classifier.
- *Configuração:* 100 estimadores com pesos de classe balanceados.
- *Justificativa:* Escolhido por sua robustez contra overfitting e capacidade de capturar relações não-lineares entre saldo e valor da transação.

### 3. Pós-processamento (Avaliação)
Geração de métricas e gráficos para análise de negócio.
- *Matriz de Confusão:* Para visualizar Falsos Positivos vs. Falsos Negativos.
- *Feature Importance:* Para entender quais variáveis indicam fraude.
- *Curva ROC:* Para medir a qualidade da separação entre classes.

---

## 🏆 Resultados Alcançados

O modelo final apresentou desempenho excepcional no conjunto de teste (dados nunca vistos pelo modelo):

| Métrica | Resultado | Interpretação |
| :--- | :--- | :--- |
| *Recall (Fraude)* | *1.00 (100%)* | O modelo detectou *todas* as fraudes reais do teste. |
| *Precision* | *0.99* | De cada 100 alertas de fraude, 99 eram reais. |
| *F1-Score* | *0.99* | Equilíbrio perfeito entre precisão e recall. |

Os gráficos detalhados (Matriz de Confusão, ROC e Importância de Features) são gerados automaticamente na pasta reports/figures/ após a execução.

---

## 📂 Estrutura do Projeto

```text
projeto-fraude/
├── data/
│   └── onlinefraud.csv     # Dataset original (Baixar do Kaggle)
|   └── processed*.csv      # Dados tratados (gerados pelo script)
|   └── README.md           # Documentação de como baixar o dataset
├── models/                 # O modelo treinado (.pkl) será salvo aqui
├── reports/
│   └── figures/            # Os gráficos PNG serão salvos aqui
├── src/
│   ├── preprocessing.py    # Pipeline ETL e SMOTE
│   ├── train.py            # Treinamento do Modelo
│   └── evaluation.py       # Geração de Gráficos e Métricas
├── requirements.txt        # Dependências do Python
└── README.md               # Documentação
```
---

## 💻 Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado. É recomendado o uso de um ambiente virtual.

```bash
# Clone este repositório
$ git clone [https://github.com/LucasRamosHyperflow/mineracao-dados.git](https://github.com/LucasRamosHyperflow/mineracao-dados.git)

# Acesse a pasta do projeto
$ cd mineracao-dados

# Crie um ambiente virtual (Linux/Mac)
$python3 -m venv venv$ source venv/bin/activate

# Instale as dependências
$ pip install -r requirements.txt
```

# rodar os scripts principais
``` bash
$ python src/preprocess.py
$ python src/train.py
$ python src/evaluation.py
```


## 👥 Autores
Lucas Ramos Fernandes da Silva

Guilherme Costa Rodrigues

Matheus Ribeiro Prado

Adryell Medeiros