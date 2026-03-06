# 🕵️‍♂️ Detecção de Fraudes em Sistemas de Pagamentos Online

Este projeto desenvolve um pipeline de mineração de dados para detecção precoce de atividades fraudulentas em transações financeiras digitais. O foco principal é **reduzir perdas financeiras** e **aumentar a confiança** dos usuários em plataformas de pagamento online.

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

**Problema de pesquisa**: detecção precoce de atividades fraudulentas em sistemas de pagamentos online, visando a redução de perdas financeiras e o aumento da confiança dos usuários em plataformas digitais.

A fraude financeira é um problema crescente que custa bilhões anualmente. Este projeto utiliza técnicas de **Aprendizado de Máquina supervisionado** para identificar padrões suspeitos em tempo hábil, lidando com o desafio do **desbalanceamento de classes**, em que as transações fraudulentas são eventos raros.

## 🎯 Objetivo

Desenvolver modelos de classificação capazes de categorizar transações como **“Lícitas”** ou **“Fraudulentas”**, com base em variáveis comportamentais e transacionais, tais como:

- **Tipo de transação** (ex.: `CASH_OUT`, `TRANSFER`)
- **Comportamento de saldo** (por exemplo, zerar a conta de origem)
- **Discrepância de valores** entre saldo inicial, valor transferido e saldo final

**Tarefa de Mineração**: Classificação Supervisionada.

## 💾 Dataset

O projeto utiliza um conjunto de dados de transações financeiras com a coluna alvo `isFraud`.

- **Nome do arquivo esperado:** `onlinefraud.csv`
- **Local:** pasta `data/`
- **Instruções de download:** consulte o arquivo `data/README.md`, que contém o link oficial de obtenção do dataset.

> Certifique-se de colocar o arquivo `onlinefraud.csv` diretamente dentro da pasta `data/` na raiz do projeto.

## 🛠 Tecnologias Utilizadas

O projeto foi desenvolvido em **Python** com foco em compatibilidade com **Linux/Ubuntu** e boas práticas de engenharia (uso de `logging`, scripts modulares e reprodutibilidade).

- **Linguagem:** Python 3.8+
- **Manipulação de Dados:** `pandas`, `numpy`
- **Visualização:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
  - `RandomForestClassifier` (modelo de floresta aleatória)
  - `LogisticRegression` (modelo linear)
- **Persistência de Modelos:** `joblib`

## 🚀 Metodologia

O pipeline está dividido em **três etapas principais**, cada uma em um script separado na pasta `src/`:

- `src/preprocess.py` – pré-processamento e preparação dos dados
- `src/train.py` – treinamento e comparação dos modelos
- `src/evaluation.py` – avaliação detalhada e geração de gráficos

### 1. Pré-processamento (`src/preprocess.py`)

Etapa responsável por carregar, limpar, transformar e particionar os dados:

- **Carga de dados:** leitura do arquivo `data/onlinefraud.csv`.
- **Limpeza:** remoção de colunas de alta cardinalidade e pouco valor preditivo (`nameOrig`, `nameDest`, `isFlaggedFraud`), quando existirem.
- **Transformação / Encoding:** aplicação de **One-Hot Encoding** na coluna categórica `type`.
- **Otimização de memória:** conversão de colunas numéricas para `float32`, reduzindo uso de RAM.
- **Divisão treino/teste:** uso de `train_test_split` estratificado para preservar a proporção de fraudes.
- **Saída:** arquivos processados salvos em `data/`:
  - `processed_X_train.csv`
  - `processed_X_test.csv`
  - `processed_y_train.csv`
  - `processed_y_test.csv`

### 2. Mineração de Dados (Modelagem) (`src/train.py`)

Etapa de treinamento, validação e salvamento dos modelos.

Dois modelos são treinados e avaliados:

- **Random Forest** (`RandomForestClassifier`)
  - 100 árvores
  - `class_weight='balanced'` para lidar com desbalanceamento
  - `max_samples=0.25` para controlar uso de memória em bases grandes
  - `n_jobs=2` para paralelismo controlado
- **Regressão Logística** (`LogisticRegression`)
  - Modelo linear com `class_weight='balanced'`
  - `max_iter=1000` para garantir convergência

Para cada modelo, o script:

- Executa **Validação Cruzada (CV=3)** na base de treino com métricas:
  - `f1_macro`
  - `recall`
  - `precision`
- Treina um modelo final usando **todo o conjunto de treino**.
- Avalia no conjunto de teste e imprime no terminal:
  - Relatório de classificação (`classification_report`)
  - Matriz de confusão
  - F1-Macro no teste
- Salva os modelos treinados na pasta `models/`:
  - `models/random_forest_fraud.pkl`
  - `models/logistic_regression.pkl`

### 3. Pós-processamento (Avaliação) (`src/evaluation.py`)

O módulo de avaliação:

- Carrega novamente os dados processados de treino e teste.
- Carrega os modelos salvos em `models/`.
- Gera automaticamente, para **Random Forest** e **Regressão Logística**:
  - **Matrizes de Confusão** (treino e teste)
  - **Importância de Features**:
    - Random Forest: importância Gini
    - Regressão Logística: magnitude absoluta dos coeficientes
  - **Curvas ROC** e AUC no conjunto de teste
- Salva todos os gráficos em `reports/figures/`:
  - `confusion_matrix_<modelo>_<treino|teste>.png`
  - `feature_importance_<modelo>.png`
  - `roc_curve_<modelo>.png`

---

## 🏆 Resultados Alcançados

Os resultados são apresentados em duas frentes:

- **Textualmente (terminal):**
  - Relatórios de classificação com precisão, recall, F1-score por classe (`Lícito` e `Fraude`).
  - F1-Macro no conjunto de teste para cada modelo.
- **Visualmente (arquivos PNG):**
  - Matrizes de confusão para treino e teste.
  - Importância das variáveis.
  - Curvas ROC com AUC para comparação dos modelos.

De forma geral:

- **Random Forest** tende a capturar relações não lineares entre variáveis, alcançando **alto recall para a classe “Fraude”**, o que reduz falsos negativos (fraudes que passam sem ser detectadas).
- **Regressão Logística (modelo linear)** atua como um **baseline interpretável**, permitindo inspecionar diretamente o peso (coeficiente) de cada variável na decisão do modelo.

Isso permite à equipe comparar um modelo mais complexo (Random Forest) com um modelo linear (Regressão Logística), equilibrando **desempenho preditivo** e **interpretabilidade**.

---

## 📂 Estrutura do Projeto

Estrutura atual do repositório:

```text
mineracao-dados/
├── data/
│   ├── README.md                # Instruções de download do dataset
│   ├── .gitignore               # Ignora CSVs grandes e arquivos derivados
│   ├── onlinefraud.csv          # Dataset bruto (NÃO versionado, colocar manualmente)
│   ├── processed_X_train.csv    # Dados de treino (features) gerados por preprocess.py
│   ├── processed_X_test.csv     # Dados de teste (features)
│   ├── processed_y_train.csv    # Rótulos de treino
│   └── processed_y_test.csv     # Rótulos de teste
├── models/
│   ├── random_forest_fraud.pkl  # Modelo Random Forest treinado
│   └── logistic_regression.pkl  # Modelo de Regressão Logística treinado
├── reports/
│   └── figures/                 # Gráficos PNG gerados em evaluation.py
├── src/
│   ├── preprocess.py            # Pipeline de pré-processamento e split
│   ├── train.py                 # Treinamento, validação cruzada e salvamento dos modelos
│   └── evaluation.py            # Avaliação detalhada e geração de gráficos
├── requirements.txt             # Dependências Python do projeto
├── references.txt               # Referências e anotações de apoio
├── reuniao.txt                  # Notas de reunião / planejamento
└── README.md                    # Este arquivo
```

> As pastas `models/` e `reports/figures/` são criadas automaticamente pelos scripts, caso ainda não existam.

---

## 💻 Como Executar

### Ambiente recomendado

- **SO alvo para execução dos scripts:** Linux/Ubuntu (por exemplo, Ubuntu 20.04+)
- **Ferramentas:** `git`, `python3`, `pip`, acesso a `sudo` para instalar pacotes, se necessário

### 1. Verificar/instalar Python 3 (Ubuntu)

Abra um terminal no Ubuntu e execute:

```bash
# Verificar se python3 está instalado
if ! command -v python3 &>/dev/null; then
  echo "Python 3 não encontrado. Instalando..."
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip
else
  echo "Python 3 já está instalado."
fi
```

### 2. Clonar o repositório

```bash
git clone https://github.com/LucasRamosHyperflow/mineracao-dados.git
cd mineracao-dados
```

### 3. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Colocar o dataset na pasta `data/`

- Baixe o arquivo `onlinefraud.csv` conforme instruções em `data/README.md`.
- Copie o arquivo para a pasta `data/` na raiz do projeto.

### 5. Executar o pipeline completo

No terminal (a partir da raiz do projeto):

```bash
# 1) Pré-processamento: gera os CSVs processados em data/
python3 src/preprocess.py

# 2) Treinamento: Random Forest e Regressão Logística,
#    com validação cruzada e salvamento em models/
python3 src/train.py

# 3) Avaliação: gera matrizes de confusão, importância de features e ROC em reports/figures/
python3 src/evaluation.py
```

---

## 👥 Autores

- Adryell Alexandre Medeiros
- Guilherme Costa Rodrigues
- Lucas Ramos Fernandes da Silva
- Matheus Ribeiro Prado
