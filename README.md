# 🕵️‍♂️ Detecção de Fraudes em Sistemas de Pagamentos Online

Este projeto visa desenvolver um modelo de mineração de dados robusto para a detecção precoce de atividades fraudulentas em transações financeiras digitais. O foco principal é a redução de perdas financeiras e o aumento da confiabilidade das plataformas.

## 📋 Índice
- [Visão Geral](#-visão-geral)
- [Objetivo](#-objetivo)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Metodologia](#-metodologia)
  - [1. Pré-processamento](#1-pré-processamento)
  - [2. Mineração de Dados (Modelagem)](#2-mineração-de-dados-modelagem)
  - [3. Pós-processamento (Avaliação)](#3-pós-processamento-avaliação)
- [Como Executar](#-como-executar)
- [Autores](#-autores)

---

## 🧐 Visão Geral

*Problema de Pesquisa:* A detecção precoce de atividades fraudulentas em sistemas de pagamentos online visando a redução de perdas financeiras e o aumento da confiança do usuário em plataformas digitais.

A fraude financeira é um problema crescente que custa bilhões anualmente. Este projeto utiliza técnicas avançadas de Machine Learning para identificar padrões suspeitos em tempo hábil.

## 🎯 Objetivo

Desenvolver um modelo de classificação capaz de categorizar transações como *'Lícitas'* ou *'Fraudulentas'*, baseando-se em variáveis comportamentais e transacionais, tais como:
- Tipo de transação (ex: CASH_OUT, TRANSFER)
- Valor da transação
- Balanço da conta (origem e destino)

*Tarefa de Mineração:* Classificação Supervisionada.

## 🛠 Tecnologias Utilizadas

O projeto foi desenvolvido em *Python* devido à sua vasta gama de bibliotecas para Ciência de Dados e compatibilidade com ambientes Linux/Ubuntu.

- *Linguagem:* Python 3.8+
- *Manipulação de Dados:* Pandas, NumPy
- *Visualização:* Matplotlib, Seaborn
- *Machine Learning:* Scikit-learn
- *Modelos Avançados:* XGBoost / LightGBM
- *Balanceamento de Dados:* Imbalanced-learn (SMOTE)

## 🚀 Metodologia

O fluxo de trabalho foi dividido em três etapas estratégicas:

### 1. Pré-processamento
Nesta etapa, tratamos a "sujeira" dos dados e preparamos o terreno para os algoritmos.
- *Limpeza:* Tratamento de valores nulos (missing values) e remoção de duplicatas.
- *Codificação (Encoding):* Transformação de variáveis categóricas em numéricas (ex: LabelEncoder ou OneHotEncoder para o tipo de transação).
- *Tratamento de Desbalanceamento:* Aplicação de técnicas como *SMOTE* (Synthetic Minority Over-sampling Technique) ou Undersampling, dado que fraudes são eventos raros em comparação a transações lícitas.
- *Escalonamento:* Normalização de variáveis contínuas (como o valor da transação) para evitar viés em modelos sensíveis à escala.

### 2. Mineração de Dados (Modelagem)
Foram selecionados e testados diferentes algoritmos para comparação de desempenho:
- *Random Forest:* Escolhido por sua robustez e capacidade de detectar padrões não-lineares complexos.
- *XGBoost / LightGBM:* Modelos baseados em Gradient Boosting, estado da arte em competições de detecção de fraude devido à alta performance e velocidade.
- *Regressão Logística:* Utilizado como baseline (linha de base) para validar se os modelos complexos estão realmente agregando valor.

### 3. Pós-processamento (Avaliação)
A acurácia não é uma métrica confiável em dados desbalanceados. O foco da avaliação está em:
- *Recall (Sensibilidade):* Prioridade máxima. Quantas fraudes reais o modelo conseguiu capturar?
- *F1-Score:* O equilíbrio harmônico entre precisão e recall.
- *Matriz de Confusão:* Visualização clara dos Falsos Positivos vs. Falsos Negativos.
- *Curva ROC/AUC:* Medição da capacidade do modelo de distinguir entre as classes.

---

## 💻 Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado. É recomendado o uso de um ambiente virtual.

```bash
# Clone este repositório
$ git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)

# Acesse a pasta do projeto
$ cd nome-do-repositorio

# Crie um ambiente virtual (Linux/Mac)
$python3 -m venv venv$ source venv/bin/activate

# Instale as dependências
$ pip install -r requirements.txt
```
# Para rodar via Jupyter Notebook
```bash
$ jupyter notebook
```
# Ou para rodar o script principal
```bash
$ python src/main.py
```

## 👥 Autores
Lucas Ramos Fernandes da Silva

Guilherme Costa Rodrigues

Matheus Ribeiro Prado

Adryell Medeiros