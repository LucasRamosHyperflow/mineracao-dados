## 📂 Pasta `data/` – Dataset do Projeto

Esta pasta armazena **todos os arquivos de dados** utilizados no projeto de detecção de fraudes.

### 1. Arquivo bruto necessário

- **Nome esperado:** `onlinefraud.csv`
- **Função:** base de transações financeiras com a coluna alvo `isFraud`.
- **Onde salvar:** diretamente dentro desta pasta `data/` (mesmo nível deste `README.md`).

Sem este arquivo, os scripts de pré-processamento e treinamento **não conseguem ser executados**.

### 2. Como obter o dataset

O arquivo pode ser baixado a partir do seguinte link fornecido pela equipe:

`https://ufubr-my.sharepoint.com/:f:/g/personal/lucasramos_ufu_br/IgCuW5RVIITaRKKl2OFsYBKRAfp4H57v6yMNjRXg5alsI_g?e=vsLIcf`

### 3. Arquivos gerados automaticamente

Após executar o script `src/preprocess.py`, serão criados nesta pasta:

- `processed_X_train.csv`
- `processed_X_test.csv`
- `processed_y_train.csv`
- `processed_y_test.csv`

Esses arquivos **não precisam ser baixados**: são **gerados automaticamente** a partir do `onlinefraud.csv`.
