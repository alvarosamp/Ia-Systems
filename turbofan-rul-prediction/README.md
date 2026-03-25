# Turbofan RUL Prediction

Projeto base de predição de Remaining Useful Life (RUL) para motores turbofan usando o dataset NASA C-MAPSS.

## Objetivo
Construir uma pipeline inicial para:
- preparar os dados
- treinar um modelo baseline
- salvar artefatos
- registrar métricas no MLflow
- gerar inferência
- expor API

## Dataset esperado
Coloque estes arquivos em `data/raw/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

## Ordem de execução

### 1. Instalar dependências
```bash
pip install -r requirements.txt