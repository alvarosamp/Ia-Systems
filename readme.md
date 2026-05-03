# Ia-Systems — projetos de IA com foco em MLOps

![CI - Turbofan RUL](https://github.com/alvarosamp/Ia-Systems/actions/workflows/ci.yml/badge.svg?branch=main)

Monorepo com projetos de Machine Learning do tipo **ponta a ponta**: dados → validação → treino → avaliação → (quando aplicável) API → deploy → observabilidade.

A ideia aqui não é só “treinar um modelo”, e sim construir **sistemas de IA** com engenharia e disciplina de MLOps: reprodutibilidade, contratos de dados, automação, testes, métricas e caminhos claros para produção.

> Este repositório vai crescer: novos projetos serão adicionados mantendo padrões parecidos de estrutura e execução.

---

## Projetos (até agora)

| Projeto | O que é | Status | Tecnologias / destaques |
| --- | --- | ---: | --- |
| [turbofan-rul-prediction](turbofan-rul-prediction/) | Predição de **RUL** (Remaining Useful Life) em motores aeronáuticos (NASA C-MAPSS FD001) com API em produção | Estável | PyTorch (LSTM), MLflow, Hydra, Optuna, FastAPI, Docker, CI, observabilidade (Prometheus) |
| [oil-well-anomaly-detection](oil-well-anomaly-detection/) | Detecção de anomalias em poços offshore (dataset Petrobras **3W**), foco em **DHSV closure (classe 2)** | Fase 1 | Pipeline de dados com validação (Pandera), Hydra para configs, scripts/CLIs e base pronta para modelagem |
| [F1-AI-Assistent](F1-AI-Assistent/) | Decision support para corridas de F1 via simulação Monte Carlo calibrada (prob. de win/podium/top6/top10) | Sprint 3 completo | FastF1 + engenharia de features + modelos (XGBoost/LogReg) + calibração + dashboard Streamlit |

Cada pasta tem seu próprio README com detalhes, decisões técnicas e comandos.

---

## Como rodar (visão rápida)

Como os projetos são relativamente independentes, a forma recomendada é:

1. Entrar na pasta do projeto

```bash
cd turbofan-rul-prediction
# ou: cd oil-well-anomaly-detection
# ou: cd F1-AI-Assistent
```

1. Criar um virtualenv e instalar dependências do projeto

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
```

1. Rodar testes / pipeline conforme o README do projeto

- Turbofan: ver [turbofan-rul-prediction/README.md](turbofan-rul-prediction/README.md)
- Oil well: ver [oil-well-anomaly-detection/README.md](oil-well-anomaly-detection/README.md)
- F1: ver [F1-AI-Assistent/readme.md](F1-AI-Assistent/readme.md)

---

## Padrões e princípios (o “jeito Ia-Systems”)

Os projetos aqui tendem a seguir estes princípios:

- **Contratos de dados**: validação explícita (ex.: Pandera) para reduzir “bugs silenciosos”.
- **Configuração rastreável**: Hydra (configs versionáveis) quando faz sentido.
- **Rastreabilidade de experimentos**: MLflow para params, métricas e artefatos.
- **Reprodutibilidade**: pipelines determinísticos sempre que possível + scripts claros para baixar/preparar dados.
- **Testes**: pytest para garantir não-regressão (incluindo testes anti-leakage quando aplicável).
- **Caminho para produção**: quando o projeto exige, inclui API (FastAPI), containerização (Docker) e CI/CD.

---

## Adicionando novos projetos

Quando você adicionar um novo projeto ao monorepo, o que costuma manter tudo organizado é:

- Criar uma pasta na raiz (ex.: `meu-novo-projeto/`)
- Incluir um README próprio com:
  - problema + dados + métrica
  - como reproduzir (install → preparar dados → treinar → testar)
  - layout do projeto
  - decisões técnicas (e trade-offs)
- Preferir uma estrutura parecida com:

```text
meu-novo-projeto/
├── README.md
├── requirements.txt  (ou pyproject.toml)
├── configs/          (se usar Hydra)
├── src/
├── tests/
├── Dockerfile        (se tiver serving)
└── render.yaml       (se for deploy em Render)
```

Se quiser, eu também consigo padronizar um “template mínimo” novo (só o skeleton) pra você plugar rapidamente novos projetos.

---

## Notas sobre dados e licenças

- Cada projeto usa fontes/datasets diferentes (ex.: NASA C-MAPSS, Petrobras 3W, FastF1/Ergast). Consulte o README de cada projeto para detalhes e restrições.
- Nem todos os projetos compartilham a mesma licença no momento; use o README do projeto como referência.

