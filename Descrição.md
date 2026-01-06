_# Desafio Técnico - Previsão de Preços de Ações com LSTM_

## Descrição

Este projeto implementa um modelo de _deep learning_ utilizando uma rede neural **Long Short-Term Memory (LSTM)** para prever o preço de fechamento de ações. A solução completa abrange desde a coleta e pré-processamento de dados até o _deploy_ do modelo em uma **API RESTful**.

## Requisitos

- Python 3.8+
- pip
- Docker

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone <URL do repositório>
   cd <nome do repositório>
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para treinar o modelo, execute o _notebook_ `notebooks/treinamento_modelo.ipynb`.

Para iniciar a API, execute o seguinte comando:

```bash
docker-compose up --build
```

## Endpoints da API

Base local: `http://localhost:5000`

- `GET /` (health)
  - Retorna status e endpoints disponíveis.

- `POST /predict`

  **Corpo da Requisição:**

  ```json
  {
    "symbol": "PETR4.SA", 
    "data": [
      {"Date": "2025-10-10T00:00:00", "Close": 30.10},
      {"Date": "2025-10-13T00:00:00", "Close": 30.20}
      /* ... total 60+ itens ... */
    ]
  }
  ```

  **Resposta de Sucesso:**

  ```json
  {
    "prediction": 153.45,
    "prediction_date": "2026-01-05T00:00:00",
    "future_target": 154.12,
    "future_date": "2026-01-06T00:00:00",
    "used_count": 60,
    "used_last_date": "2025-12-26T00:00:00"
  }
  ```

- `GET /predict/ticker?symbol=PETR4.SA&period=180d&interval=1d`
  - Busca fechamentos via yfinance e retorna a previsão.
  - Alternativa com datas: `GET /predict/ticker?symbol=PETR4.SA&start=2024-01-01&end=2025-12-26&interval=1d`
  - Dica: se faltar histórico suficiente, aumente `period` (ex.: `1y`).
  - Resposta típica:
  ```json
  {
    "symbol": "PETR4.SA",
    "prediction": 30.85,
    "prediction_date": "2025-12-29T00:00:00",
    "future_target": 31.02,
    "future_date": "2025-12-30T00:00:00",
    "sequence_length": 60,
    "last_date": "2025-12-26T00:00:00",
    "period_used": "180d",
    "interval_used": "1d",
    "start_used": null,
    "end_used": null
  }
  ```

- `GET /data/ticker?symbol=PETR4.SA&period=180d&interval=1d&n=60`
  - Retorna os últimos `n` fechamentos para popular o JSON do `/predict`.
  - Resposta:
  ```json
  {
    "symbol": "PETR4.SA",
    "count": 60,
    "closes": [
      { "date": "2025-10-10T00:00:00", "Close": 30.10 }
      /* ... */
    ]
  }
  ```

- `POST /train`
  - Treina sob demanda um modelo para o ticker informado e salva em `model/{SYMBOL}/`.
  - Corpo (opcionalmente aceite `start` e `end` em formato `YYYY-MM-DD`):
  ```json
  {
    "symbol": "PETR4.SA",
    "period": "1y",
    "interval": "1d",
    "sequence_length": 60,
    "epochs": 10,
    "batch_size": 32,
    "start": "2024-01-01",
    "end": "2024-12-26"
  }
  ```
  - Após treinar, as previsões para este símbolo usam automaticamente os artefatos em cache.

- `GET /metrics`
  - Exposição Prometheus (contagem/latência por endpoint, CPU, memória).

- `GET /ui`
  - Interface web simples para consultar `/predict/ticker`, preencher o JSON com os últimos 60 fechamentos via `/data/ticker` e enviar o `POST /predict`.

### Exemplos (cURL)
- Health:
```bash
curl http://localhost:5000/
```
- Previsão com JSON:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[{"Close":100.00},{"Close":100.01}]}'
```
- Previsão por ticker:
```bash
curl "http://localhost:5000/predict/ticker?symbol=PETR4.SA&period=180d&interval=1d"
```

### UI (Front-end simples)
- Acesse `http://localhost:5000/ui`.
- Preencha o ticker e clique em:
  - “Consultar /predict/ticker” para uma previsão direta via Yahoo.
  - “Preencher com últimos 60 fechamentos” para popular o JSON (inclui `symbol` e `Date`) e depois “Enviar /predict”.
- O retorno aparece na área “pre” abaixo de cada cartão. Para `/predict/ticker`, a UI também formata duas linhas: “Previsão” (T+1) e “Próximo alvo” (T+2).

## Monitoramento

O monitoramento da aplicação pode ser realizado utilizando **Prometheus** e **Grafana**. Métricas expostas em `/metrics`.

Exemplo de scrape Prometheus:
```yaml
scrape_configs:
  - job_name: lstm-api
    static_configs:
      - targets: ['host.docker.internal:5000']
    metrics_path: /metrics
    scrape_interval: 15s
```

Métricas principais:
- `api_requests_total{endpoint,method,status}`
- `api_request_latency_seconds{endpoint,method}`
- `api_cpu_percent`, `api_mem_mb`

## Deploy (Render/Railway com Docker)

1. Gere a imagem localmente ou configure um deploy a partir do repositório:
   - Build local:
   ```bash
   docker build -t lstm-api:latest .
   ```
2. Suba com Docker Compose (local ou no provedor):
   ```bash
   docker-compose up --build -d
   ```
3. Configure variáveis de ambiente no provedor (Render/Railway):
   - `MODEL_PATH=/app/model/lstm_model.h5`
   - `SCALER_PATH=/app/model/scaler.pkl`
   - `SEQUENCE_LENGTH=60`
   - `YF_USE_CURL_CFFI=0`
   - `YF_NO_IMPERSONATE=1`
   - `TZ=America/Sao_Paulo`
4. Monte um volume/pasta persistente para `/app/model` para manter modelos treinados.
5. Exponha a porta 5000 e aponte o health check para `/health`.

### Publicando imagem em um registry (opcional)
```bash
docker build -t <dockerhub-usuario>/lstm-api:latest .
docker login
docker push <dockerhub-usuario>/lstm-api:latest
```
No Render/Railway, você pode apontar o deploy para a imagem do Docker Hub ou para o próprio repositório Git (eles constroem automaticamente via Dockerfile).

## Entregáveis
- Código-fonte completo neste repositório, incluindo `src/`, `Dockerfile`, `docker-compose.yml`, `requirements.txt` e notebooks.
- Documentação do projeto neste arquivo (`Descrição.md`) e um guia rápido em `README.md`.
- Scripts/contêineres Docker prontos para deploy (Dockerfile e docker-compose).
- Link da API em produção (quando disponível) ou instruções acima para publicar via Docker Hub + Render/Railway.
- Coleção do Postman em `postman/lstm-api.postman_collection.json` com exemplos prontos.

## Checklist para submissão
- [ ] Build local funcionando (`docker-compose up --build` sobe em http://localhost:5000).
- [ ] UI acessível em `/ui` e `/metrics` exposto.
- [ ] Previsão por ticker retorna `prediction`, `prediction_date`, `future_target`, `future_date`.
- [ ] Previsão por JSON aceita 60+ fechamentos e retorna os mesmos campos.
- [ ] `POST /train` treina e salva artefatos em `model/{SYMBOL}/`.
- [ ] Documentação atualizada e coleção do Postman incluída.

## Vídeo de demonstração (sugestão de roteiro)
1. Subir a API com Docker Compose.
2. Acessar `/ui`, treinar um ticker (ex.: `AAPL`).
3. Consultar previsão por ticker e mostrar “Previsão” e “Próximo alvo”.
4. Preencher o JSON com os últimos 60 fechamentos via `/data/ticker` e enviar `POST /predict`.
5. Mostrar `/metrics`.
6. Se publicado, abrir o link público da API em produção.

## Melhorias Futuras

- Implementar um mecanismo de _retraining_ automático do modelo.
- Adicionar mais _features_ para o treinamento do modelo.
- Otimizar os hiperparâmetros do modelo utilizando técnicas como _Grid Search_ ou _Bayesian Optimization_.

## Arquitetura da Solução

A arquitetura da solução proposta é composta pelos seguintes módulos:

1.  **Módulo de Coleta e Pré-processamento de Dados:** Responsável por obter dados históricos de ações (via `yfinance`) e prepará-los para o treinamento do modelo LSTM. Isso inclui normalização, criação de sequências temporais e divisão em conjuntos de treino e teste.

2.  **Módulo de Treinamento e Avaliação do Modelo LSTM:** Contém a implementação do modelo LSTM, o processo de treinamento com ajuste de hiperparâmetros e a avaliação do desempenho utilizando métricas como MAE, RMSE e MAPE.

3.  **Módulo de Persistência do Modelo:** Encarregado de salvar o modelo treinado em um formato adequado para inferência (ex: HDF5 ou `pickle`).

4.  **Módulo de API RESTful:** Uma API desenvolvida com Flask ou FastAPI que expõe um _endpoint_ para receber dados históricos de preços e retornar previsões futuras utilizando o modelo LSTM carregado.

5.  **Módulo de _Containerization_ (Docker):** Scripts e configurações Docker para empacotar a aplicação da API e suas dependências, garantindo portabilidade e reprodutibilidade em diferentes ambientes.

6.  **Módulo de Monitoramento (Opcional):** Integração com ferramentas como Prometheus e Grafana para monitorar a performance da API em produção, coletando métricas como tempo de resposta, taxa de erros e utilização de recursos.

```mermaid
graph TD
    A[Coleta de Dados (yfinance)] --> B{Pré-processamento de Dados}
    B --> C[Treinamento do Modelo LSTM]
    C --> D[Avaliação do Modelo]
    C --> E[Salvamento do Modelo]
    E --> F[API RESTful (Flask/FastAPI)]
    F --> G[Containerização (Docker)]
    G --> H[Deploy em Nuvem]
    H --> I[Monitoramento (Prometheus/Grafana)]
    Cliente --> F
```

## Tecnologias Utilizadas

-   **Linguagem de Programação:** Python
-   **Coleta de Dados:** `yfinance`
-   **Processamento de Dados:** `pandas`, `numpy`, `scikit-learn`
-   **Modelagem:** `TensorFlow` / `Keras`
-   **API:** `Flask` ou `FastAPI`
-   **Containerização:** `Docker`, `Docker Compose`
-   **Monitoramento:** `Prometheus`, `Grafana` (sugestão)

## Estrutura do Projeto

```
.github/
├── workflows/
│   └── main.yml  # CI/CD pipeline (opcional)
src/
├── data/
│   └── __init__.py
│   └── data_collector.py  # Coleta de dados
│   └── preprocessor.py    # Pré-processamento
├── models/
│   └── __init__.py
│   └── lstm_model.py      # Definição e treinamento do modelo
│   └── predictor.py       # Carregamento e inferência
├── api/
│   └── __init__.py
│   └── main.py            # Implementação da API (Flask/FastAPI)
├── utils/
│   └── __init__.py
│   └── metrics.py         # Funções de métricas
│   └── helpers.py         # Funções auxiliares
notebooks/
├── treinamento_modelo.ipynb  # Notebook para experimentação e treinamento
├── analise_exploratoria.ipynb  # Notebook para EDA (opcional)
Dockerfile
docker-compose.yml
requirements.txt
README.md
.gitignore
```

