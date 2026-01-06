## Previsão de Preços com LSTM (API Flask)

Projeto completo para prever o preço de fechamento de ações usando LSTM, com API Flask, Docker e UI simples.

### Como rodar
```bash
docker-compose up --build -d
```
API local: `http://localhost:5000`

### Endpoints
- `GET /` – health
- `POST /predict` – previsão a partir de 60+ fechamentos (JSON)
- `GET /predict/ticker` – previsão buscando dados no Yahoo
- `GET /data/ticker` – últimos fechamentos para montar o JSON
- `POST /train` – treina e salva artefatos em `model/{SYMBOL}/`
- `GET /metrics` – métricas Prometheus
- `GET /ui` – interface web simples

Respostas de previsão incluem:
- `prediction` (T+1) e `prediction_date`
- `future_target` (T+2) e `future_date`

### Documentação completa
Consulte `Descrição.md` para:
- Guia detalhado de uso
- Exemplos cURL
- Deploy em Render/Railway e publicação de imagem Docker
- Checklist de entregáveis e roteiro do vídeo

### Postman
Coleção disponível em `postman/lstm-api.postman_collection.json`.


