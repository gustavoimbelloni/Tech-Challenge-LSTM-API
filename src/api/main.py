from flask import Flask, request, jsonify, Response
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
# Permite yfinance gerenciar a sessão (curl_cffi) sem impersonation
os.environ.setdefault("YF_NO_IMPERSONATE", "1")
import yfinance as yf
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import requests
import io
from datetime import datetime, timedelta
import json as _json
from src.models.trainer import train_and_save_for_symbol

app = Flask(__name__)

# Caminhos para o modelo e scaler (assumindo que estarão na raiz do projeto ou em um diretório específico)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/lstm_model.h5")
SCALER_PATH = os.environ.get("SCALER_PATH", "model/scaler.pkl")
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 60))

model = None
scaler = None
models_cache = {}  # symbol -> {"model": model, "scaler": scaler, "sequence_length": int}

def normalize_symbol(symbol_raw: str | None) -> str | None:
    """
    Normaliza tickers de entrada:
    - remove prefixo '$' se houver
    - converte para maiúsculas
    - adiciona '.SA' para tickers B3 se não tiver sufixo e terminar com 3/4/5/6/11
    """
    if not symbol_raw:
        return symbol_raw
    s = symbol_raw.strip().upper()
    if s.startswith("$"):
        s = s[1:]
    # se já contém sufixo (.) não altera
    if "." not in s:
        ends = ("3", "4", "5", "6", "11")
        if any(s.endswith(e) for e in ends) and len(s) >= 5:
            s = s + ".SA"
    return s

def _next_business_day(d: datetime) -> datetime:
    """Retorna o próximo dia útil (pula sábado/domingo)."""
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:
        nd = nd + timedelta(days=1)
    return nd

def load_artifacts():
    global model, scaler
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modelo e scaler carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo ou scaler: {e}")
        model = None
        scaler = None

# Carregar artefatos na inicialização da aplicação
with app.app_context():
    load_artifacts()

# -------------------- Métricas Prometheus --------------------
REQUEST_COUNT = Counter("api_requests_total", "Total de requisições", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latência por endpoint", ["endpoint", "method"])
CPU_USAGE = Gauge("api_cpu_percent", "CPU percent do processo")
MEM_USAGE_MB = Gauge("api_mem_mb", "Memória (MB) do processo")

def observe_request(endpoint_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                resp = func(*args, **kwargs)
                status_code = resp[1] if isinstance(resp, tuple) and len(resp) > 1 else 200
                return resp
            finally:
                duration = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=endpoint_name, method=request.method).observe(duration)
                # atualiza métricas de recursos
                proc = psutil.Process()
                CPU_USAGE.set(proc.cpu_percent(interval=None))
                MEM_USAGE_MB.set(proc.memory_info().rss / (1024 * 1024))
                # incrementa contador
                REQUEST_COUNT.labels(endpoint=endpoint_name, method=request.method, status=str(status_code)).inc()
        return wrapper
    return decorator

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
# -------------------------------------------------------------

@app.route("/ui", methods=["GET"])
def ui():
    html = """
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Previsão LSTM - UI</title>
  <style>
    :root {
      --bg: #0b1020;
      --card-bg: #121832;
      --muted: #96a0bd;
      --text: #e8ecf7;
      --primary: #4f8cff;
      --primary-2: #2b69f7;
      --border: #243055;
      --success: #1db954;
      --warn: #f7b32b;
      --danger: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; color: var(--text); background: radial-gradient(1200px 800px at 20% -10%, #1a2252 0%, transparent 60%), radial-gradient(900px 700px at 100% 0%, #13204a 0%, transparent 60%), var(--bg); }
    .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
    header { padding: 24px 0 8px 0; }
    h1 { margin: 0 0 8px 0; font-weight: 700; letter-spacing: 0.3px; }
    p.lead { margin: 0 0 20px 0; color: var(--muted); }
    .hint a { color: var(--primary); text-decoration: none; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 900px) {
      .grid-2 { grid-template-columns: 1fr 1fr; }
    }
    .card { border: 1px solid var(--border); border-radius: 12px; padding: 18px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.0)); box-shadow: 0 0 0 1px rgba(0,0,0,0.15), 0 8px 24px rgba(0,0,0,0.35); }
    .card h2 { margin: 0 0 6px 0; font-size: 20px; }
    .card small { display: block; color: var(--muted); margin-bottom: 12px; }
    label { display: block; font-weight: 600; margin-top: 8px; }
    input, select, textarea { width: 100%; padding: 10px; margin-top: 6px; color: var(--text); background: #0f1430; border: 1px solid var(--border); border-radius: 8px; }
    textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    button { margin-top: 12px; padding: 10px 14px; border: 0; border-radius: 8px; background: linear-gradient(180deg, var(--primary), var(--primary-2)); color: #fff; cursor: pointer; font-weight: 600; box-shadow: 0 4px 14px rgba(79,140,255,0.3); }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
    @media (max-width: 700px) { .row, .row-3 { grid-template-columns: 1fr; } }
    pre { background: #0f1430; padding: 12px; border-radius: 8px; overflow: auto; border: 1px solid var(--border); }
    .muted { color: var(--muted); }
    .subtle { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .spacer { height: 8px; }
    .tag { display: inline-block; padding: 4px 8px; font-size: 12px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); margin-left: 8px; }
    .section-title { display: flex; align-items: center; gap: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Previsão de Preços (LSTM) <span class="tag">Demo</span></h1>
      <p class="lead">Treine um modelo para um ticker, consulte a previsão pelo ticker e, se preferir, envie seu próprio JSON.</p>
      <p class="hint muted">Métricas: <a href="/metrics" target="_blank">/metrics</a> • Dashboard: <a href="/monitor" target="_blank">/monitor</a></p>
    </header>

    <div class="grid">
      <div class="card">
        <div class="section-title">
          <h2>Treinar Modelo (POST /train)</h2>
        </div>
        <small>Treine e salve artefatos em <code>model/{SYMBOL}/</code>. Após o treino, o ticker é propagado para a consulta.</small>
        <div class="row-3">
          <div>
            <label for="symbolTrain">Ticker</label>
            <input id="symbolTrain" value="AAPL" />
          </div>
          <div>
            <label for="periodTrain">Período</label>
            <select id="periodTrain">
              <option value="60d">60d</option>
              <option value="180d">180d</option>
              <option value="1y" selected>1y</option>
            </select>
          </div>
          <div>
            <label for="intervalTrain">Intervalo</label>
            <select id="intervalTrain">
              <option value="1d" selected>1d</option>
            </select>
          </div>
        </div>
        <div class="row-3">
          <div>
            <label for="seqLen">sequence_length</label>
            <input id="seqLen" type="number" value="60" />
          </div>
          <div>
            <label for="epochs">epochs</label>
            <input id="epochs" type="number" value="10" />
          </div>
          <div>
            <label for="batchSize">batch_size</label>
            <input id="batchSize" type="number" value="32" />
          </div>
        </div>
        <div class="row">
          <div>
            <label for="startTrain">Data inicial (opcional)</label>
            <input id="startTrain" type="date" />
          </div>
          <div>
            <label for="endTrain">Data final (opcional)</label>
            <input id="endTrain" type="date" />
          </div>
        </div>
        <button id="btnTrain">Treinar /train</button>
        <pre id="outTrain"></pre>
      </div>

      <div class="card">
        <div class="section-title">
          <h2>Previsão por Ticker (GET /predict/ticker)</h2>
        </div>
        <small>Busca dados no Yahoo Finance com múltiplos fallbacks.</small>
        <div class="row">
          <div>
            <label for="symbol">Ticker (ex.: PETR4.SA, AAPL)</label>
            <input id="symbol" value="PETR4.SA" />
          </div>
          <div>
            <label for="interval">Intervalo</label>
            <select id="interval">
              <option value="1d" selected>1d</option>
            </select>
          </div>
          <div>
            <label for="period">Período</label>
            <select id="period">
              <option value="60d">60d</option>
              <option value="180d" selected>180d</option>
              <option value="1y">1y</option>
            </select>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Data inicial (opcional)</label>
            <input id="start" type="date" />
          </div>
          <div>
            <label>Data final (opcional)</label>
            <input id="end" type="date" />
          </div>
        </div>
        <button id="btnTicker">Consultar /predict/ticker</button>
        <pre id="outTicker"></pre>
      </div>

      <div class="card">
        <div class="section-title">
          <h2>Previsão por JSON (POST /predict)</h2>
        </div>
        <small>Envie ao menos 60 valores de <code>Close</code>. Dica: preencha automaticamente a partir do ticker.</small>
        <textarea id="jsonBody" rows="10">{ "data": [ {"Close": 100.00}, {"Close": 100.01} ] }</textarea>
        <div class="row">
          <div>
            <button id="btnFillJson">Preencher com últimos 60 fechamentos (via ticker)</button>
          </div>
          <div>
            <button id="btnJson">Enviar /predict</button>
          </div>
        </div>
        <pre id="outJson"></pre>
      </div>
    </div>
  </div>

  <script>
    const btnTicker = document.getElementById('btnTicker');
    const outTicker = document.getElementById('outTicker');
    const btnJson = document.getElementById('btnJson');
    const outJson = document.getElementById('outJson');

    const btnFillJson = document.getElementById('btnFillJson');
    const btnTrain = document.getElementById('btnTrain');
    const outTrain = document.getElementById('outTrain');

    btnTicker.onclick = async () => {
      btnTicker.disabled = true;
      outTicker.textContent = 'Consultando...';
      try {
        const symbol = document.getElementById('symbol').value.trim();
        const period = document.getElementById('period').value;
        const interval = document.getElementById('interval').value;
        const start = document.getElementById('start').value;
        const end = document.getElementById('end').value;

        const params = new URLSearchParams({ symbol, period, interval });
        if (start) params.set('start', start);
        if (end) params.set('end', end);

        const res = await fetch('/predict/ticker?' + params.toString(), { method: 'GET' });
        const data = await res.json();
        if (!res.ok) {
          outTicker.textContent = (data && data.error) ? data.error : JSON.stringify(data, null, 2);
        } else if (typeof data.prediction === 'number' && typeof data.future_target === 'number') {
          const pd = data.prediction_date || '(próximo dia útil)';
          const fd = data.future_date || '(dia útil seguinte)';
          outTicker.textContent =
            `Previsão: ${pd} → ${Number(data.prediction).toFixed(2)}\n` +
            `Próximo alvo: ${fd} → ${Number(data.future_target).toFixed(2)}`;
        } else {
          outTicker.textContent = JSON.stringify(data, null, 2);
        }
      } catch (e) {
        outTicker.textContent = String(e);
      } finally {
        btnTicker.disabled = false;
      }
    };

    btnJson.onclick = async () => {
      btnJson.disabled = true;
      outJson.textContent = 'Enviando...';
      try {
        const body = document.getElementById('jsonBody').value;
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body
        });
        const txt = await res.text();
        try {
          outJson.textContent = JSON.stringify(JSON.parse(txt), null, 2);
        } catch {
          outJson.textContent = txt;
        }
      } catch (e) {
        outJson.textContent = String(e);
      } finally {
        btnJson.disabled = false;
      }
    };

    btnFillJson.onclick = async () => {
      btnFillJson.disabled = true;
      outJson.textContent = 'Buscando últimos 60 fechamentos...';
      try {
        const symbol = document.getElementById('symbol').value.trim();
        const period = document.getElementById('period').value;
        const interval = document.getElementById('interval').value;
        const params = new URLSearchParams({ symbol, period, interval, n: "60" });
        const res = await fetch('/data/ticker?' + params.toString(), { method: 'GET' });
        const data = await res.json();
        if (!res.ok) {
          outJson.textContent = JSON.stringify(data, null, 2);
          return;
        }
        const closes = (data.closes || []).map(c => ({ Date: c.date, Close: c.Close }));
        const payload = { symbol, data: closes };
        document.getElementById('jsonBody').value = JSON.stringify(payload, null, 2);
        outJson.textContent = 'JSON preenchido com ' + closes.length + ' valores de Close.';
      } catch (e) {
        outJson.textContent = String(e);
      } finally {
        btnFillJson.disabled = false;
      }
    };

    btnTrain.onclick = async () => {
      btnTrain.disabled = true;
      outTrain.textContent = 'Treinando seu modelo, por favor aguarde...';
      try {
        const symbol = document.getElementById('symbolTrain').value.trim();
        const period = document.getElementById('periodTrain').value;
        const interval = document.getElementById('intervalTrain').value;
        const sequence_length = parseInt(document.getElementById('seqLen').value || '60', 10);
        const epochs = parseInt(document.getElementById('epochs').value || '10', 10);
        const batch_size = parseInt(document.getElementById('batchSize').value || '32', 10);
        const start = document.getElementById('startTrain').value;
        const end = document.getElementById('endTrain').value;

        const body = { symbol, period, interval, sequence_length, epochs, batch_size };
        if (start) body.start = start;
        if (end) body.end = end;

        const res = await fetch('/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) {
          outTrain.textContent = (data && data.error) ? ('Erro no treino: ' + data.error) : 'Erro ao treinar.';
        } else if (data && data.trained) {
          outTrain.textContent = `Treinamento concluído para ${symbol}. Faça sua previsão abaixo.`;
          const symbolInput = document.getElementById('symbol');
          if (symbolInput) symbolInput.value = symbol;
        } else {
          outTrain.textContent = 'Não foi possível confirmar o treino.';
        }
      } catch (e) {
        outTrain.textContent = String(e);
      } finally {
        btnTrain.disabled = false;
      }
    };
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")

@app.route("/monitor", methods=["GET"])
def monitor():
    html = """
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Monitoramento - LSTM API</title>
  <style>
    :root {
      --bg: #0b1020;
      --card-bg: #121832;
      --muted: #96a0bd;
      --text: #e8ecf7;
      --primary: #4f8cff;
      --border: #243055;
      --good: #1db954;
      --warn: #f7b32b;
      --bad: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: var(--text); background: radial-gradient(1200px 800px at 20% -10%, #1a2252 0%, transparent 60%), radial-gradient(900px 700px at 100% 0%, #13204a 0%, transparent 60%), var(--bg); }
    .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
    header { padding: 24px 0 8px 0; }
    h1 { margin: 0 0 6px 0; font-weight: 700; }
    .muted { color: var(--muted); }
    a { color: var(--primary); text-decoration: none; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 900px) { .grid-3 { grid-template-columns: 1fr 1fr 1fr; } .grid-2 { grid-template-columns: 1fr 1fr; } }
    .card { border: 1px solid var(--border); border-radius: 12px; padding: 16px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.0)); box-shadow: 0 0 0 1px rgba(0,0,0,0.15), 0 8px 24px rgba(0,0,0,0.35); }
    .card h2 { margin: 0 0 10px 0; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border-bottom: 1px solid var(--border); text-align: left; }
    .kpi { font-size: 28px; font-weight: 700; }
    .spark { width: 100%; height: 40px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }
    .ok { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .small { font-size: 12px; }
    pre { background: #0f1430; padding: 10px; border: 1px solid var(--border); border-radius: 8px; overflow: auto; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Monitoramento da API</h1>
      <p class="muted">Atualiza a cada 5s a partir de <a href="/metrics" target="_blank">/metrics</a>. <a href="/ui" target="_blank">Voltar ao UI</a></p>
    </header>

    <div class="grid grid-3">
      <div class="card">
        <h2>CPU (%)</h2>
        <div class="kpi" id="cpuKpi">--</div>
        <svg class="spark" id="cpuSpark"></svg>
        <div class="small muted">api_cpu_percent</div>
      </div>
      <div class="card">
        <h2>Memória (MB)</h2>
        <div class="kpi" id="memKpi">--</div>
        <svg class="spark" id="memSpark"></svg>
        <div class="small muted">api_mem_mb</div>
      </div>
      <div class="card">
        <h2>Requisições</h2>
        <div class="kpi" id="reqKpi">--</div>
        <div class="small muted">api_requests_total</div>
      </div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h2>Requisições por endpoint</h2>
        <table>
          <thead><tr><th>Endpoint</th><th>Método</th><th>Status</th><th>Total</th></tr></thead>
          <tbody id="reqTable"></tbody>
        </table>
      </div>
      <div class="card">
        <h2>Latência média por endpoint</h2>
        <table>
          <thead><tr><th>Endpoint</th><th>Método</th><th>Média (s)</th></tr></thead>
          <tbody id="latTable"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Bruto (/metrics)</h2>
      <pre id="rawMetrics" class="small"></pre>
    </div>
  </div>

  <script>
    function parseMetrics(text) {
      const lines = text.split(/\\r?\\n/);
      const out = {};
      const re = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\\{([^}]*)\\})?\\s+([0-9.eE+-]+)$/;
      for (const line of lines) {
        if (!line || line[0] === '#') continue;
        const m = line.match(re);
        if (!m) continue;
        const name = m[1];
        const labelsStr = m[3] || '';
        const value = Number(m[4]);
        const labels = {};
        if (labelsStr) {
          for (const pair of labelsStr.split(',')) {
            const [k, v] = pair.split('=');
            if (!k) continue;
            labels[k] = (v || '').replace(/^\"|\"$/g, '');
          }
        }
        if (!out[name]) out[name] = [];
        out[name].push({ labels, value });
      }
      return out;
    }

    function renderSpark(svg, series, color = '#4f8cff') {
      const w = svg.clientWidth || svg.parentElement.clientWidth;
      const h = svg.clientHeight || 40;
      svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      svg.innerHTML = '';
      if (!series.length) return;
      const min = Math.min(...series);
      const max = Math.max(...series);
      const norm = (v) => (max === min ? h/2 : h - ((v - min) / (max - min)) * (h - 4) - 2);
      const step = series.length > 1 ? (w - 4) / (series.length - 1) : w;
      let d = '';
      for (let i = 0; i < series.length; i++) {
        const x = 2 + i * step;
        const y = norm(series[i]);
        d += (i === 0 ? 'M' : ' L') + x + ' ' + y;
      }
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2');
      svg.appendChild(path);
    }

    const cpuSeries = [];
    const memSeries = [];
    let totalReq = 0;

    async function refresh() {
      try {
        const res = await fetch('/metrics', { cache: 'no-store' });
        const text = await res.text();
        document.getElementById('rawMetrics').textContent = text;
        const m = parseMetrics(text);

        const cpu = (m['api_cpu_percent'] && m['api_cpu_percent'][0]?.value) ?? null;
        const mem = (m['api_mem_mb'] && m['api_mem_mb'][0]?.value) ?? null;
        if (cpu !== null) {
          cpuSeries.push(cpu);
          if (cpuSeries.length > 50) cpuSeries.shift();
          document.getElementById('cpuKpi').textContent = cpu.toFixed(1) + '%';
          renderSpark(document.getElementById('cpuSpark'), cpuSeries);
        }
        if (mem !== null) {
          memSeries.push(mem);
          if (memSeries.length > 50) memSeries.shift();
          document.getElementById('memKpi').textContent = mem.toFixed(1);
          renderSpark(document.getElementById('memSpark'), memSeries, '#1db954');
        }

        const req = m['api_requests_total'] || [];
        totalReq = req.reduce((acc, r) => acc + (r.value || 0), 0);
        document.getElementById('reqKpi').textContent = totalReq.toFixed(0);
        const rows = [];
        for (const r of req) {
          const ep = r.labels.endpoint || '-';
          const method = r.labels.method || '-';
          const status = r.labels.status || '-';
          rows.push({ ep, method, status, total: r.value || 0 });
        }
        rows.sort((a, b) => b.total - a.total);
        const reqHtml = rows.slice(0, 12).map(r => (
          '<tr><td><code>' + r.ep + '</code></td><td>' + r.method + '</td><td>' + r.status + '</td><td>' + r.total + '</td></tr>'
        )).join('');
        document.getElementById('reqTable').innerHTML = reqHtml || '<tr><td colspan="4" class="muted">Sem dados</td></tr>';

        const sums = m['api_request_latency_seconds_sum'] || [];
        const counts = m['api_request_latency_seconds_count'] || [];
        const mapCount = new Map();
        for (const c of counts) {
          const key = (c.labels.endpoint || '-') + '|' + (c.labels.method || '-');
          mapCount.set(key, c.value || 0);
        }
        const latRows = [];
        for (const s of sums) {
          const key = (s.labels.endpoint || '-') + '|' + (s.labels.method || '-');
          const cnt = mapCount.get(key) || 0;
          const avg = cnt > 0 ? (s.value || 0) / cnt : 0;
          latRows.push({ ep: s.labels.endpoint || '-', method: s.labels.method || '-', avg });
        }
        latRows.sort((a, b) => b.avg - a.avg);
        const latHtml = latRows.slice(0, 12).map(r => (
          '<tr><td><code>' + r.ep + '</code></td><td>' + r.method + '</td><td>' + r.avg.toFixed(3) + '</td></tr>'
        )).join('');
        document.getElementById('latTable').innerHTML = latHtml || '<tr><td colspan="3" class="muted">Sem dados</td></tr>';
      } catch (e) {
        document.getElementById('rawMetrics').textContent = String(e);
      }
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
@observe_request("health")
def health():
    return jsonify({
        "status": "ok",
        "message": "API em execução",
        "endpoints": [
            "POST /predict",
            "GET /predict/ticker",
            "GET /data/ticker",
            "POST /train",
            "GET /metrics",
            "GET /ui"
        ]
    })

def load_artifacts_for_symbol(symbol: str):
    """Carrega (e cacheia) artefatos para um ticker específico, se existir em model/{symbol}/."""
    global models_cache
    if symbol in models_cache:
        return models_cache[symbol]
    symbol_dir = os.path.join("model", symbol)
    model_path = os.path.join(symbol_dir, "lstm_model.h5")
    scaler_path = os.path.join(symbol_dir, "scaler.pkl")
    meta_path = os.path.join(symbol_dir, "meta.json")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        m = load_model(model_path)
        s = joblib.load(scaler_path)
        seq_len = SEQUENCE_LENGTH
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                    seq_len = int(meta.get("sequence_length", SEQUENCE_LENGTH))
            except Exception:
                pass
        models_cache[symbol] = {"model": m, "scaler": s, "sequence_length": seq_len}
        return models_cache[symbol]
    return None

def get_active_artifacts(optional_symbol: str | None):
    """Retorna (model, scaler, sequence_length) usando cache por symbol, ou o default."""
    if optional_symbol:
        entry = load_artifacts_for_symbol(optional_symbol)
        if entry:
            return entry["model"], entry["scaler"], entry["sequence_length"]
    # default global
    return model, scaler, SEQUENCE_LENGTH

@app.route("/predict", methods=["POST"])
@observe_request("predict_post")
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Modelo ou scaler não carregados. Tente novamente mais tarde."}), 500

    try:
        # Tenta obter JSON independentemente do Content-Type
        data = request.get_json(silent=True)
        if data is None:
            # Fallback: tentar interpretar o corpo bruto como JSON
            raw_body = request.data.decode("utf-8") if request.data else ""
            if raw_body:
                try:
                    data = json.loads(raw_body)
                except Exception:
                    data = None

        # Aceita tanto {"data":[...]} quanto a lista diretamente [...]
        historical_data_raw = None
        symbol_param = None
        if isinstance(data, dict):
            historical_data_raw = data.get("data")
            symbol_param = normalize_symbol(data.get("symbol"))
        elif isinstance(data, list):
            historical_data_raw = data
        else:
            historical_data_raw = None

        if not historical_data_raw:
            return jsonify({
                "error": "Dados históricos não fornecidos ou corpo inválido.",
                "hint": "Envie JSON com Content-Type: application/json.",
                "expected_examples": [
                    {"data": [{"Close": 123.45}, {"Close": 123.56}, "..."]},
                    [{"Close": 123.45}, {"Close": 123.56}, "..."]
                ]
            }), 415

        # Converter dados históricos para DataFrame
        historical_df = pd.DataFrame(historical_data_raw)
        # Se vier data, ordenar por data para garantir sequência correta
        if "Date" in historical_df.columns:
            try:
                historical_df["Date"] = pd.to_datetime(historical_df["Date"], errors="coerce")
                historical_df = historical_df.dropna(subset=["Date"])
                historical_df = historical_df.sort_values("Date")
                historical_df = historical_df.drop_duplicates(subset=["Date"], keep="last")
            except Exception:
                pass
        # Normaliza Close
        historical_df["Close"] = historical_df["Close"].astype(float)
        historical_df = historical_df.dropna(subset=["Close"])

        active_model, active_scaler, active_seq_len = get_active_artifacts(symbol_param)
        if active_model is None or active_scaler is None:
            return jsonify({"error": "Artefatos não encontrados para o símbolo solicitado."}), 500

        if len(historical_df) < active_seq_len:
            return jsonify({"error": f"Dados históricos devem ter pelo menos {SEQUENCE_LENGTH} entradas."}), 400

        # Pega os últimos 'sequence_length' valores da coluna 'Close'
        last_sequence_series = historical_df["Close"].tail(active_seq_len)
        last_sequence = last_sequence_series.values.reshape(-1, 1)
        # Data da última observação (se houver coluna Date)
        used_last_date = None
        if "Date" in historical_df.columns and len(historical_df) > 0:
            try:
                used_last_date = historical_df["Date"].iloc[-1]
                used_last_date = used_last_date.isoformat() if hasattr(used_last_date, "isoformat") else str(used_last_date)
            except Exception:
                used_last_date = None

        # Escala os dados
        scaled_last_sequence = active_scaler.transform(last_sequence)
        # Reshape para o formato esperado pelo LSTM (1, sequence_length, 1)
        x_input = scaled_last_sequence.reshape(1, active_seq_len, 1)
        # Faz a previsão
        predicted_scaled_price = active_model.predict(x_input)
        # Inverte a escala para obter o preço real
        predicted_price = active_scaler.inverse_transform(predicted_scaled_price)
        predicted_value = float(predicted_price[0][0])
        # Calcula alvo futuro (T+2) com mais um passo iterativo
        try:
            history_values = historical_df["Close"].values.astype(float)
            history_plus = np.append(history_values, predicted_value)
            win2 = history_plus[-active_seq_len:].reshape(-1, 1)
            scaled2 = active_scaler.transform(win2)
            x2 = scaled2.reshape(1, active_seq_len, 1)
            p2_scaled = active_model.predict(x2)
            future_target = float(active_scaler.inverse_transform(p2_scaled)[0][0])
        except Exception:
            future_target = None
        # Datas previstas
        prediction_date = None
        future_date = None
        try:
            if used_last_date:
                d0 = pd.to_datetime(used_last_date)
                d1 = _next_business_day(d0)
                d2 = _next_business_day(d1)
                prediction_date = d1.isoformat()
                future_date = d2.isoformat()
        except Exception:
            prediction_date = None
            future_date = None
        return jsonify({
            "prediction": predicted_value,
            "used_count": int(len(last_sequence_series)),
            "used_last_date": used_last_date,
            "prediction_date": prediction_date,
            "future_target": future_target,
            "future_date": future_date
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET"])
@observe_request("predict_get_info")
def predict_get_info():
    return jsonify({
        "error": "Use POST com JSON. Exemplo de corpo: {\"data\": [{\"Close\": 123.45}, ...]}",
        "requirements": {
            "Content-Type": "application/json",
            "min_length": SEQUENCE_LENGTH
        }
    }), 405

@app.route("/predict/ticker", methods=["GET"])
@observe_request("predict_from_ticker")
def predict_from_ticker():
    if model is None or scaler is None:
        return jsonify({"error": "Modelo ou scaler não carregados. Tente novamente mais tarde."}), 500

    symbol = normalize_symbol(request.args.get("symbol"))
    period = request.args.get("period", "120d")
    interval = request.args.get("interval", "1d")
    start = request.args.get("start")  # opcional: YYYY-MM-DD
    end = request.args.get("end")      # opcional: YYYY-MM-DD
    if not symbol:
        return jsonify({"error": "Parâmetro 'symbol' obrigatório. Ex.: symbol=PETR4.SA"}), 400

    try:
        # 1) Primeiro, tenta Yahoo CHART JSON (query2) – normalmente não exige crumb/impersonation
        df = None
        try:
            if start:
                dt_start = int(datetime.fromisoformat(start).timestamp())
                dt_end = int(datetime.fromisoformat(end).timestamp()) if end else int(datetime.now().timestamp())
                url = (
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
                    f"?period1={dt_start}&period2={dt_end}&interval={interval}"
                )
            else:
                now = datetime.now()
                rng = period
                # Normaliza alguns valores comuns
                if period.endswith(("d","y","mo")):
                    rng = period
                elif period.endswith("m"):  # 'm' pode significar minutos; evita ambiguidade
                    rng = "1y"
                url = (
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
                    f"?range={rng}&interval={interval}"
                )
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200 and resp.headers.get("Content-Type","").startswith("application/json"):
                payload = resp.json()
                result = payload.get("chart", {}).get("result", [])
                if result:
                    r0 = result[0]
                    closes = r0.get("indicators", {}).get("quote", [{}])[0].get("close", [])
                    ts = r0.get("timestamp", [])
                    if closes and ts and len(closes) == len(ts):
                        dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
                        df = pd.DataFrame({"Close": closes}, index=dates)
        except Exception:
            df = None

        # 2) Se CHART JSON falhar/estiver vazio, tenta baixar CSV direto (query1)
        if df is None or df.empty:
            try:
                headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
                if start:
                    dt_start = int(datetime.fromisoformat(start).timestamp())
                    dt_end = int(datetime.fromisoformat(end).timestamp()) if end else int(datetime.now().timestamp())
                else:
                    now = datetime.now()
                    if period.endswith("d"):
                        days = int(period[:-1])
                    elif period.endswith("y"):
                        years = int(period[:-1])
                        days = 365 * years
                    else:
                        days = 365
                    dt_start = int((now - timedelta(days=days)).timestamp())
                    dt_end = int(now.timestamp())
                url = (
                    f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                    f"?period1={dt_start}&period2={dt_end}&interval={interval}"
                    f"&events=history&includeAdjustedClose=true"
                )
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200 and "Date,Open,High,Low,Close,Adj Close,Volume" in resp.text:
                    df = pd.read_csv(io.StringIO(resp.text))
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        df = df.dropna(subset=["Date"])
                        df = df.set_index("Date")
            except Exception:
                df = None

        # 3) Se ainda vazio, tenta UMA vez via yfinance (download)
        if df is None or df.empty:
            try:
                if start:
                    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, threads=False, progress=False)
                else:
                    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, threads=False, progress=False)
                if (df is None or df.empty) and not start:
                    df = yf.download(symbol, period="1y", interval=interval, auto_adjust=False, threads=False, progress=False)
            except Exception:
                df = None

        # 4) Se ainda vazio, tenta UMA vez via yfinance (history)
        if df is None or df.empty:
            try:
                t = yf.Ticker(symbol)
                if start:
                    df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
                else:
                    df = t.history(period=period, interval=interval, auto_adjust=False)
                if (df is None or df.empty) and not start:
                    df = t.history(period="1y", interval=interval, auto_adjust=False)
            except Exception:
                df = None

        if df is None or df.empty:
            return jsonify({
                "error": f"Nenhum dado retornado para {symbol}.",
                "used": {"period": period, "interval": interval, "start": start, "end": end},
                "hints": [
                    "Aumente 'period' (ex.: 1y) ou informe 'start' e 'end'.",
                    "Use 'interval=1d' para diários.",
                    "Verifique o símbolo (ex.: PETR4.SA) e conectividade de rede."
                ]
            }), 400

        # Tenta obter a série de fechamento de forma robusta
        closes_series = None
        if "Close" in df.columns:
            closes_series = df["Close"]
        else:
            # MultiIndex handling
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if "Close" in df.columns.get_level_values(0):
                        closes_series = df["Close"].iloc[:, 0]
                    elif "Close" in df.columns.get_level_values(1):
                        temp = df.xs("Close", axis=1, level=1)
                        closes_series = temp.iloc[:, 0] if isinstance(temp, pd.DataFrame) else temp
            except Exception:
                closes_series = None

        if closes_series is None or len(closes_series) < SEQUENCE_LENGTH:
            return jsonify({
                "error": f"Dados insuficientes: necessário >= {SEQUENCE_LENGTH} fechamentos.",
                "found": 0 if closes_series is None else int(len(closes_series)),
                "hint": "Aumente 'period' (ex.: 180d, 1y) ou ajuste 'start'/'end'."
            }), 400

        # Escolhe artefatos por ticker, se existirem
        active_model, active_scaler, active_seq_len = get_active_artifacts(symbol)
        if active_model is None or active_scaler is None:
            active_model, active_scaler, active_seq_len = model, scaler, SEQUENCE_LENGTH

        # Previsão single-step
        last_sequence = closes_series.tail(active_seq_len).astype(float).values.reshape(-1, 1)
        scaled_last_sequence = active_scaler.transform(last_sequence)
        x_input = scaled_last_sequence.reshape(1, active_seq_len, 1)
        predicted_scaled_price = active_model.predict(x_input)
        predicted_price = active_scaler.inverse_transform(predicted_scaled_price)
        predicted_value = float(predicted_price[0][0])

        last_date = closes_series.index[-1]
        last_date_str = getattr(last_date, "isoformat", lambda: str(last_date))()
        # Calcula alvo futuro (T+2)
        try:
            history_values = closes_series.values.astype(float)
            history_plus = np.append(history_values, predicted_value)
            win2 = history_plus[-active_seq_len:].reshape(-1, 1)
            scaled2 = active_scaler.transform(win2)
            x2 = scaled2.reshape(1, active_seq_len, 1)
            p2_scaled = active_model.predict(x2)
            future_target = float(active_scaler.inverse_transform(p2_scaled)[0][0])
        except Exception:
            future_target = None
        # Datas previstas
        try:
            d1 = _next_business_day(last_date)
            d2 = _next_business_day(d1)
            prediction_date = getattr(d1, "isoformat", lambda: str(d1))()
            future_date = getattr(d2, "isoformat", lambda: str(d2))()
        except Exception:
            prediction_date = None
            future_date = None
        return jsonify({
            "symbol": symbol,
            "prediction": predicted_value,
            "prediction_date": prediction_date,
            "future_target": future_target,
            "future_date": future_date,
            "sequence_length": active_seq_len,
            "last_date": last_date_str,
            "period_used": period,
            "interval_used": interval,
            "start_used": start,
            "end_used": end
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/data/ticker", methods=["GET"])
@observe_request("data_ticker_get")
def data_ticker():
    symbol = normalize_symbol(request.args.get("symbol"))
    period = request.args.get("period", "180d")
    interval = request.args.get("interval", "1d")
    start = request.args.get("start")
    end = request.args.get("end")
    n = int(request.args.get("n", "60"))
    if not symbol:
        return jsonify({"error": "Parâmetro 'symbol' obrigatório."}), 400
    try:
        # Reutiliza a pipeline de busca: chart JSON -> CSV -> yfinance
        df = None
        # chart JSON
        try:
            if start:
                dt_start = int(datetime.fromisoformat(start).timestamp())
                dt_end = int(datetime.fromisoformat(end).timestamp()) if end else int(datetime.now().timestamp())
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={dt_start}&period2={dt_end}&interval={interval}"
            else:
                now = datetime.now()
                rng = period if period.endswith(("d","y","mo")) else "1y"
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval={interval}"
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200 and r.headers.get("Content-Type","").startswith("application/json"):
                payload = r.json()
                result = payload.get("chart", {}).get("result", [])
                if result:
                    r0 = result[0]
                    closes = r0.get("indicators", {}).get("quote", [{}])[0].get("close", [])
                    ts = r0.get("timestamp", [])
                    if closes and ts and len(closes) == len(ts):
                        dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
                        df = pd.DataFrame({"Close": closes}, index=dates)
        except Exception:
            df = None
        # CSV fallback
        if df is None or df.empty:
            try:
                headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
                if start:
                    dt_start = int(datetime.fromisoformat(start).timestamp())
                    dt_end = int(datetime.fromisoformat(end).timestamp()) if end else int(datetime.now().timestamp())
                else:
                    now = datetime.now()
                    if period.endswith("d"):
                        days = int(period[:-1])
                    elif period.endswith("y"):
                        years = int(period[:-1])
                        days = 365 * years
                    else:
                        days = 365
                    dt_start = int((now - timedelta(days=days)).timestamp())
                    dt_end = int(now.timestamp())
                url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={dt_start}&period2={dt_end}&interval={interval}&events=history&includeAdjustedClose=true"
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200 and "Date,Open,High,Low,Close,Adj Close,Volume" in r.text:
                    df = pd.read_csv(io.StringIO(r.text))
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        df = df.dropna(subset=["Date"]).set_index("Date")
            except Exception:
                df = None
        # yfinance fallback (uma vez)
        if df is None or df.empty:
            try:
                if start:
                    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, threads=False, progress=False)
                else:
                    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, threads=False, progress=False)
                if (df is None or df.empty) and not start:
                    df = yf.download(symbol, period="1y", interval=interval, auto_adjust=False, threads=False, progress=False)
            except Exception:
                df = None
        if df is None or df.empty:
            return jsonify({"error": f"Nenhum dado para {symbol}."}), 400
        # Normaliza e seleciona últimos n
        if "Close" not in df.columns and isinstance(df.columns, pd.MultiIndex):
            try:
                if "Close" in df.columns.get_level_values(0):
                    s = df["Close"].iloc[:, 0]
                else:
                    s = df.xs("Close", axis=1, level=1)
                df = pd.DataFrame({"Close": s})
            except Exception:
                pass
        if "Close" not in df.columns:
            return jsonify({"error": "Coluna Close não encontrada no retorno de dados."}), 400
        df = df.dropna(subset=["Close"])
        tail = df.tail(n)
        closes = [{"date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                   "Close": float(row["Close"])} for idx, row in tail.iterrows()]
        return jsonify({"symbol": symbol, "count": len(closes), "closes": closes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
@observe_request("train_post")
def train_endpoint():
    """
    Treina e salva artefatos para um ticker sob demanda.
    body: { "symbol": "PETR4.SA", "period": "1y", "interval": "1d", "sequence_length": 60, "epochs": 10, "batch_size": 32 }
    """
    try:
        body = request.get_json(silent=True) or {}
        symbol = normalize_symbol(body.get("symbol"))
        if not symbol:
            return jsonify({"error": "Campo 'symbol' é obrigatório."}), 400
        period = body.get("period", "1y")
        interval = body.get("interval", "1d")
        sequence_length = int(body.get("sequence_length", SEQUENCE_LENGTH))
        epochs = int(body.get("epochs", 10))
        batch_size = int(body.get("batch_size", 32))
        start = body.get("start")
        end = body.get("end")

        info = train_and_save_for_symbol(
            symbol=symbol,
            period=period,
            interval=interval,
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size,
            start=start,
            end=end,
        )
        # limpa cache para recarregar artefatos atualizados
        if symbol in models_cache:
            del models_cache[symbol]
        entry = load_artifacts_for_symbol(symbol)
        ok = entry is not None
        return jsonify({"trained": ok, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

