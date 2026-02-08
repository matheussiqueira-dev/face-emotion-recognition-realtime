# EmotionAI Realtime

Plataforma de reconhecimento de emoções faciais em tempo real com backend FastAPI, pipeline de visão computacional em Python e dashboard web responsivo para operação contínua.

## Visão Geral

O sistema foi estruturado para cenários de monitoramento em tempo real (laboratórios, protótipos de atendimento, pesquisa comportamental e observabilidade operacional), com foco em:

- baixa latência de streaming;
- clareza visual para operadores;
- configuração dinâmica sem reinício;
- segurança para ambientes de produção;
- arquitetura evolutiva e testável.

## Objetivo de Negócio

- transformar vídeo em dados emocionais acionáveis;
- disponibilizar métricas operacionais e analíticas em tempo real;
- reduzir custo de operação com painel único de monitoramento e controle.

## Arquitetura e Decisões Técnicas

### Camadas

- `app/core`: domínio e processamento (configuração, detecção, tracking, análise, métricas de sessão).
- `app/backend`: interface de entrega (REST + WebSocket), autenticação/autorização e orquestração de stream.
- `app/frontend`: dashboard operacional, visualização ao vivo, analytics e controles de runtime.

### Princípios aplicados

- **SRP (Single Responsibility)**: separação clara entre pipeline de vídeo, segurança, contratos de API e interface.
- **DRY**: centralização de configuração, autenticação e serialização de payloads.
- **Clean boundaries**: domínio (`core`) desacoplado da camada HTTP (`backend`) e da UI (`frontend`).
- **Fail-safe behavior**: validações e tratamento explícito de erros em configuração, API e streaming.

## Stack

- **Backend**: Python, FastAPI, Uvicorn.
- **CV/IA**: OpenCV, DeepFace, NumPy, TensorFlow/Keras.
- **Frontend**: HTML5, CSS moderno (Design System próprio), JavaScript ES6, Chart.js.
- **Qualidade**: Pytest, TestClient (FastAPI), tipagem estática com type hints.

## Principais Melhorias Implementadas

- API versionada (`/api/v1`) com contratos tipados e documentação OpenAPI.
- Segurança com autenticação por token e autorização por permissão (`read` e `admin`).
- Serviço de stream desacoplado com lifecycle controlado (start/stop/restart).
- Métricas de sessão em tempo real (FPS atual/médio, trilhas, distribuição emocional e histórico de sentimento).
- Configuração dinâmica via API (`PATCH /api/v1/config`) sem reiniciar servidor.
- Frontend totalmente redesenhado para:
  - hierarquia visual forte;
  - acessibilidade e estados claros;
  - responsividade desktop/mobile;
  - exportação de métricas e captura de quadro.
- Testes unitários e de API para reduzir regressões.

## Estrutura do Projeto

```text
.
├── app
│   ├── backend
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── security.py
│   │   └── services.py
│   ├── core
│   │   ├── analyzer.py
│   │   ├── config.py
│   │   ├── detector.py
│   │   ├── metrics.py
│   │   ├── processor.py
│   │   └── tracker.py
│   └── frontend
│       ├── app.js
│       ├── index.html
│       └── style.css
├── tests
│   ├── test_api.py
│   ├── test_config.py
│   └── test_metrics.py
├── main.py
├── run_api.py
├── run.bat
└── requirements.txt
```

## Instalação e Execução

### 1. Clonar repositório

```bash
git clone https://github.com/matheussiqueira-dev/face-emotion-recognition-realtime.git
cd face-emotion-recognition-realtime
```

### 2. Criar ambiente e instalar dependências

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Rodar aplicação

```bash
python run_api.py --host 127.0.0.1 --port 8000
```

Ou no Windows:

```bash
run.bat
```

### 4. Acessar dashboard

- Interface: `http://127.0.0.1:8000`
- Docs API: `http://127.0.0.1:8000/api/v1/docs`

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|---|---|---|
| `EMOTION_VIDEO_SOURCE` | Câmera (`0`) ou caminho de vídeo | `0` |
| `EMOTION_WIDTH` | Largura da captura | `1280` |
| `EMOTION_HEIGHT` | Altura da captura | `720` |
| `EMOTION_MAX_FPS` | Limite de FPS do pipeline | `30` |
| `EMOTION_INTERVAL` | Intervalo de inferência por face (s) | `0.5` |
| `EMOTION_MIN_SCORE` | Confiança mínima da emoção (%) | `0` |
| `EMOTION_JPEG_QUALITY` | Qualidade do frame no WebSocket | `80` |
| `EMOTION_API_READ_TOKEN` | Token para leitura (config/metrics/ws) | vazio |
| `EMOTION_API_ADMIN_TOKEN` | Token para operações administrativas | vazio |
| `EMOTION_CORS_ORIGINS` | Lista de origens CORS (csv) | `*` |

> Compatibilidade: `EMOTION_API_TOKEN` também é aceito como token legado.

## API (Resumo)

- `GET /api/v1/health` - healthcheck.
- `GET /api/v1/config` - configuração atual (perm. `read`).
- `PATCH /api/v1/config` - atualização de configuração (perm. `admin`).
- `GET /api/v1/metrics` - métricas da sessão ativa (perm. `read`).
- `WS /api/v1/ws/stream` - stream de frames e detecções (perm. `read`).

## Qualidade e Testes

Executar suíte:

```bash
pytest -q
```

Cobertura atual:

- validação de configuração;
- regras de métricas de sessão;
- autorização e comportamento dos endpoints principais.

## Deploy (Produção)

Recomendado:

- rodar com `uvicorn` atrás de proxy reverso (Nginx/Caddy);
- configurar tokens de API e CORS restritivo;
- usar observabilidade de logs centralizada;
- habilitar restart supervisionado (systemd, Docker ou process manager).

Exemplo:

```bash
uvicorn app.backend.api:app --host 0.0.0.0 --port 8000 --workers 1
```

## Boas Práticas Adotadas

- contratos de API explícitos (Pydantic);
- separação clara de responsabilidades;
- validação forte de entradas;
- tratamento de erros previsíveis com feedback claro;
- interface focada em legibilidade operacional e fluxo contínuo.

## Melhorias Futuras

- persistência de sessões em banco (PostgreSQL/TimescaleDB);
- autenticação OAuth2/JWT com RBAC completo;
- suporte a múltiplas câmeras e múltiplos workers;
- relatórios PDF/CSV com insights por janela temporal;
- monitoramento com Prometheus/Grafana.

Autoria: Matheus Siqueira  
Website: https://www.matheussiqueira.dev/
