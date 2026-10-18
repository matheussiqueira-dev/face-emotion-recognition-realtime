# EmotionAI Realtime

Plataforma de reconhecimento de emoções faciais em tempo real com backend FastAPI, pipeline de visão computacional em Python e dashboard web responsivo para operação contínua.

## Visão Geral

O sistema foi estruturado para cenários de monitoramento em tempo real (laboratórios, protótipos de atendimento, pesquisa comportamental e observabilidade operacional), com foco em:

- baixa latência de streaming;
- clareza visual para operadores;
- configuração dinâmica sem reinício;
- segurança para ambientes de produção;
- arquitetura evolutiva e testável.

> 📘 Roteiro técnico e de produto completo: [docs/EVOLUTION.md](docs/EVOLUTION.md). Ciclo de modernização mais recente: [plano mestre](docs/MASTER_EVOLUTION_PLAN.md), [auditoria](docs/REPOSITORY_TECHNICAL_AUDIT.md), [validação](docs/VALIDATION_REPORT.md), [pesquisa](docs/GITHUB_RESEARCH_AND_FEATURES.md), [roadmap](docs/ROADMAP_AND_SKILLS.md) e [CHANGELOG](CHANGELOG.md).

## Destaques técnicos

| Problema | Solução | Evidência |
|---|---|---|
| O rótulo de cada rosto oscilava entre quadros (classificação independente a cada inferência) | Suavização exponencial das pontuações **por rosto rastreado** (`EMOTION_SMOOTHING`, padrão 0,5), com o rótulo bruto do modelo exposto em `raw_dominant` | Testes em `tests/test_pipeline.py`; no vídeo de demonstração, 600 quadros com o modelo real: 9 trocas de rótulo brutas → 3 suavizadas (uma execução, um rosto; não é benchmark) |
| O rastreador casava rostos na ordem de criação, e um rastro antigo podia "roubar" a caixa do vizinho, trocando IDs | Casamento guloso global por IoU decrescente | Teste de regressão `test_tracker_matches_best_overlaps_first_so_neighbours_keep_their_ids` (falhava antes da correção) |
| Haar Cascade perde rostos inclinados | Detector **YuNet** opcional (`EMOTION_FACE_DETECTOR=yunet`), CNN do OpenCV com modelo MIT de ~230 KB, carregado da memória (caminhos com acentos no Windows) | Nos 1.210 quadros do vídeo, ambos acham o rosto em todos; YuNet leva p50 3,5 ms contra 8,2 ms do Haar. Com os quadros girados (teste sintético), YuNet acha 120/121 a 30° contra 1/121 do Haar, mas **0/121 a 45°** |
| CI só rodava `pytest`, com faixas de versão | CI com `ruff check`, `ruff format --check`, `mypy` e testes, instalando os pins exatos de `requirements-test.in` | `.github/workflows/tests.yml` |

## Modos de demonstração

| Modo | Como funciona | Quando usar |
|---|---|---|
| **Vídeo do servidor** | O servidor lê uma câmera local ou um arquivo de vídeo (em loop) e transmite frames anotados | Demos sem câmera, apresentações e telões |
| **Minha câmera** | O navegador captura a webcam, envia quadros reduzidos ao servidor e recebe as detecções; nada é gravado | Qualquer pessoa testa com o próprio rosto, inclusive com o servidor em Docker |

O modo câmera exige `https://` ou `localhost` (regra dos navegadores para `getUserMedia`).

## Objetivo de Negócio

- transformar vídeo em dados emocionais acionáveis;
- disponibilizar métricas operacionais e analíticas em tempo real;
- reduzir custo de operação com painel único de monitoramento e controle.

## Arquitetura e Decisões Técnicas

### Camadas

- `app/core`: domínio e processamento. `FramePipeline` concentra a lógica por quadro (detectar → rastrear → classificar) e recebe detector e analisador por injeção; `models.py` guarda os objetos de valor sem dependências pesadas.
- `app/backend`: entrega (REST + WebSocket). `create_app()` é a raiz de composição; rotas recebem colaboradores via `Depends`, com erros padronizados, ID de requisição e cabeçalhos de segurança.
- `app/frontend`: dashboard operacional, visualização ao vivo, analytics e controles de runtime.

A API sobe sem importar TensorFlow, DeepFace ou OpenCV; o modelo só é carregado (numa thread) quando a primeira transmissão começa.

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
│   │   ├── api.py            # create_app(): composição, middlewares, rotas
│   │   ├── analysis.py       # modo câmera do navegador (sessões limitadas)
│   │   ├── dependencies.py   # injeção de dependências (Depends)
│   │   ├── errors.py         # contrato de erro padronizado
│   │   ├── middleware.py     # X-Request-ID e cabeçalhos de segurança
│   │   ├── routes.py         # REST /api/v1
│   │   ├── schemas.py
│   │   ├── security.py       # tokens, papéis, 401/403
│   │   ├── serializers.py
│   │   ├── services.py       # StreamService (vídeo do servidor)
│   │   └── websockets.py     # /ws/stream e /ws/analyze
│   ├── core
│   │   ├── analyzer.py       # DeepFace
│   │   ├── config.py
│   │   ├── detector.py       # Haar Cascade
│   │   ├── metrics.py
│   │   ├── models.py         # FaceBox, EmotionResult
│   │   ├── pipeline.py       # FramePipeline
│   │   ├── processor.py      # captura de vídeo do servidor
│   │   └── tracker.py
│   └── frontend
├── docs/EVOLUTION.md
├── tests
├── Dockerfile / docker-compose.yml
├── main.py / run_api.py / run.bat
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

### Opcional: detector YuNet

O YuNet ([opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), licença MIT) detecta rostos inclinados e parcialmente virados que o Haar Cascade perde. O modelo não acompanha o repositório. Para OpenCV 4.x, use `face_detection_yunet_2023mar.onnx` (~230 KB):

```bash
mkdir models
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Depois inicie com `EMOTION_FACE_DETECTOR=yunet` e `EMOTION_YUNET_MODEL=models/face_detection_yunet_2023mar.onnx`. Um caminho relativo é resolvido a partir do diretório em que o servidor é iniciado; se o arquivo não existir, a aplicação recusa iniciar com uma mensagem explícita. Para validar com um vídeo local que mostre um rosto: `EMOTION_YUNET_MODEL=models/face_detection_yunet_2023mar.onnx EMOTION_DEMO_VIDEO=<vídeo> pytest tests/test_yunet_detector.py`.

Comparação no vídeo de demonstração (detalhes em [VALIDATION_REPORT](docs/VALIDATION_REPORT.md)): para rostos frontais, os dois detectores acertam; o YuNet é mais rápido e tolera inclinações de até ~30°, mas não detectou o rosto girado a 45°. O padrão continua `haar`.

### Alternativa: Docker

Útil quando o TensorFlow não carrega no host (por exemplo, DLLs bloqueadas pelo Controle Inteligente de Aplicativos do Windows). O container processa um arquivo de vídeo; webcam não é repassada por padrão.

```bash
docker compose up -d --build
```

- Interface: `http://127.0.0.1:8001` (porta do host configurável com `HOST_PORT`).
- Vídeo: o repositório não inclui vídeo de demonstração. Por padrão o container lê `demo.mp4` da raiz do projeto (ignorado pelo Git); para outro arquivo, defina `VIDEO_DIR` (pasta) e `VIDEO_FILE` (nome). Sem vídeo, use o modo "Minha câmera" no navegador.
- Os pesos do DeepFace são baixados no primeiro stream e ficam no volume `deepface-models`.
- Testes no container: `docker compose run --rm emotionai python -m pytest -q`.

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|---|---|---|
| `EMOTION_VIDEO_SOURCE` | Câmera (`0`) ou caminho de vídeo | `0` |
| `EMOTION_WIDTH` | Largura da captura | `1280` |
| `EMOTION_HEIGHT` | Altura da captura | `720` |
| `EMOTION_MAX_FPS` | Limite de FPS do pipeline | `30` |
| `EMOTION_INTERVAL` | Intervalo de inferência por face (s) | `0.5` |
| `EMOTION_MIN_SCORE` | Confiança mínima da emoção (%), aplicada à saída bruta do modelo | `0` |
| `EMOTION_SMOOTHING` | Suavização das pontuações por rosto: peso da estimativa anterior, em `[0, 1)`; `0` desliga | `0.5` |
| `EMOTION_FACE_DETECTOR` | Detector de rostos: `haar` (incluído no OpenCV) ou `yunet` (exige `EMOTION_YUNET_MODEL`) | `haar` |
| `EMOTION_YUNET_MODEL` | Caminho do arquivo `.onnx` do YuNet. Só por variável de ambiente, não pela API | vazio |
| `EMOTION_JPEG_QUALITY` | Qualidade do frame no WebSocket | `80` |
| `EMOTION_LOOP_VIDEO` | Reinicia arquivos de vídeo ao chegar no fim (demo contínua) | `true` |
| `EMOTION_MAX_FRAME_BYTES` | Tamanho máximo de um quadro enviado pelo navegador | `2000000` |
| `EMOTION_MAX_FRAME_PIXELS` | Resolução máxima (largura × altura) de um quadro do navegador, verificada no cabeçalho antes de decodificar | `8294400` (4K) |
| `EMOTION_MAX_ANALYZE_SESSIONS` | Sessões simultâneas do modo câmera | `4` |
| `EMOTION_API_READ_TOKEN` | Token para leitura (config/metrics/ws). Vazio com `EMOTION_API_ADMIN_TOKEN` definido = leitura pública (demo aberta, alterações só com o token de admin) | vazio |
| `EMOTION_API_ADMIN_TOKEN` | Token para operações administrativas | vazio |
| `EMOTION_CORS_ORIGINS` | Origens externas permitidas (csv). Requisições autenticadas de navegador só são aceitas da mesma origem ou desta lista | vazio |
| `EMOTION_MAX_STREAM_CLIENTS` | Clientes simultâneos no vídeo do servidor | `16` |

> Compatibilidade: `EMOTION_API_TOKEN` também é aceito como token legado.
>
> Sem tokens configurados, a API fica aberta (modo local/demo) e um aviso aparece no log ao iniciar. Nesse modo, trocar `video_source` pela API é recusado (403), porque permitiria apontar o servidor para qualquer arquivo ou URL.

## API (Resumo)

- `GET /api/v1/health` - healthcheck.
- `GET /api/v1/config` - configuração atual (perm. `read`).
- `GET /api/v1/status` - estado do processador de vídeo do servidor (`idle`, `starting`, `running`, `stopping`, `degraded`, `failed`) (perm. `read`).
- `PATCH /api/v1/config` - atualização de configuração (perm. `admin`).
- `GET /api/v1/metrics` - métricas da sessão ativa (perm. `read`).
- `GET /api/v1/metrics/prometheus` - as mesmas métricas no formato de texto do Prometheus (perm. `read`).
- `WS /api/v1/ws/stream` - stream de frames e detecções do vídeo do servidor (perm. `read`).
- `WS /api/v1/ws/analyze` - modo câmera: recebe quadros JPEG binários e responde `{"type": "analysis", "tracks": [...], "latency_ms": ...}` (perm. `read`).

Em cada rastro, `emotion` traz `dominant` e `scores` **suavizados** (conforme `emotion_smoothing`) e `raw_dominant`, o rótulo do modelo na inferência mais recente. Com `emotion_smoothing = 0`, os três refletem a saída bruta.

Token: cabeçalho `Authorization: Bearer <token>` ou `X-API-Key`. Nos WebSockets, navegadores não enviam cabeçalhos, então o token vai na **primeira mensagem** (`{"type": "auth", "token": "..."}`), nunca na URL; `?token=` é recusado. Token inválido ou ausente → `401`; token de leitura em operação de admin → `403`; origem de navegador não permitida → `403`.

Todos os erros seguem o mesmo formato, e toda resposta traz `X-Request-ID` para correlacionar com os logs:

```json
{"error": {"code": "forbidden", "message": "This operation requires the admin token.", "request_id": "9f1c..."}}
```

## Qualidade e Testes

Instalar ferramentas de teste (pins exatos usados pelo CI) e executar:

```bash
pip install -r requirements-test.in
ruff check . && ruff format --check . && mypy && pytest -q
```

Verificações opcionais com o modelo real (baixam pesos do DeepFace se ausentes, sobem um servidor em loopback e nunca abrem câmera física). Exigem um vídeo local com um rosto visível, fora do repositório, indicado em `EMOTION_DEMO_VIDEO`:

```bash
export EMOTION_DEMO_VIDEO=/caminho/para/video.mp4
cd scripts
python -X utf8 verify_demo.py
python -X utf8 measure_smoothing.py 600 0.5
```

Cobertura atual (81 testes + 1 opt-in, sem carregar TensorFlow; o pipeline usa detector e analisador substituídos por dublês via injeção, e o teste do detector usa o XML real do OpenCV):

- validação e atualização atômica de configuração;
- pipeline por quadro (intervalo de inferência, confiança mínima);
- regras de métricas de sessão e exportação Prometheus;
- autorização (401 vs 403), contrato de erro, ID de requisição e cabeçalhos de segurança;
- WebSocket do modo câmera (autenticação, formato e resolução dos quadros, limite de sessões);
- detecção de build sem GUI no `main.py`;
- suavização temporal por rosto, casamento global do rastreador e contrato do detector YuNet.
- carregamento do detector em caminhos com acentos e inicialização com saída CP1252 no Windows.

Diagnóstico verificado do ambiente e correções de restauração: [Fase 1](docs/PHASE1_DIAGNOSTIC.md).

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
- dashboards Grafana sobre o endpoint Prometheus.

Detalhes, priorização e guias de implementação em [docs/EVOLUTION.md](docs/EVOLUTION.md).

## Uso Responsável

Reconhecimento de emoções a partir da face é uma inferência probabilística, não uma leitura do estado interno de alguém, e é regulado: o AI Act da UE proíbe seu uso em ambiente de trabalho e de ensino, e a LGPD trata dados biométricos como sensíveis. Use apenas com consentimento explícito e em contextos como pesquisa de UX e testes de mídia. Veja a seção de conformidade em [docs/EVOLUTION.md](docs/EVOLUTION.md).

Autoria: Matheus Siqueira  
Website: https://www.matheussiqueira.dev/
