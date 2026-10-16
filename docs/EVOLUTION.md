# EmotionAI — Documento Master de Evolução

**Versão do produto:** 2.1.0 · **Data:** 15/10/2026 · **Branch:** `feat/enterprise-hardening`

Este documento registra a passagem do EmotionAI de projeto pessoal para uma base de produto: o que foi diagnosticado, o que foi implementado (e medido), o que ficou como roteiro e por quê. Itens marcados **✅ Implementado** estão no código e cobertos por testes; itens **🧭 Proposto** são roteiro, com código de referência **não testado**.

---

## 1. Sumário executivo

O EmotionAI já tinha uma boa separação em camadas (`core`, `backend`, `frontend`). O problema não era a arquitetura em si, e sim as fronteiras: estado global no módulo da API, segurança permissiva por padrão, erros sem contrato, configuração que prometia funções inexistentes e uma demo que **parava sozinha após ~40 segundos** e **não funcionava com a câmera de quem estava assistindo**.

A evolução preservou a arquitetura (sem reescrita) e atacou esses pontos com mudanças pequenas, verificáveis e reversíveis.

| Indicador | Antes | Depois |
|---|---|---|
| Testes automatizados | 13 | **65** (sem TensorFlow: dublês via injeção de dependências) |
| Composição da API | Globais criados na importação do módulo | `create_app()` com injeção de dependências |
| Contrato de erro | Formato variável por rota | `{"error": {code, message, request_id}}` + `X-Request-ID` |
| Autorização | Comparação `==`, 401 para tudo | Tempo constante, **401 vs 403** corretos |
| Superfície com API aberta | Qualquer um trocava `video_source` (arquivo/URL arbitrários) | Recusado sem tokens configurados |
| Cabeçalhos de segurança | Nenhum | CSP, `X-Frame-Options`, `nosniff`, `Referrer-Policy`, `Permissions-Policy` |
| Dependência de CDN | Chart.js sem versão e sem SRI | Versão fixa 4.5.1 + `integrity` (SRI) |
| Modos de demo | Vídeo do servidor (congela no fim do arquivo) | Vídeo **em loop** + **câmera do próprio usuário** |
| Observabilidade | JSON de métricas | JSON + **formato Prometheus** + ID de requisição nos logs |

**Resultados medidos** (container Docker, CPU, 1 rosto, quadros de 640 px):

| Medida | Valor |
|---|---|
| Modo câmera: latência de análise | **p50 29,6 ms · p95 80,5 ms** |
| Modo câmera: vazão | **21,9 análises/s** (envio *stop-and-wait*) |
| Modo câmera: primeira sessão pronta | 12,1 s (importação do TensorFlow + *warmup*, fora do event loop) |
| `/health` durante o carregamento do modelo | mediana 2–3 ms · máximo 0,17–0,37 s |
| Vídeo em loop | 1.400 quadros recebidos de um vídeo de 1.210 (loop confirmado) |

**Bugs pré-existentes corrigidos nesta fase:** `PATCH /config` deixava a configuração em estado inválido quando rejeitado; campos desconhecidos no `PATCH` eram ignorados em silêncio (respondia 200 sem mudar nada); câmera local usava `CAP_DSHOW` (só existe no Windows), então nunca abria no Linux/Docker; o gráfico de distribuição alternava entre contagens e percentuais.

**Não validado:** o modo câmera com uma webcam física (o navegador usado nos testes bloqueia câmera). O protocolo foi validado ponta a ponta enviando quadros reais do vídeo de demonstração pelo mesmo WebSocket que o navegador usa.

---

## 2. Skills ativadas e racional

| Skill | Uso nesta entrega |
|---|---|
| `ultrapowers-dev:fastapi-patterns` | Fábrica da aplicação, `Depends`, organização em `APIRouter`, `response_model` nas rotas |
| `ultrapowers-dev:error-handling` | "Validar na fronteira, traduzir erros na borda": contrato único de erro, 500 sem vazar detalhes internos, sem engolir exceções |
| `security-review` | Auditoria do diff (autorização, entradas externas, XSS, exposição de dados). Resultado: nenhuma vulnerabilidade com confiança ≥ 8 |
| `engineering:architecture` | Formato dos ADRs da seção 7 |

**Deixadas de fora de propósito:** skills de SEO, marca e design system. O produto é um painel operacional autenticado, não uma página de marketing; a interface existente já tinha boa base visual e recebeu ajustes pontuais em vez de um redesign.

---

## 3. Diagnóstico (Fase 1, resumo)

| Área | Achado | Status |
|---|---|---|
| Acoplamento | `runtime_config` e `stream_service` globais; testes mutavam estado do módulo | ✅ `create_app()` + DI |
| Segurança | API aberta por padrão podia apontar o servidor para qualquer arquivo/URL | ✅ Bloqueado sem tokens; aviso no log ao iniciar |
| Segurança | `==` em tokens; 401 para permissão insuficiente; CORS `*` + credenciais; CDN sem SRI | ✅ Corrigidos |
| Erros | Sem tratamento global, sem correlação | ✅ Envelope + `X-Request-ID` |
| Configuração | 6 campos sem efeito (`record_session`, `output_path`, `draw_landmarks`, `show_fps`, `theme_color`, `enable_gpu`) | ✅ Removidos |
| Demo | Vídeo acaba e o painel fica "Online" congelado | ✅ Loop de vídeo |
| Demo | Webcam inacessível no container | ✅ Modo câmera do navegador |
| Desempenho | Inferência síncrona no loop de captura | 🧭 R1 |
| Desempenho | Quadros em base64 dentro de JSON (+33%) | 🧭 R6 |
| Conformidade | Nenhuma menção a AI Act/LGPD | ✅ README + 🧭 R7 |

---

## 4. Melhorias

### 4.1 Arquitetura

- ✅ **`FramePipeline`** (`app/core/pipeline.py`): a lógica por quadro (detectar → rastrear → classificar) saiu do `VideoProcessor` e passou a ser compartilhada pelo vídeo do servidor e pelo modo câmera. Detector e analisador entram por *ports* (`Protocol`), o que permite testar sem OpenCV/TensorFlow e trocar o detector (R5) sem tocar no pipeline.
- ✅ **Objetos de valor sem dependências pesadas** (`app/core/models.py`): `FaceBox` e `EmotionResult` saíram de módulos que importam OpenCV/DeepFace. Resultado verificável: importar a API não carrega TensorFlow, DeepFace nem OpenCV.
- ✅ **Raiz de composição** (`create_app()`): configuração, `StreamService` e `BrowserAnalysisService` são criados num único lugar ou injetados (testes, outros deployments). `app = create_app()` mantém compatibilidade com `uvicorn app.backend.api:app`.
- ✅ **Atualização de configuração atômica**: validação numa cópia (`dataclasses.replace`) antes de aplicar.
- 🧭 **R1** Pool de inferência desacoplado do loop de captura.
- 🧭 **R3** Persistência de agregados de sessão.

### 4.2 Segurança

- ✅ Comparação de tokens em tempo constante (`hmac.compare_digest`) e papéis explícitos (`anonymous`, `reader`, `admin`).
- ✅ 401 (sem credencial válida) vs 403 (credencial válida sem permissão); WebSocket fecha com 4401/4403.
- ✅ CORS sem `allow_credentials` (auth é por cabeçalho; credenciais só ampliariam a exposição) e cabeçalhos permitidos explícitos.
- ✅ Middleware de cabeçalhos de segurança com CSP restritiva (exceto nas páginas de documentação do FastAPI, que dependem de script inline).
- ✅ `X-Request-ID` aceito apenas se casar com `^[A-Za-z0-9._-]{1,64}$` (sem injeção de cabeçalho).
- ✅ Modo câmera: só JPEG; tamanho e resolução verificados no cabeçalho antes de decodificar (um JPEG de 1,5 MB podia declarar 10000×10000 px e ocupar ~300 MB); quadros decodificados em memória e descartados; sessões simultâneas limitadas (fecha com 1013).
- ✅ **R2** Token de WebSocket fora da URL: enviado na primeira mensagem (`{"type": "auth"}`); `?token=` é recusado. Origem de navegador verificada contra a própria origem ou `EMOTION_CORS_ORIGINS`.
- 🧭 **R9** Chaves por cliente e OAuth2 para B2B.

### 4.3 Visual e experiência de demonstração

- ✅ Seletor **Vídeo do servidor / Minha câmera** acessível (radios nativos em `fieldset` com `legend`, foco visível).
- ✅ Modo câmera com desenho a 60 fps via `requestAnimationFrame` e caixas atualizadas na cadência da inferência; aviso de privacidade visível no modo câmera.
- ✅ Mensagens claras para câmera negada, limite de sessões (1013) e token inválido (4401/4403).
- ✅ Troca de fonte limpa o canvas e zera indicadores (antes ficava o último quadro da fonte anterior).
- ✅ Gráfico de distribuição sempre em percentuais.
- 🧭 Indicador de latência no painel e tela de consentimento antes da câmera (R7).

### 4.4 Produto e negócio

**Reposicionamento.** "Monitoramento emocional" genérico é juridicamente arriscado e cientificamente frágil. A proposta é posicionar o EmotionAI como **analytics de reação para pesquisa de UX e testes de mídia, com consentimento e privacidade desde a concepção**:

> *EmotionAI mede reações faciais agregadas, com consentimento, durante testes de usabilidade e exibições de conteúdo — sem gravar imagens.*

**Por que esse recorte:**

- **Regulação.** O AI Act da UE (art. 5(1)(f), aplicável desde fevereiro de 2025) proíbe sistemas que inferem emoções no local de trabalho e em instituições de ensino, salvo razões médicas ou de segurança; sistemas de reconhecimento de emoções também entram na lista de alto risco e têm dever de transparência perante as pessoas expostas. Na LGPD, dado biométrico é dado pessoal sensível (art. 5º, II), e o tratamento depende, em regra, de consentimento específico e destacado (art. 11, I). *Isto é contexto de produto, não aconselhamento jurídico.*
- **Ciência.** A revisão de Barrett et al. (2019, *Psychological Science in the Public Interest*) mostra que expressões faciais não mapeiam de forma confiável emoções internas. O produto deve vender **tendências agregadas de reação a um estímulo**, não "o que a pessoa sente".

**Clientes-alvo:** times de pesquisa de UX; agências de teste de anúncios e trailers; laboratórios acadêmicos com protocolo aprovado em comitê de ética.

**Features de valor agregado:**

| Feature | Valor | Status |
|---|---|---|
| Modo câmera do navegador | Qualquer participante testa sem instalar nada | ✅ |
| Métricas Prometheus | Encaixa no stack de observabilidade do cliente | ✅ |
| Sessões com linha do tempo sincronizada ao estímulo (vídeo/protótipo) | O "produto" de fato para pesquisa de UX | 🧭 R3 |
| Webhooks assinados (ex.: pico de reação negativa) | Integração B2B com ferramentas de pesquisa | 🧭 R4 |
| Relatórios por sessão (CSV/PDF) | Entregável para o cliente final da agência | 🧭 |
| Consentimento e retenção configuráveis | Requisito de venda em mercados regulados | 🧭 R7 |

**Monetização sugerida:** núcleo self-hosted aberto (atração de desenvolvedores e credibilidade técnica); plano Team em SaaS cobrado por **hora de sessão analisada**; Enterprise on-premises com SSO, retenção customizada e suporte. KPIs: horas analisadas/mês, sessões por cliente/semana, retenção de contas em 90 dias.

---

## 5. Guias táticos de implementação

Prioridade por impacto × esforço:

| # | Item | Impacto | Esforço | Prioridade |
|---|---|---|---|---|
| R1 | Pool de inferência desacoplado | Alto (FPS com vários rostos) | Médio | 1 |
| R7 | Consentimento e governança | Alto (venda em mercado regulado) | Baixo | 1 |
| R2 | Token de WebSocket fora da URL | Médio | Baixo | 2 |
| R8 | CI ampliado | Médio | Baixo | 2 |
| R3 | Persistência de agregados | Alto (habilita o produto de pesquisa) | Médio | 3 |
| R5 | Detector YuNet | Médio (precisão em ângulos) | Baixo | 3 |
| R4 | Webhooks assinados | Médio (B2B) | Médio | 4 |
| R6 | Quadros binários no stream do servidor | Médio (banda) | Médio | 4 |
| R9 | Multi-tenant (chaves por cliente / OAuth2) | Alto para SaaS | Alto | 5 |

### R1 — Pool de inferência desacoplado

- **Problema:** o DeepFace roda dentro do loop de captura; cada rosto novo custa dezenas de milissegundos de bloqueio, e o FPS cai com vários rostos.
- **Mudança:** o `FramePipeline` passa a *enviar* recortes a um worker e a *ler* o último resultado por trilha. A política "o recorte mais recente vence" impede filas crescentes.
- **Bibliotecas:** nenhuma nova (`threading`).
- **Negócio:** a demo continua fluida com várias pessoas em cena, o cenário de uma sala de teste de mídia.
- **Código:** seção 6.7.

### R2 — Token de WebSocket fora da URL (✅ implementado via primeira mensagem)

- **Problema:** navegadores não enviam cabeçalhos em WebSocket, então o token vai em `?token=` e pode aparecer em logs de acesso e de proxy.
- **Mudança:** enviar o token como subprotocolo (`new WebSocket(url, ["emotionai.v1", "bearer.<token>"])`) e aceitar com `subprotocol="emotionai.v1"`. Os tokens precisam usar apenas caracteres válidos em subprotocolo (ex.: base64url).
- **Negócio:** requisito comum em questionários de segurança de clientes enterprise.

### R3 — Persistência de agregados de sessão

- **Mudança:** gravar **apenas agregados por minuto** (sem imagens nem vetores faciais): quadros, rostos analisados, sentimento médio e participação de cada emoção.
- **Bibliotecas:** SQLAlchemy 2 + Alembic + PostgreSQL (TimescaleDB opcional para séries temporais).
- **Negócio:** habilita histórico, comparação entre sessões e relatórios — o núcleo do produto de pesquisa.

```sql
-- Proposto (não testado): agregados sem dados biométricos.
CREATE TABLE session_minute_aggregates (
    session_id     uuid        NOT NULL,
    bucket         timestamptz NOT NULL,
    frames         integer     NOT NULL,
    faces_analyzed integer     NOT NULL,
    avg_sentiment  real,
    emotion_share  jsonb       NOT NULL,
    PRIMARY KEY (session_id, bucket)
);
```

### R4 — Webhooks assinados

- **Mudança:** eventos como "sentimento médio abaixo de −0,5 por 30 s" são enviados para URLs cadastradas pelo cliente, assinados com HMAC-SHA256 e timestamp (proteção contra replay).
- **Bibliotecas:** cliente HTTP assíncrono (`httpx`); fila com retentativas e *backoff* exponencial.
- **Código:** seção 6.7.

### R5 — Detector YuNet

- **Mudança:** implementar um `YuNetDetector` com `cv2.FaceDetectorYN` (OpenCV 4.x) e injetá-lo no `FramePipeline` pelo `FaceDetectorPort`. Requer o modelo ONNX `face_detection_yunet` do `opencv_zoo`.
- **Negócio:** o Haar Cascade falha com rosto inclinado e pouca luz, que são condições reais de teste.
- **Código:** seção 6.7.

### R6 — Quadros binários no stream do servidor

- **Mudança:** enviar o JPEG como mensagem binária e os metadados como JSON (ou um envelope binário com prefixo de tamanho), eliminando o base64.
- **Negócio:** cerca de 25% menos banda por quadro, o que importa em demos remotas.

### R7 — Consentimento e governança

- **Mudança:** tela de consentimento antes de ativar a câmera (finalidade, retenção e direito de recusa), registro do aceite (data/hora e versão do texto, sem imagem) e política de retenção configurável dos agregados (R3).
- **Negócio:** sem isso, o produto não passa em compras de clientes regulados.

### R8 — CI ampliado

- **Mudança:** adicionar ao workflow lint (`ruff`), auditoria de dependências (`pip-audit`) e build da imagem Docker.
- **Atenção:** o código ainda não foi passado no `ruff`; a primeira execução vai exigir correções antes de virar bloqueante.

```yaml
# Proposto (não testado): passos extras em .github/workflows/tests.yml
- name: Lint
  run: pip install ruff && ruff check app tests
- name: Audit dependencies
  run: pip install pip-audit && pip-audit -r requirements.txt
```

### R9 — Multi-tenant

- **Mudança:** chaves de API por cliente, armazenadas como hash (SHA-256), com escopos (`read`, `admin`, `analyze`) e expiração; OAuth2 *client credentials* para integrações B2B; SSO (OIDC) no painel para o plano Enterprise.
- **Negócio:** pré-requisito do SaaS cobrado por cliente.

---

## 6. Blocos de código críticos por domínio

Os blocos 6.1–6.6 estão **implementados** no repositório (trechos; consulte o arquivo para a versão completa). Os blocos de 6.7 são **propostas não testadas**.

### 6.1 Domínio — `FramePipeline` (`app/core/pipeline.py`)

```python
class FramePipeline:
    def __init__(self, config, detector: FaceDetectorPort, analyzer: EmotionAnalyzerPort, tracker=None):
        ...

    def process(self, frame: np.ndarray, now: float | None = None) -> list[Track]:
        now = time.time() if now is None else now
        detections = self.detector.detect(frame, scale=self.config.detect_scale, ...)
        tracks = self.tracker.update(detections)
        for track in tracks:
            if (now - track.last_analyzed) < self.config.emotion_interval:
                continue
            emotion = self.analyzer.analyze(crop_face(frame, track.box))
            ...
        return tracks
```

### 6.2 Composição — `create_app()` (`app/backend/api.py`)

```python
def create_app(config=None, stream_service=None, analysis_service=None) -> FastAPI:
    config = config or AppConfig()
    stream_service = stream_service or StreamService(config)
    analysis_service = analysis_service or BrowserAnalysisService(config)
    app = FastAPI(title="EmotionAI API", version=__version__, lifespan=lifespan, ...)
    app.state.config = config
    app.state.stream_service = stream_service
    app.state.analysis_service = analysis_service
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins, allow_credentials=False, ...)
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)
    app.include_router(routes.router)
    app.include_router(websockets.router)
    return app
```

### 6.3 Segurança — autorização (`app/backend/security.py`)

```python
def _matches(candidate: str, expected: str | None) -> bool:
    return bool(expected) and hmac.compare_digest(candidate.encode(), expected.encode())

def authorize(config, token, permission: Permission) -> Role:
    role = resolve_role(config, token)
    if role is None or role is Role.ANONYMOUS:
        raise ApiError(401, "Missing or invalid API token.")
    if permission is Permission.ADMIN and role is not Role.ADMIN:
        raise ApiError(403, "This operation requires the admin token.")
    return role
```

### 6.4 Erros — contrato único (`app/backend/errors.py`)

```python
async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id(request)
    LOGGER.exception("Unhandled error (request_id=%s)", request_id)  # detalhe só no log
    return JSONResponse(status_code=500, content=error_body("internal_error", "Unexpected server error.", request_id))
```

### 6.5 Tempo real — modo câmera (`app/backend/websockets.py`)

```python
@router.websocket("/api/v1/ws/analyze")
async def websocket_analyze(websocket: WebSocket) -> None:
    if not await authorize_websocket(websocket, state.config, Permission.READ):
        return
    await websocket.accept()
    try:
        async with state.analysis_service.session() as session:   # limite de sessões
            await websocket.send_json({"type": "ready"})
            while True:
                message = await websocket.receive()
                ...
                result = await session.analyze(message.get("bytes") or b"")  # inferência em thread
                await websocket.send_json(result)
    except SessionLimitReached:
        await _close_quietly(websocket, WS_CLOSE_TRY_AGAIN_LATER, "Too many concurrent camera sessions.")
```

### 6.6 Frontend — envio *stop-and-wait* (`app/frontend/app.js`)

```js
// Um quadro em trânsito por vez: servidor lento reduz o FPS em vez de acumular fila.
function sendCameraFrame() {
    const scale = Math.min(1, CAMERA_MAX_WIDTH / video.videoWidth);
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCanvas.toBlob(async (blob) => ws.send(await blob.arrayBuffer()), "image/jpeg", CAMERA_JPEG_QUALITY);
}
```

### 6.7 Propostas (não testadas)

**R1 — worker de inferência ("o recorte mais recente vence"):**

```python
class InferenceWorker:
    """Runs emotion inference off the capture loop; a newer crop for a track replaces the queued one."""

    def __init__(self, analyzer: EmotionAnalyzerPort):
        self._analyzer = analyzer
        self._pending: dict[int, np.ndarray] = {}
        self._results: dict[int, EmotionResult] = {}
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="emotion-inference", daemon=True)
        self._thread.start()

    def submit(self, track_id: int, face: np.ndarray) -> None:
        with self._lock:
            self._pending[track_id] = face.copy()
        self._wakeup.set()

    def latest(self, track_id: int) -> EmotionResult | None:
        with self._lock:
            return self._results.get(track_id)

    def close(self) -> None:
        self._stopped.set()
        self._wakeup.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stopped.is_set():
            self._wakeup.wait(timeout=0.5)
            self._wakeup.clear()
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    track_id, face = self._pending.popitem()
                result = self._analyzer.analyze(face)
                if result:
                    with self._lock:
                        self._results[track_id] = result
```

**R2 — token pelo subprotocolo do WebSocket:**

```python
def token_from_subprotocols(websocket: WebSocket) -> str | None:
    offered = [item.strip() for item in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    for protocol in offered:
        if protocol.startswith("bearer."):
            return protocol.removeprefix("bearer.") or None
    return None

# Após autorizar: o servidor precisa aceitar um subprotocolo oferecido pelo cliente.
await websocket.accept(subprotocol="emotionai.v1")
```

**R4 — assinatura de webhook:**

```python
def sign_webhook(secret: bytes, timestamp: int, body: bytes) -> str:
    digest = hmac.new(secret, f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"

async def deliver_webhook(url: str, secret: bytes, event: dict) -> None:
    body = json.dumps(event, separators=(",", ":")).encode()
    timestamp = int(time.time())
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            url,
            content=body,
            headers={"Content-Type": "application/json", "X-EmotionAI-Signature": sign_webhook(secret, timestamp, body)},
        )
        response.raise_for_status()
```

**R5 — detector YuNet injetável:**

```python
class YuNetDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.8):
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold)

    def detect(self, frame: np.ndarray, **_unused) -> list[FaceBox]:
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces[:, :4]]

# Uso: FramePipeline(config, detector=YuNetDetector(Path("models/face_detection_yunet.onnx")), analyzer=...)
```

---

## 7. Decisões de arquitetura (ADRs)

### ADR-001: Fábrica da aplicação com injeção de dependências

**Status:** Aceito · **Data:** 15/10/2026

**Contexto.** `api.py` criava configuração e serviço como globais ao ser importado. Os testes alteravam esse estado e o restauravam manualmente; era impossível testar o modo câmera sem TensorFlow ou subir duas instâncias isoladas.

**Decisão.** `create_app(config, stream_service, analysis_service)` como raiz de composição; colaboradores em `app.state` e resolvidos por `Depends`. O módulo mantém `app = create_app()` para compatibilidade.

**Opções consideradas.**

| Opção | Complexidade | Testabilidade | Observação |
|---|---|---|---|
| A. Globais + `dependency_overrides` | Baixa | Média | Mantém estado compartilhado entre testes |
| **B. Fábrica + `app.state` + `Depends`** | Baixa | Alta | Padrão idiomático do FastAPI |
| C. Contêiner de DI (ex.: `dependency-injector`) | Média | Alta | Dependência nova sem ganho proporcional neste tamanho de projeto |

**Consequências.** Testes criam apps isolados com dublês (65 testes sem TensorFlow). Novos colaboradores (ex.: repositório de sessões do R3) entram pela fábrica. Quem importava `runtime_config`/`stream_service` do módulo precisa usar `app.state`.

### ADR-002: Modo câmera por envio de quadros via WebSocket

**Status:** Aceito · **Data:** 15/10/2026

**Contexto.** Containers não acessam a webcam do host, e a demo precisava funcionar com o rosto de quem está assistindo, de qualquer máquina.

**Opções consideradas.**

| Opção | Complexidade | Latência | Privacidade | Observação |
|---|---|---|---|---|
| A. Repasse de USB (usbipd/V4L2) | Alta, por máquina | Baixa | Igual à atual | Só serve para a máquina do servidor |
| B. WebRTC até o servidor | Alta (sinalização, STUN/TURN) | Baixíssima | Igual | Exagero para análise a ~20 fps |
| **C. JPEG reduzido via WebSocket, *stop-and-wait*** | Baixa | p50 ~30 ms | Quadros só em memória | Reaproveita o `FramePipeline` |
| D. Inferência no navegador (ONNX Runtime Web / TF.js) | Média | Depende do aparelho | Máxima (nada sai do navegador) | Exige portar e validar o modelo |

**Decisão.** Opção C agora; D como evolução para o plano com requisito de privacidade máxima.

**Consequências.** Funciona em qualquer navegador moderno (exige HTTPS ou localhost). Cada sessão consome CPU do servidor, por isso existe o limite `EMOTION_MAX_ANALYZE_SESSIONS`. O controle de fluxo *stop-and-wait* faz a taxa se adaptar sozinha à capacidade do servidor.

---

## 8. Validação executada

| Verificação | Resultado |
|---|---|
| `pytest -q` (Windows e container) | 65 passed |
| Importar a API não carrega TensorFlow/DeepFace/OpenCV | Confirmado |
| Modo câmera ponta a ponta (quadros reais via WebSocket) | 120 quadros, p50 29,6 ms, p95 80,5 ms |
| Vídeo em loop | 1.400 quadros de um vídeo de 1.210 |
| Dois clientes simultâneos + `/health` durante o carregamento | 1 processador, `/health` com mediana de 2–3 ms |
| Navegador: Chart.js com SRI sob CSP | Carregou (4.5.1), console sem erros |
| Navegador: troca de fonte | Canvas limpo e indicadores zerados |
| Navegador: largura de celular (375 px) | Sem rolagem horizontal |
| Revisão de segurança do diff | Nenhuma vulnerabilidade com confiança ≥ 8 |
| **Não validado** | Webcam física no modo câmera; latência em GPU; carga com várias sessões de câmera simultâneas |
