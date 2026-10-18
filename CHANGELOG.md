# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/). As versões anteriores a esta entrada não tinham changelog; o histórico delas está nos commits e em [docs/EVOLUTION.md](docs/EVOLUTION.md). Nenhuma versão nova foi publicada; `app.__version__` continua `2.1.0`.

## [Não publicado]

### Adicionado

- Suavização temporal das pontuações de expressão **por rosto rastreado** (média móvel exponencial), configurável por `EMOTION_SMOOTHING` / `PATCH /api/v1/config {"emotion_smoothing": ...}` e pelo painel. Valor em `[0, 1)`; `0` reproduz a saída bruta.
- Campo `raw_dominant` em cada `emotion` dos WebSockets `/ws/stream` e `/ws/analyze`: rótulo do modelo na inferência mais recente.
- Detector de rostos YuNet opcional (`EMOTION_FACE_DETECTOR=yunet`, `EMOTION_YUNET_MODEL=<arquivo .onnx>`), carregado da memória para funcionar em caminhos com acentos no Windows. Seleção apenas por variável de ambiente. Validado com `face_detection_yunet_2023mar.onnx` (o arquivo não é versionado).
- `requirements-test.in` (runtime + pytest, httpx2, ruff, mypy com versões exatas).
- Script opt-in `scripts/measure_smoothing.py` que conta trocas de rótulo bruto e suavizado com o modelo real.
- Documentos do ciclo: `docs/REPOSITORY_TECHNICAL_AUDIT.md`, `docs/GITHUB_RESEARCH_AND_FEATURES.md`, `docs/ROADMAP_AND_SKILLS.md`, `docs/MASTER_EVOLUTION_PLAN.md`, `docs/VALIDATION_REPORT.md`.

### Alterado

- **Comportamento padrão:** `dominant` e `scores` passam a ser suavizados (`emotion_smoothing = 0.5`). A distribuição de expressões das métricas conta os rótulos suavizados, que são os exibidos no painel. Para o comportamento anterior, defina `EMOTION_SMOOTHING=0`.
- `min_emotion_score` é aplicado só uma vez, à saída bruta do modelo, dentro do `FramePipeline`. As métricas deixaram de refiltrar pontuações (que agora podem estar suavizadas).
- O rastreador devolve os rastros ordenados por ID.
- **Segurança:** com só `EMOTION_API_ADMIN_TOKEN` definido, a leitura (modo câmera, stream, status, métricas, `GET /config`) passa a ser pública; alterações continuam exigindo o token de admin. Antes, qualquer token configurado fechava também a leitura. Para manter a leitura restrita, defina `EMOTION_API_READ_TOKEN`.
- O repositório deixa de versionar o vídeo de demonstração `Face-emotions.mp4`. Os scripts opt-in (`verify_demo.py`, `measure_smoothing.py`, `verify_ui.py`) e o teste real do YuNet leem um vídeo local de `EMOTION_DEMO_VIDEO`; o Docker Compose usa `demo.mp4` por padrão (`VIDEO_DIR`/`VIDEO_FILE`).
- CI: novo job de lint (`ruff check`, `ruff format --check`) e `mypy` antes dos testes; instala os pins exatos em vez das faixas de `requirements.txt`.
- `requirements-dev.in` passa a incluir `requirements-test.in` e fixa versões de pip-audit, pip-tools e Playwright.
- `StreamFrameDataSchema` documenta `analysis_enabled`, que já era enviado.

### Corrigido

- Rastreador: o casamento por rastro, na ordem de criação, deixava um rastro antigo ficar com a caixa do vizinho quando os rostos estavam próximos, trocando IDs. Agora o casamento é guloso global por IoU decrescente.
- Métricas: rótulos exibidos no overlay podiam ficar fora da distribuição quando a pontuação suavizada caía abaixo de `min_emotion_score`.
- Painel: ao estreitar a janela (redimensionar ou girar o dispositivo), os gráficos mantinham a largura anterior e criavam rolagem horizontal. As trilhas do grid usam `minmax(0, …)` e o canvas é limitado à largura do contêiner.
- Formatação de blocos de código em `docs/EVOLUTION.md` segundo `ruff format`.

### Segurança

- O caminho do modelo YuNet não é aceito por `PATCH /api/v1/config` (o esquema rejeita o campo com 422), seguindo a regra já usada para `video_source`: a API não aponta o servidor para arquivos arbitrários.
- `*.onnx` adicionado ao `.gitignore` para evitar versionar pesos baixados.
