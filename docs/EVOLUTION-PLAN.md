# Plano de Evolução — face-emotion-recognition-realtime

> Documento vivo, revisado semanalmente. Última revisão: 2026-10-23.
> Regra central: nenhuma mudança entra sem estar ligada a um objetivo desta página e sem manter ou melhorar as métricas da seção 2.

## 1. Identidade do projeto

- **Propósito:** aplicação local (EmotionAI) que detecta rostos e estima a expressão facial aparente em 7 categorias, em tempo real, sobre vídeo do servidor ou sobre a câmera do próprio navegador, com painel web de métricas. Público: avaliador técnico que roda o projeto na própria máquina. Expressão aparente não é estado emocional; rosto é dado biométrico (dado pessoal sensível, LGPD art. 5º, II).
- **Tipo:** pipeline de visão computacional com API REST + WebSocket e painel web estático.
- **Stack:** Python 3.13 (`requires-python = ">=3.13,<3.14"`) · FastAPI 0.143.3 · pydantic 2.14.1 · uvicorn 0.54.0 · websockets 17.2 · OpenCV contrib headless 4.14.0.94 (Haar cascade e YuNet opcional) · DeepFace 0.0.102 + tf-keras 2.21.0 (TensorFlow 2.21) · NumPy 2.5.4 · Pillow 12.3.0 · frontend HTML/CSS/JS sem build com Chart.js 4.5.1 servido localmente (`app/frontend/vendor/`, com SRI) · pytest 9.1.1, ruff 0.17.1, mypy 2.4.0 (strict em 4 módulos de `app/core`).
- **Gerenciador de pacotes:** pip com arquivos `requirements*.in` (pins exatos dos diretos) e `requirements.txt` (faixas, usado pelo Dockerfile e por `run.bat`); sem lock de transitivas. `pyproject.toml` declara `[project]` espelhando `requirements.in` (teste `tests/test_packaging.py`).
- **Deploy:** nenhum publicado. Há `Dockerfile` (python:3.13-slim) e `docker-compose.yml` (porta presa a `127.0.0.1`), e `[tool.vercel] entrypoint = "app.backend.api:app"` no `pyproject.toml` (commits de 2026-10-17); nenhuma URL de produção no README.
- **Estrutura:** `app/core` (domínio: `FramePipeline`, detector Haar/YuNet, analisador DeepFace, rastreador, métricas, configuração), `app/backend` (`create_app()`, rotas, WebSockets `/ws/stream` e `/ws/analyze`, segurança, erros, middleware), `app/frontend` (painel), `tests/` (14 módulos), `scripts/` (smoke e medições opt-in com modelo real), `main.py` (modo janela OpenCV legado), `run_api.py`, `docs/`.
- **Comandos:** install `uv venv -p python3.13 .venv && uv pip install -p .venv/bin/python -r requirements-test.in` (CI: `pip install -r requirements-test.in`) · dev `python run_api.py` (http://127.0.0.1:8000) · lint `ruff check . && ruff format --check .` · typecheck `python -m mypy` · test `python -m pytest -q` · build `docker compose up --build` (sem etapa de build Python).
- **Documento master existente:** [`docs/EVOLUTION.md`](EVOLUTION.md) (2026-10-15, versão 2.1.0) concentra o diagnóstico da Fase 1, os ADRs (ADR-001 fábrica `create_app()` com DI; ADR-002 modo câmera via WebSocket *stop-and-wait*), as propostas R1–R9 e a validação registrada naquele ciclo. Este plano não o substitui: usa-o como fonte, atualiza o status de R1–R9 contra o código de 2026-10-18 (`57b4dae`) e define o ciclo de 4–8 semanas. Complementos do ciclo de 2026-10-17: `docs/MASTER_EVOLUTION_PLAN.md`, `docs/VALIDATION_REPORT.md`, `docs/ROADMAP_AND_SKILLS.md`.

Status de R1–R9 (fonte: `docs/EVOLUTION.md` seção 5, conferido no código):

| Item | Status em 2026-10-23 | Evidência |
| --- | --- | --- |
| R1 Pool de inferência desacoplado | pendente | inferência continua síncrona no `FramePipeline`; nenhum worker em `app/` |
| R2 Token de WebSocket fora da URL | implementado | primeira mensagem `{"type":"auth"}`; `?token=` recusado (`tests/test_security.py`, `tests/test_analyze_ws.py`) |
| R3 Persistência de agregados | pendente | sem banco nem exportação por sessão além do JSON |
| R4 Webhooks assinados | pendente | nenhum código |
| R5 Detector YuNet | implementado (opcional) | `EMOTION_FACE_DETECTOR=yunet`; falha a 45° (`docs/VALIDATION_REPORT.md`) |
| R6 Quadros binários no stream do servidor | pendente | `/ws/stream` ainda envia JPEG em base64 (`app/backend/services.py:307`, `app/frontend/app.js:329`) |
| R7 Consentimento e governança | parcial | checkbox de ciência/consentimento no painel (`app/frontend/index.html:34`); sem registro do aceite nem política de retenção |
| R8 CI ampliado | parcial | `ruff check`, `ruff format --check` e `mypy` no CI; faltam `pip-audit` e build Docker |
| R9 Multi-tenant | pendente (fora do escopo de 4–8 semanas) | — |

## 2. Estado atual (baseline)

Ambiente: Linux x86_64, Python 3.13.16, uv 0.11.32, commit `57b4dae` (2026-10-18), sem câmera, sem GPU.

| Verificação | Comando | Resultado | Data |
| --- | --- | --- | --- |
| Instalação | `uv venv -p python3.13 .venv && uv pip install -p .venv/bin/python -r requirements-test.in` | PASS — 82 pacotes, 21 s, `.venv` com 2,4 GB; `uv pip check`: compatíveis | 2026-10-23 |
| Lint | `ruff check .` · `ruff format --check .` | PASS — 0 problemas; 52 arquivos já formatados | 2026-10-23 |
| Typecheck | `python -m mypy` | PASS — 0 problemas em 4 arquivos (strict) | 2026-10-23 |
| Testes | `python -m pytest -q` | 88 passando / 0 falhando / 1 pulado (YuNet real, opt-in: exige `EMOTION_YUNET_MODEL` e `EMOTION_DEMO_VIDEO`); 6,7 s; nenhum peso baixado (`~/.deepface` inexistente após a execução) | 2026-10-23 |
| Cobertura | — | não verificado: `pytest-cov`/`coverage` não configurado no projeto | 2026-10-23 |
| Build | `docker compose up --build` | não verificado: daemon Docker indisponível no ambiente. Smoke substituto: `python run_api.py --port 8765` → `GET /api/v1/health` 200, `GET /` 200, `PATCH /api/v1/config {"video_source": ...}` sem token 403; importar `app.backend.api` não carrega `tensorflow`, `deepface` nem `cv2` | 2026-10-23 |
| Vulnerabilidades (audit) | `pip-audit --path .venv/lib/python3.13/site-packages` (pip-audit 2.10.1) | 37 vulnerabilidades conhecidas em 1 pacote: `pillow 12.3.0` (correção em 13.0.0, upgrade major). critical/high/moderate: não verificado (pip-audit não informa gravidade; consulta à OSV bloqueada no ambiente) | 2026-10-23 |
| CI no GitHub | `gh run list --limit 5` | workflow `Tests` (`.github/workflows/tests.yml`: jobs `lint` e `pytest` com mypy); últimas 5 execuções `success`, a mais recente em `main` 2026-10-18 | 2026-10-23 |

Observações:
- O modo câmera com webcam física, a inferência com o modelo real (DeepFace/TensorFlow), o YuNet real e o Docker não foram executados nesta baseline (sem câmera, download de pesos pesado, sem daemon). As medições com modelo real ficam as registradas em `docs/EVOLUTION.md` seção 8 e `docs/VALIDATION_REPORT.md` (p50 29,6 ms / p95 80,5 ms no modo câmera; Haar p50 8,19 ms vs. YuNet p50 3,53 ms), de 2026-10-15/17.
- Sem tokens, `PATCH /api/v1/config` aceita campos de análise (ex.: `emotion_interval`, resposta 200) e recusa caminhos (`video_source`, `yunet_model_path`). CORS padrão é lista vazia (`app/core/config.py:106`), então requisições de outra origem com preflight são bloqueadas pelo navegador; o processo escuta em `127.0.0.1` por padrão.
- `Dockerfile` e `run.bat` instalam `requirements.txt` (faixas `>=`), não os pins de `requirements.in`: imagem e ambiente Windows não são reproduzíveis em relação ao CI.
- Higiene ainda presente no código: comentário de rascunho em `.gitignore:13-16`; "Senior Experience" em `run.bat:6`; título de janela "Face Emotion Recognition - Senior Edition" em `app/core/processor.py:97`; rodapé "Autoria: Matheus Siqueira" com link de barra final em `app/frontend/index.html:167-170` e `README.md:301`; posicionamento de monitoramento e atendimento em `README.md:7,12,39`; emoji em `README.md:15`; KPI "Índice visual heurístico" calculado com pesos arbitrários (`app/core/metrics.py:9-17`) sem os pesos visíveis na UI.
- Documentos rastreados citam produtos externos usados no desenvolvimento e na validação (`docs/ROADMAP_AND_SKILLS.md:36-45`, `docs/EVOLUTION.md` seção 2, `docs/PHASE1_DIAGNOSTIC.md:138`), o que viola a regra de rastros da política de engenharia. `docs/EVOLUTION.md` não é alterado por este plano; a correção fica no backlog (B5) e depende do dono.
- Histórico com quatro identidades de autor (`hellomatheusdev@gmail.com` em 27 commits; `matheussiqueirahub@gmail.com` com três nomes em 8 commits).

## 3. Histórico e trajetória

35 commits de 2025-12-07 a 2026-10-18, em quatro fases:

1. **2025-12-07 — origem:** commit inicial, tradução do README e licença MIT.
2. **2026-01-22 a 2026-02-08 — refatoração modular:** pipeline com CLI e rastreamento, FastAPI e painel "EmotionAI"; um merge com mensagem padrão (`Merge branch 'main' of ...`) e mensagens fora de Conventional Commits.
3. **2026-10-11 a 2026-10-12 — retomada (PRs #2–#4):** correção do stream e da detecção, Docker, CI com pytest, uma única distribuição OpenCV, `httpx2` no cliente de testes.
4. **2026-10-15 a 2026-10-18 — endurecimento (PRs #5–#7, branch `feat/enterprise-hardening`):** `create_app()` com DI, contrato de erro, cabeçalhos de segurança, modo câmera do navegador, vídeo em loop, autenticação do WebSocket pela primeira mensagem, suavização exponencial por rosto rastreado, rastreador com casamento global por IoU, YuNet opcional, CI com ruff/mypy e pins exatos, remoção do `Face-emotions.mp4` do índice (o arquivo segue no histórico), leitura pública quando só o token admin está definido, ajustes de configuração para a Vercel.

Áreas de maior churn (últimos 50 commits): `README.md` (13), `requirements.txt` (8), `app/core/analyzer.py` (8), `app/frontend/app.js` (7), `main.py`, `app/core/config.py` e `app/backend/services.py` (6 cada). Padrões recorrentes: falhas de importação do DeepFace/TensorFlow (`fix: disable emotion analysis on any DeepFace import failure`), conflito de pacotes OpenCV e ajuste de versão de Python para wheels do TensorFlow (`fix: pin Vercel's Python to 3.13`). Débitos herdados: inferência síncrona (R1), stream em base64 (R6), dependências transitivas sem lock, Pillow vulnerável, README com posicionamento de monitoramento, resíduos textuais e documentação do ciclo dispersa em 7 arquivos de `docs/`. A versão continua `2.1.0` com uma seção `[Não publicado]` extensa no `CHANGELOG.md`.

## 4. Objetivos definidos (critério de parada)

Cada objetivo tem métrica, baseline e meta. Atingida a meta, o objetivo sai do ciclo ativo e passa a ser apenas monitorado (sem novas mudanças por ele).

| # | Eixo | Objetivo | Métrica | Baseline | Meta | Prioridade | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O1 | Funcionamento | Ambiente reproduzível igual ao CI em Linux, Windows e Docker | Instalações a partir de um lock com hashes que passam `pytest` (Linux, Windows, imagem Docker) | 0 de 3 (sem lock; Docker e `run.bat` usam faixas) | 3 de 3 | P0 | ativo |
| O2 | Cobertura de testes | Medir e manter a cobertura dos módulos de lógica | Cobertura de linhas em `app/core` (exceto `analyzer.py`/`processor.py`) e `app/backend` | não medida | ≥ 80%, com limite no CI | P2 | ativo |
| O3 | Segurança | Zerar vulnerabilidades conhecidas e manter a API fechada por padrão | `pip-audit` no ambiente instalado; testes de segurança | 37 vulnerabilidades (Pillow 12.3.0); `pip-audit` fora do CI | 0 vulnerabilidades conhecidas ou exceção documentada; `pip-audit` bloqueante no CI | P1 | ativo |
| O4 | Performance | Não degradar o FPS com vários rostos | FPS médio do stream do servidor com vídeo de 3 rostos (CPU) e banda por quadro | não medido com 3 rostos; base64 adiciona ~33% ao JPEG | FPS com 3 rostos ≥ 80% do FPS com 1 rosto; −25% de bytes por quadro no `/ws/stream` | P2 | ativo |
| O5 | Qualidade/Manutenibilidade | Repositório sem resíduos e com texto verdadeiro | Ocorrências de `git grep` para resíduos (`Senior`, rascunho do `.gitignore`, "Autoria:", posicionamento de monitoramento) e de termos de rastro | 9 locais com resíduo (seção 2) + 3 documentos com rastro | 0 e 0 | P1 | ativo |
| O6 | Acessibilidade/UX e conformidade | Uso responsável explícito e painel acessível | Violações axe serious/critical em 320/768/1280 px; aviso de limitação e consentimento registrado | não medido; consentimento só como checkbox | 0 violações; aceite registrado sem imagem (R7) e pesos do índice visíveis ou índice removido | P2 | ativo |

## 5. Guardrails

- Mudança só com objetivo associado (O#) ou item de backlog (B#); refatoração sem ganho mensurável é proibida.
- Nenhuma métrica da seção 2 pode piorar; se piorar, a mudança é revertida.
- Upgrade de versão major, mudança de arquitetura, remoção de funcionalidade e alteração de deploy exigem aprovação do dono (entram em "Decisões pendentes").
- No máximo uma feature nova por semana, de risco baixo, atrás de testes.
- Nenhum quadro, recorte de rosto ou embedding é gravado em disco, enviado a terceiros ou registrado em log; exportações contêm só contagens e séries agregadas.
- Testes não abrem câmera nem carregam DeepFace/TensorFlow; detector e analisador entram por *ports* com dublês; o CI não baixa pesos.
- Não trocar o modelo de expressão nem o detector padrão (`haar`) sem medição comparativa registrada; YuNet continua opcional e só por variável de ambiente.
- A API nunca aceita caminhos de arquivo pelo `PATCH /api/v1/config` (`video_source`, `cascade_path`, `yunet_model_path`).
- Sem reposicionar o produto para monitoramento de pessoas no trabalho, em atendimento, em sala de aula ou para decisões sobre indivíduos (AI Act, art. 5º, n.º 1, alínea f).
- `docs/EVOLUTION.md` (ADRs e histórico R1–R9) só muda com aprovação do dono; novos ADRs entram como seções novas, sem reescrever os aceitos.
- Nada de vídeo com rosto de terceiros no repositório; `Face-emotions.mp4` permanece fora do índice.

## 6. Backlog priorizado

- P0 = quebra funcionamento/segurança · P1 = risco alto · P2 = qualidade · P3 = melhoria

- [ ] **B1 (P0, O1)** Lock de dependências com hashes (`uv pip compile --universal --generate-hashes requirements.in -o requirements.lock` e equivalente para teste), usado por CI, `Dockerfile` e `run.bat` no lugar de `requirements.txt` — CI instala só do lock; `pytest` verde em Linux e Windows; `docker build` instala do lock; `git grep -n "requirements.txt" Dockerfile run.bat` vazio.
- [ ] **B2 (P1, O3)** Atualizar Pillow para 13.x (major, exige aprovação) após checar compatibilidade com DeepFace 0.0.102 — `pip-audit` sem vulnerabilidades conhecidas; testes de validação de JPEG (`tests/test_analyze_ws.py`) verdes; registro no CHANGELOG.
- [ ] **B3 (P1, O3)** `pip-audit` bloqueante no CI (R8, parte restante) sobre o ambiente instalado do lock — job falha com vulnerabilidade conhecida sem exceção documentada; primeira execução verde no PR.
- [ ] **B4 (P1, O5)** Remover resíduos: `.gitignore:13-16`, `run.bat:6`, `app/core/processor.py:97`, emoji em `README.md:15`; rodapé e README com a assinatura padrão "Desenvolvido por Matheus Siqueira" e links normalizados — `git grep -n -E "Senior|The user wants|Autoria:"` vazio.
- [ ] **B5 (P1, O5)** Retirar dos documentos rastreados as menções a produtos externos usados no desenvolvimento (`docs/ROADMAP_AND_SKILLS.md:36-45`, `docs/PHASE1_DIAGNOSTIC.md:138`, `docs/EVOLUTION.md` seção 2, esta última só com aprovação) e consolidar os 7 documentos do ciclo — busca pelo padrão de rastros da política retorna vazio; README aponta para no máximo 2 documentos de roteiro.
- [ ] **B6 (P1, O6)** README verdadeiro: substituir o posicionamento de `README.md:7,12,39` (monitoramento, atendimento, "segurança para ambientes de produção") por uso local de demonstração e pesquisa com consentimento, com seção "Limitações e privacidade" — nenhuma ocorrência de "atendimento"/"monitoramento" como caso de uso; texto revisado contra o código.
- [ ] **B7 (P2, O2)** Configurar `pytest-cov` (justificar a dependência de teste) com relatório no CI e `--cov-fail-under` igual à baseline medida, subindo até 80% — CI mostra cobertura e falha abaixo do limite.
- [ ] **B8 (P2, O6)** Índice heurístico honesto: exibir os pesos de `EMOTION_SENTIMENT_WEIGHTS` junto ao KPI ou trocar por "expressão predominante (moda)" — teste em `tests/test_metrics.py`; UI sem número sem explicação.
- [ ] **B9 (P2, O6)** R7 restante: registrar o aceite do modo câmera (data/hora e versão do texto, sem imagem) no lado do servidor ou na exportação de sessão — teste do contrato; nenhum dado biométrico no registro.
- [ ] **B10 (P2, O4)** R1: worker de inferência "recorte mais recente vence" (código de referência em `docs/EVOLUTION.md` 6.7), com medição antes/depois em vídeo de 3 rostos sem terceiros identificáveis — meta de O4 atingida; testes com analisador lento falso.
- [ ] **B11 (P2, O1)** Build Docker no CI (R8, parte restante) com `docker build` sem rodar o modelo — job verde; imagem instala do lock.
- [ ] **B12 (P2, O6)** E2E com Playwright (já em `requirements-dev.in`) apontando `EMOTION_VIDEO_SOURCE` para vídeo de teste, axe em 320/768/1280 px — 0 violações serious/critical; sem rolagem horizontal em 320 px. Execução local, fora do CI.
- [ ] **B13 (P3, O4)** R6: quadros binários no `/ws/stream` (JPEG binário + metadados JSON) — bytes por quadro −25% medidos; contrato documentado e testado.
- [ ] **B14 (P3, O5)** Versão e CHANGELOG: publicar `2.2.0` com a seção `[Não publicado]` atual após B1–B4 — `app.__version__`, `pyproject.toml` e CHANGELOG coerentes.

## 7. Radar técnico

Itens a pesquisar e reavaliar toda semana (novas versões, ferramentas, práticas). Cada entrada traz fonte e veredito.

| Data | Item | Fonte | Veredito (adotar / avaliar / descartar) | Motivo |
| --- | --- | --- | --- | --- |
| 2026-10-23 | Pillow 13.0.0 (correção das 37 vulnerabilidades apontadas em 12.3.0) | https://pillow.readthedocs.io/en/stable/releasenotes/ | adotar (B2, com aprovação de major) | Pillow decodifica JPEG enviado pelo navegador no modo câmera: é superfície de entrada externa |
| 2026-10-23 | `uv pip compile --universal --generate-hashes` | https://docs.astral.sh/uv/pip/compile/ | adotar (B1) | Lock único multiplataforma a partir dos `.in` existentes, sem migrar o projeto para outro fluxo |
| 2026-10-23 | DeepFace releases após 0.0.102 | https://github.com/serengil/deepface/releases | avaliar | Novas versões mudam dependências de TensorFlow e cache de pesos; atualizar só com smoke de modelo real (`scripts/verify_demo.py`) |
| 2026-10-23 | `actions/checkout` e `actions/setup-python` em versões com Node 24, fixadas por SHA | https://github.com/actions/checkout/releases | adotar | Workflow usa `@v4` por tag; fixar por SHA reduz risco de cadeia de suprimentos |
| 2026-10-23 | MediaPipe Face Detector para Web (`@mediapipe/tasks-vision`) | https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/web_js | avaliar (F2) | Base para inferência no navegador (opção D do ADR-002), que viabiliza demo estática sem enviar quadros |
| 2026-10-23 | OpenCV 5 | https://github.com/opencv/opencv/releases | descartar por ora | `requirements.in` registra que as wheels do OpenCV 5 não trazem os XML do Haar cascade usados por `app/core/detector.py` |

## 8. Ideias de features

| # | Feature | Valor para o usuário/portfólio | Risco | Status (proposta / aprovada / implementada / descartada) |
| --- | --- | --- | --- | --- |
| F1 | Exportação CSV da linha do tempo da sessão (segundo, rastro, expressão suavizada, confiança; `;` e vírgula decimal), só local | Entregável que abre direto em planilha; prepara R3 sem banco | Baixo | proposta |
| F2 | Inferência no navegador (MediaPipe para detecção + classificador leve em ONNX Runtime Web) | Demo publicável como site estático, com prova de que nenhum quadro sai do navegador | Alto (porte e licença do modelo, desempenho em celular) | proposta |
| F3 | Indicador de latência e FPS de inferência no painel do modo câmera | Torna visível o custo do servidor e o efeito do *stop-and-wait* | Baixo | proposta |
| F4 | Faixa de expressões da sessão (fita por segundo com expressão predominante e altura pela confiança) no lugar do donut | Leitura da sessão inteira sem métrica inventada | Médio (refazer gráficos) | proposta |
| F5 | Fallback para Haar quando o YuNet não encontra rosto | Cobre a falha do YuNet a 45° registrada na validação | Baixo, mas só com dados reais que justifiquem | proposta |

## 9. Decisões pendentes do dono

- Aprovar o upgrade major de Pillow 12.3.0 → 13.x (B2).
- Deploy: os commits de 2026-10-17 configuram a Vercel (`[tool.vercel]`, Python fixado em 3.13), mas a regra do repositório é "sem Vercel" e o modo vídeo do servidor não funciona remotamente. Manter, remover a configuração ou trocar por demo estática (F2)?
- Autorizar a alteração de `docs/EVOLUTION.md` para retirar a seção 2 (produtos externos citados) e consolidar os documentos do ciclo (B5).
- `Face-emotions.mp4` saiu do índice em `f87a572`, mas continua no histórico público. Se houver rosto de terceiros, reescrever o histórico? (Hipótese não verificada: vídeo do próprio dono.)
- Nome do produto: "EmotionAI" ou descritivo em PT-BR ("Expressões Faciais em Tempo Real")?
- Unificar a identidade de autor dos commits futuros em `Matheus Siqueira <hellomatheusdev@gmail.com>`?
- Manter o modo janela OpenCV (`main.py`) como legado documentado?

## 10. Registro semanal

| Data | Auditoria | Mudanças (commits) | Métricas | Próximo foco |
| --- | --- | --- | --- | --- |
| 2026-10-23 | Documento inicial e baseline (Python 3.13, sem câmera, sem Docker) | — | ver seção 2: lint/format/mypy PASS; 88 testes passando, 1 pulado; 37 vulnerabilidades (Pillow); cobertura e Docker não verificados | B1, B2, B4 |
