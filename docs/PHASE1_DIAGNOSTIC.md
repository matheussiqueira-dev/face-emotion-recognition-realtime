# Diagnóstico inicial — Fase 1

Data: 16/10/2026, America/Fortaleza. Base inspecionada: commit `8e3f7ae`, branch `feat/enterprise-hardening`.

## Conclusão

O EmotionAI está funcional no ambiente local após duas correções de portabilidade no Windows. Foram verificados HTTP, detecção e classificação reais, transmissão WebSocket e análise de JPEG enviado pelo cliente. Isso não certifica prontidão para produção, escalabilidade, precisão científica ou funcionamento com uma webcam física.

A premissa de um repositório inteiramente abandonado não corresponde ao checkout atual: há restauração recente, testes e evolução arquitetural documentados em `docs/EVOLUTION.md`, de 15/10/2026. Essa base deve ser preservada e suas lacunas tratadas incrementalmente.

## Propósito e arquitetura existente

O sistema detecta rostos, associa detecções entre quadros por interseção de caixas (IoU) e estima uma categoria de expressão facial com DeepFace. Os resultados são inferências do modelo, não comprovação do estado emocional de uma pessoa. Não há banco de dados nem identidade facial persistente na implementação inspecionada.

Fluxos principais:

```text
Vídeo/câmera do servidor
  → VideoProcessor → FramePipeline → Haar Cascade → FaceTracker → EmotionAnalyzer
  → StreamService → JPEG/base64 + JSON por WebSocket → dashboard

JPEG enviado pelo navegador
  → WebSocket /api/v1/ws/analyze → validação de tamanho/formato/resolução
  → BrowserAnalysisSession → FramePipeline → JSON de caixas e expressões
```

| Camada | Implementação | Observação |
|---|---|---|
| Domínio/processamento | `app/core` | Configuração, modelos de dados, detector, tracker, pipeline, analisador e métricas |
| Entrega | `app/backend` | FastAPI, REST versionado, WebSocket, autorização, serialização e middlewares |
| Interface | `app/frontend` | HTML, CSS e JavaScript sem etapa de build; Chart.js e fontes externas |
| Composição | `create_app()` | Colaboradores injetáveis, `Depends` e ciclo de vida do serviço |
| Execução | `run_api.py`, `run.bat`, Docker/Compose | API local; wrapper `main.py` depende de OpenCV com GUI |
| Qualidade | `tests`, GitHub Actions | Pytest; CI em Linux/Python 3.13, sem lint/typecheck configurados |

Não há `package.json`, lockfile Python ou `pyproject.toml`. Não é necessário introduzir Node/React para restaurar a aplicação. A arquitetura já possui portas por `Protocol` e injeção; uma reescrita ampla não se justifica nesta etapa.

## Histórico e continuidade

- Dezembro/2025: aplicação inicial concentrada em `main.py`, README e licença MIT.
- Janeiro/2026: vídeo de demonstração, pipeline e rastreamento.
- Fevereiro/2026: modularização, FastAPI, painel e segurança.
- 11–12/10/2026: restauração do stream, Docker, CI, inicialização fora do event loop e distribuição única de OpenCV.
- 15/10/2026: fábrica de aplicação, endurecimento da API, câmera do navegador, loop de vídeo e validação de uploads.

O diretório estava sem alterações reportadas por `git status` no início. Este ciclo não criou commits, não fez push, merge ou deploy. A relação com o remoto foi apenas lida da configuração local; não foi feita uma consulta de PRs ou atualização de refs remotas.

## Ambiente e dependências

O executável global `python` reportou 3.12.10; o ambiente do projeto `.venv/Scripts/python.exe` reportou **3.13.14**. Todas as verificações do projeto usaram o segundo explicitamente.

| Dependência | Instalada | Resultado |
|---|---|---|
| FastAPI | 0.143.0 | HTTP e testes passaram; índice consultado oferece 0.143.3 |
| Uvicorn | 0.54.0 | Servidor real iniciado em loopback |
| DeepFace | 0.0.102 | Importação e inferência reais passaram |
| TensorFlow / tf-keras | 2.21.0 / 2.21.0 | Carregados no Windows; há aviso de depreciação de API interna do tf-keras |
| OpenCV contrib headless | 4.14.0.94 | Decodificação e detecção passaram após correção do carregamento do XML |
| NumPy / Pillow | 2.5.4 / 12.3.0 | Presentes no ambiente auditado |
| Pydantic | 2.14.0 | Contratos passaram; índice oferece 2.14.1 |
| Starlette | 1.7.0 | TestClient/HTTP passaram; índice oferece 1.8.0 |
| Pytest / httpx2 | 9.1.1 / 2.13.1 | Suíte executada com sucesso |

`pip check`: **No broken requirements found**.

`pip-audit`: **83 pacotes consultados, nenhuma vulnerabilidade conhecida reportada, nenhum pacote marcado como ignorado no JSON**. O auditor foi instalado em ambiente temporário separado, usando um snapshot exato das versões instaladas no projeto; não alterou suas dependências. A consulta não prova ausência de falhas no código nem cobre componentes nativos do sistema, scripts CDN, pesos de modelos ou futuras resoluções do manifesto.

Evidência local da auditoria: `C:\Users\mathe\AppData\Local\Temp\emotionai-phase1-audit-deacgo6e\audit.json`. Esse caminho é temporário e específico desta máquina.

`pip list --outdated` também listou atualizações para filelock, h5py, pluggy, pydantic_core, soupsieve, truststore e wrapt. Atualização disponível não equivale a vulnerabilidade. OpenCV 5 é uma mudança major fora do limite `<5` atual; deve passar por avaliação específica, pois o projeto depende de APIs e XMLs da série 4.

**Decisão:** preservar as versões funcionais nesta fase. O principal problema de manutenção é o manifesto com limites amplos e sem resolução travada. A próxima etapa deve gerar um conjunto reproduzível, separar dependências de execução/teste e validar instalações limpas antes de atualizar seletivamente. Não foi feita uma reinstalação integral do zero.

## Falhas reproduzidas e corrigidas

### Inicialização com saída redirecionada no Windows

`run_api.py` imprimia um emoji antes de iniciar o Uvicorn. Com stdout CP1252, a execução falhava com `UnicodeEncodeError` e código 1.

- Correção: banner ASCII, preservando argumentos e comportamento do servidor.
- Regressão: `tests/test_run_api.py` executa o launcher com um stream CP1252 estrito e verifica a chamada ao Uvicorn.
- Verificação operacional: launcher iniciado com saída redirecionada; dashboard, JavaScript, health, métricas, Swagger e OpenAPI responderam 200.

### XML do detector em caminho Unicode

O XML distribuído pelo OpenCV existia, mas `CascadeClassifier(str(path))` falhava no caminho deste projeto, que contém `Programação`.

- Correção: Python lê o XML como UTF-8; `FileStorage` em memória fornece seu nó ao classificador. O recurso é liberado em `finally`; não há cópia permanente, download alternativo ou troca do detector.
- Regressão: `tests/test_detector.py` copia o XML real para uma pasta com acentos e exercita carregamento e detecção. Falhou antes da correção e passou depois.
- Verificação operacional: pipeline classificou um rosto do vídeo fornecido usando o analisador real.

## Verificações e limites

| Verificação | Resultado nesta execução |
|---|---|
| Suíte original | 38 testes passaram |
| Suíte após correções | **40 testes passaram** |
| `git diff --check` | Sem erros |
| HTTP em processo real | `/`, `/static/app.js`, `/api/v1/health`, `/api/v1/metrics`, docs e OpenAPI: 200 |
| Vídeo fornecido | 1.210 quadros; primeiro quadro decodificado, resolução 634 × 508 |
| Inferência real direta | Expressão `neutral` retornada; não é validação de acurácia |
| WebSocket do servidor | Três quadros recebidos com imagem, caixa e classificação reais |
| WebSocket de análise | JPEG real enviado; resposta com caixa/classificação e `latency_ms` de 17,4 ms em uma amostra |
| Docker | CLI presente; daemon inacessível; build/container não validados |
| Webcam física e navegador | Não exercitados; protocolo da câmera foi testado com quadro do vídeo |
| Carga, concorrência e benchmark | Não executados; a amostra de latência não representa p50/p95 ou capacidade |
| Auditoria profunda de aplicação | Reservada à Fase 2; não há certificação de segurança |

O primeiro download dos pesos do modelo (aproximadamente 5,98 MB) foi realizado pelo DeepFace a partir de seu repositório de modelos no GitHub e ficou no cache padrão `C:\Users\mathe\.deepface\weights`, fora do Git. O download não usa API paga. A licença e a procedência de cada peso e do vídeo devem ser verificadas antes de redistribuição comercial; a licença MIT do código não é prova suficiente para todos os ativos.

Os logs do próprio DeepFace durante o primeiro download também usam Unicode. Para a prova de inferência e WebSockets, foi usado `-X utf8`; esse detalhe deve entrar no onboarding Windows. O warmup ainda usa o detector padrão do DeepFace, que emite avisos de caminho Unicode antes de prosseguir; alinhar o warmup ao `detector_backend="skip"` já usado na análise é um ajuste recomendado para a Fase 2.

Comando local para executar com suporte a esses logs:

```powershell
$env:EMOTION_VIDEO_SOURCE = "C:\caminho\para\video.mp4"  # vídeo local, fora do repositório
.\.venv\Scripts\python.exe -X utf8 run_api.py --host 127.0.0.1 --port 8000
```

Os servidores de teste deste ciclo usaram portas efêmeras e foram encerrados após a verificação. Nenhuma câmera física foi ativada.

## Prioridades para a Fase 2

As observações abaixo são pontos a investigar/refatorar, não vulnerabilidades exploradas ou mudanças já entregues.

1. **Reprodutibilidade e onboarding:** fixar dependências compatíveis, separar ferramentas de desenvolvimento, documentar Python/UTF-8, preparar cache do modelo e validar instalação limpa. Acrescentar smoke test real opt-in sem downloads no teste unitário padrão.
2. **Estado operacional verdadeiro:** `/health` hoje confirma o processo HTTP; não representa prontidão do detector/modelo/fonte. O analisador pode desabilitar-se após erro de importação e retornar `None`. Tornar esse estado visível na API e na interface.
3. **Ciclo de vida concorrente:** `StreamService` espera até três segundos pela thread e depois elimina referências mesmo se ela ainda estiver viva. Verificar encerramento e reinício com inferência lenta. O broadcast aguarda clientes sequencialmente; medir impacto de consumidor lento e definir timeout/isolamento.
4. **Segurança de exposição:** tokens ausentes liberam endpoints, CORS padrão é `*`, e WebSocket usa token na URL. Há proteção para troca da fonte sem auth e o Compose publica em loopback, mas uma exposição pública exige políticas explícitas, verificação de origem e revisão completa dos fluxos.
5. **Semântica de métricas e backoff:** `analyzed_faces` soma observações por quadro, inclusive expressões reaproveitadas, não pessoas únicas ou inferências novas. `last_analyzed` só avança quando há resultado, podendo repetir tentativas a cada quadro após falhas. Definir contratos e testes antes de alterar números.
6. **Demo resiliente:** hospedar Chart.js/fontes localmente, validar consentimento e mensagens, estados de erro, acessibilidade e mobile em navegador. Atualmente `new Chart(...)` depende de CDN disponível para a inicialização da UI.
7. **Escala com evidência:** manter um worker enquanto captura/configuração/métricas forem locais ao processo. Medir fila, latência e CPU antes de introduzir pool de inferência, serviços externos ou persistência.

As opções acima podem usar bibliotecas livres e execução local. Gratuidade de software não garante hospedagem ou capacidade computacional gratuitas; não foi habilitado nenhum serviço pago ou nova integração externa.

## Ferramentas e continuidade documental

Skills utilizadas neste ciclo: `repository-health-check` e `dependency-audit`. Ferramentas: Git, PowerShell, Python/venv/pip, Pytest, pip-audit isolado, urllib, cliente WebSocket e documentação oficial. Não houve delegação a subagentes.

Este é o relatório da Fase 1 solicitado antes da auditoria profunda. `PROJECT_OVERVIEW.md`, `ROADMAP_AND_SKILLS.md` e `MASTER_EVOLUTION_PLAN.md` permanecem como entregas da Fase 4; não foram antecipados como se as demais fases já tivessem ocorrido. `docs/EVOLUTION.md` foi preservado como registro anterior; suas medições históricas não foram reapresentadas como medições atuais.

Referências consultadas:

- [DeepFace: implementação e licença do framework](https://github.com/serengil/deepface).
- [TensorFlow: instalação e limites de GPU no Windows nativo](https://www.tensorflow.org/install/pip).
- [OpenCV: leitura de CascadeClassifier a partir de FileNode](https://docs.opencv.org/4.13.0/d1/de5/classcv_1_1CascadeClassifier.html).
- [pip-audit: escopo e operação da auditoria de dependências](https://github.com/pypa/pip-audit).
