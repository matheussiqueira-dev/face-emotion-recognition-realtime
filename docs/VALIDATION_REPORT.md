# Relatório de validação

Data: 17/10/2026. Branch `feat/enterprise-hardening`, a partir do commit `1359b4a`, com as alterações deste ciclo **não commitadas** no momento da validação.

## Ambiente

| Item | Valor |
|---|---|
| SO | Windows 11 Home 10.0.26300 |
| Python do projeto | 3.13.14 (`.venv/Scripts/python.exe`) |
| OpenCV contrib headless | 4.14.0 (`cv2.FaceDetectorYN` disponível) |
| TensorFlow / DeepFace | 2.21.0 / 0.0.102 |
| Ferramentas | pytest 9.1.1, ruff 0.17.1, mypy 2.4.0, pip-audit 2.10.1 |

Nenhuma dependência foi instalada ou atualizada neste ciclo; o `.venv` existente foi usado.

## Baseline (antes das alterações)

| Verificação | Comando | Resultado |
|---|---|---|
| Testes | `python -m pytest -q` | 65 passaram |
| Lint | `ruff check .` | Sem problemas |
| Formatação | `ruff format --check .` | **Falhou**: 1 arquivo (`docs/EVOLUTION.md`, blocos de código) |
| Tipos | `mypy` (4 módulos de `core`) | Sem problemas |
| Dependências | `pip check` | Sem conflitos |
| Vulnerabilidades | `pip_audit -r requirements.txt` | Nenhuma conhecida |

## Final (depois das alterações)

| Verificação | Comando | Resultado | Status |
|---|---|---|---|
| Testes | `python -m pytest -q` | **81 passaram, 1 pulado** (YuNet real, opt-in; executado à parte, abaixo) | Validado |
| Lint | `ruff check .` | Sem problemas | Validado |
| Formatação | `ruff format --check .` | 45 arquivos formatados | Validado |
| Tipos | `mypy` | Sem problemas | Validado |
| Dependências | `pip check` | Sem conflitos | Validado |
| Vulnerabilidades | `pip_audit -r requirements.in` | Nenhuma conhecida | Validado (escopo: pacotes PyPI dos pins diretos) |
| Whitespace | `git diff --check` | Sem erros | Validado |
| Smoke com modelo real | `python -X utf8 scripts/verify_demo.py` | 3 quadros do stream com rastros e expressões; protocolo da câmera aprovado; latência de 29,5 ms em uma amostra; contrato de métricas aprovado | Validado |
| Suavização com modelo real | `python -X utf8 scripts/measure_smoothing.py 600 0.5` | 600 quadros, 1 rosto rastreado, 600 observações rotuladas; **9 trocas de rótulo bruto, 3 suavizado** | Validado (uma execução) |
| Painel no navegador | Navegador embutido em `http://127.0.0.1:8765` | Campo de suavização com rótulo acessível; carrega 0,5 da API; aplicar 0,7 → `GET /config` retorna 0,7 e mensagem de sucesso; sem erros no console | Validado |
| Mobile (375 px) | Carregar a 1280 px e reduzir para 375 px | Antes: largura da página 992 px (gráficos presos). Depois: 375 px, canvas 309 px = contêiner | Validado |
| CI no GitHub | `.github/workflows/tests.yml` | Não executado: nada foi enviado ao remoto | Não executado |
| YuNet com modelo real | `EMOTION_YUNET_MODEL=models/face_detection_yunet_2023mar.onnx pytest tests/test_yunet_detector.py` | Modelo `face_detection_yunet_2023mar.onnx` baixado de opencv/opencv_zoo; SHA-256 `8f2383e4…2552fa4` igual ao ponteiro LFS do repositório (232.589 bytes). 8 de 8 testes passaram, incluindo detecção real no vídeo | Validado |
| YuNet × Haar, vídeo completo | Script ad hoc, `detect_scale=0.5`, `min_size=30` | 1.210 quadros: ambos acham 1 rosto em 1.210/1.210, sem caixas extras. Tempo de detecção: Haar p50 8,19 ms / p95 10,32 ms; YuNet p50 3,53 ms / p95 4,91 ms | Validado (um vídeo, um rosto frontal) |
| YuNet × Haar, rotação sintética | 1 a cada 10 quadros (121), girados com `warpAffine` | Haar: 0° 121, 15° 102, 30° 1, −30° 11, 45° 27. YuNet: 0° 121, 15° 121, 30° 120, −30° 115, **45° 0** | Validado como teste sintético; sem rótulos verdadeiros, as 27 detecções do Haar a 45° não foram conferidas |
| Smoke com YuNet | `EMOTION_FACE_DETECTOR=yunet EMOTION_YUNET_MODEL=<caminho absoluto> python -X utf8 scripts/verify_demo.py` | Stream, protocolo da câmera e métricas aprovados; latência de 15,0 ms em uma amostra | Validado |
| Suavização com YuNet | `measure_smoothing.py 600 0.5` com YuNet | 600 observações de 1 rosto: 2 trocas brutas, 2 suavizadas | Validado (uma execução) |
| Docker | `docker compose up --build` | Não executado neste ciclo; o daemon estava inacessível no ciclo anterior | Não executado |
| Webcam física | Modo "Minha câmera" | Não exercitado; o protocolo foi validado com quadros do vídeo | Não executado |

## Testes adicionados

| Arquivo | O que cobre |
|---|---|
| `tests/test_pipeline.py` | Suavização combina pontuações, segue mudança sustentada, `0` = saída bruta, estado por rastro, inferência abaixo do limiar não altera o estado; rastreador mantém IDs de vizinhos próximos |
| `tests/test_config.py` | `emotion_smoothing` restrito a `[0, 1)` e finito |
| `tests/test_services.py` | Métricas contam todo rótulo exibido no overlay |
| `tests/test_api.py` | `PATCH emotion_smoothing` aceito e validado |
| `tests/test_analyze_ws.py` | `raw_dominant` no contrato do modo câmera |
| `tests/test_yunet_detector.py` | Escala de caixas, sem rostos, tamanho mínimo, validação de configuração, API recusa troca de detector, fábrica padrão Haar, teste real opt-in |

O teste do rastreador e o de consistência das métricas foram executados antes das correções e falharam, confirmando que reproduzem os defeitos.

## Limites das medições

- A medição de suavização vem de uma única execução, sobre um vídeo e um rosto. Ela mostra que o mecanismo reduz trocas nesse caso; não estima acurácia nem garante o mesmo efeito em outros vídeos.
- As latências de 29,5 ms (Haar) e 15,0 ms (YuNet) são amostras únicas, não p50/p95.
- Com YuNet, o modelo de emoção trocou de rótulo só 2 vezes em 600 quadros, contra 9 com Haar. Uma hipótese é que recortes mais estáveis reduzam a oscilação, mas são execuções únicas e não isolam essa causa. Nesse caso, a suavização não teve oscilação a remover.
- A primeira tentativa do smoke com YuNet falhou porque o caminho relativo `../models/...` foi passado a um servidor que roda na raiz do projeto. A validação da configuração recusou iniciar, como esperado; a segunda execução usou caminho absoluto.
- `pip-audit` não cobre bibliotecas nativas, scripts de terceiros servidos pelo próprio app, pesos de modelos nem o vídeo.

## Status final

**B — Modernizado parcialmente.** O núcleo foi validado com evidência (testes, lint, tipos, auditoria de dependências, smoke com o modelo real nos dois detectores e verificação no navegador). Ficam pendentes: CI rodando no GitHub, Docker e webcam física.
