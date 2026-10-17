# Pesquisa open-source e features

Data: 17/10/2026. Objetivo: encontrar abordagens verificáveis para os dois pontos fracos observados no EmotionAI: rótulos que oscilam entre quadros e a detecção limitada do Haar Cascade. Estrelas e datas foram consultadas via `gh api` nesta data. Nenhum repositório foi clonado e nenhum código de terceiros foi copiado.

## Repositórios analisados

| Repositório | Feature analisada | Como funciona | Licença | Inspeção | Aplicabilidade |
|---|---|---|---|---|---|
| [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) (`models/face_detection_yunet`) | Detector YuNet | `cv.FaceDetectorYN.create(model, "", input_size, score, nms, top_k)`; `setInputSize` a cada quadro; `detect()` devolve `None` ou linhas `x, y, w, h`, 5 pontos de referência e score | Repositório Apache-2.0; pasta do YuNet com LICENSE MIT | Código-fonte (`yunet.py`, `demo.py`) | **Implementado** como detector opcional |
| [serengil/deepface](https://github.com/serengil/deepface) | Análise de emoção e streaming | `demography.analyze` recorta, preenche para 224×224, converte para cinza 48×48, CNN de 7 classes; `detector_backend="skip"` usa a imagem inteira como rosto. `modules/streaming.py` congela um quadro a cada 5 com rosto e mantém o resultado por 5 s, **sem suavização** | MIT | Código-fonte | Confirma o padrão "recortar e depois `skip`" que o projeto já usa |
| [justinshenk/fer](https://github.com/justinshenk/fer) | Processamento de vídeo | Haar (padrão) ou MTCNN; `Video.analyze` usa `frequency` para analisar 1 a cada N quadros; sem rastreamento nem suavização | MIT | Código-fonte (`fer.py`, `classes.py`) | Baixa: parâmetros Haar semelhantes aos nossos |
| [oarriaga/face_classification](https://github.com/oarriaga/face_classification) | Estabilização de rótulo | Moda dos últimos 10 rótulos em **uma lista compartilhada por todos os rostos** do quadro | MIT | Código-fonte (`video_emotion_color_demo.py`) | Mostra a lacuna: sem rastreamento, rostos diferentes se misturam |
| [yodawu28/face-emotion-cnn](https://github.com/yodawu28/face-emotion-cnn) | EMA das pontuações | `ema = (1-a)*ema + a*p`, com `a` adaptativo pela margem entre os dois primeiros rótulos | MIT | Código-fonte (um arquivo) | Repositório pequeno (0 estrelas): só referência de padrão |
| [abewley/sort](https://github.com/abewley/sort) | Rastreamento | Filtro de Kalman por caixa + casamento húngaro por IoU | **GPL-3.0** | Código-fonte (`sort.py`) | Ideia útil; código **não** reutilizável sem tornar o projeto GPL |
| [tryolabs/norfair](https://github.com/tryolabs/norfair) | Rastreamento | — | BSD-3-Clause | Só metadados | Alternativa permissiva, não avaliada |

## O que foi decidido

### Implementado: suavização por rosto rastreado

- **Problema:** cada inferência é independente, então o rótulo de um mesmo rosto oscila. Nenhum dos projetos populares analisados combina suavização com rastreamento por rosto; o mais conhecido (oarriaga) mistura os rótulos de rostos diferentes.
- **Lógica:** média móvel exponencial do vetor de pontuações, guardada no `Track` e descartada com ele. `alpha` = peso da estimativa anterior. O limiar de confiança vale para a saída bruta. O rótulo bruto continua visível (`raw_dominant`).
- **Implementação própria**, inspirada no conceito comum de EMA; nenhum código copiado. Arquivos: `app/core/pipeline.py` (`FramePipeline._smooth`), `app/core/tracker.py`.
- **Teste:** unidade em `tests/test_pipeline.py`; medição real em `scripts/measure_smoothing.py`.
- **Não adotado:** o `alpha` adaptativo do yodawu28. Ele acrescenta parâmetros sem evidência de ganho no nosso caso. Fica como possível melhoria se a medição mostrar atraso excessivo em mudanças reais.

### Implementado: detector YuNet opcional

- **Problema:** Haar Cascade perde rostos inclinados ou parcialmente de perfil e gera falsos positivos.
- **Lógica:** `YuNetDetector` (`app/core/detector.py`) implementa a mesma porta `FaceDetectorPort`, redimensiona por `detect_scale`, ajusta `setInputSize` a cada quadro, reescala caixas e filtra por `min_face_size`. O modelo é lido da memória pelo overload `create(framework, bufferModel, ...)`, pela mesma razão do Haar (caminhos Unicode no Windows).
- **Modelo:** para OpenCV 4.x, `face_detection_yunet_2023mar.onnx` (232.589 bytes segundo o ponteiro LFS). O `2026may` é o novo padrão do zoo, mas o README o associa ao OpenCV 5.x; não foi avaliado.
- **Validação:** modelo baixado e conferido pelo SHA-256 do ponteiro LFS. Desempenho igual ao do Haar em rostos frontais do vídeo, com cerca de 2,3× menos tempo de detecção. Muito melhor até ~30° de inclinação e sem detecção a 45° (teste sintético). Detalhes em [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

### Implementado: casamento global no rastreador

Inspirado conceitualmente no casamento por IoU do SORT, sem Kalman e sem código GPL: pares rastro/detecção ordenados por IoU decrescente, guloso global. Para poucos rostos, o resultado coincide com o húngaro na maioria dos casos, sem nova dependência (scipy).

### Descartado ou adiado

| Ideia | Motivo |
|---|---|
| Copiar `sort.py` | GPL-3.0 |
| Kalman / húngaro completo | Benefício não medido para 1 a 4 rostos de webcam; adiciona dependência |
| Modelo de emoção ONNX do DeepFace | Não avaliado; mudaria o runtime de inferência |
| Alinhamento facial pelos 5 pontos do YuNet | Depende de validar o YuNet real primeiro (P3) |
| MTCNN / RetinaFace | Mais pesados que o YuNet, sem necessidade comprovada |

## Fontes oficiais

- OpenCV `FaceDetectorYN`: documentação da classe em docs.opencv.org (4.x) e assinatura conferida localmente com `cv2.FaceDetectorYN.create.__doc__` no OpenCV 4.14.0.
- DeepFace: código-fonte em `deepface/modules/demography.py`, `detection.py`, `streaming.py`.
