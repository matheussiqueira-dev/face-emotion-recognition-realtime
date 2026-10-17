# Roadmap e ferramentas

Atualizado em 17/10/2026. Estados: **Concluído**, **Em andamento**, **Pendente**, **Bloqueado**, **Descartado**. Os itens R1–R9 seguem a numeração de [EVOLUTION.md](EVOLUTION.md).

## Priorização

Pontuação de 1 a 5. Pesos: valor para o usuário 25%, viabilidade 20%, diferenciação técnica 15%, demonstrabilidade 15%, manutenibilidade 10%, risco (invertido) 10%, custo operacional (invertido) 5%. Valor e viabilidade pesam mais porque o objetivo é software funcionando e validado, não volume de features.

| Item | Valor | Viab. | Difer. | Demo | Manut. | Risco↓ | Custo↓ | Total | Prioridade | Estado |
|---|---|---|---|---|---|---|---|---|---|---|
| CI com lint/tipos e pins exatos | 3 | 5 | 2 | 2 | 5 | 5 | 5 | 3,8 | P1 | Concluído |
| Casamento global no rastreador | 4 | 5 | 3 | 3 | 4 | 5 | 5 | 4,2 | P1 | Concluído |
| Gráficos responsivos ao redimensionar | 3 | 5 | 1 | 3 | 5 | 5 | 5 | 3,7 | P1 | Concluído |
| Suavização por rosto rastreado | 5 | 5 | 4 | 5 | 4 | 4 | 5 | 4,7 | P2 | Concluído |
| Detector YuNet opcional | 4 | 4 | 4 | 4 | 4 | 3 | 5 | 4,0 | P2 | Concluído (validado no vídeo de demonstração; falha a 45°) |
| Lock completo de dependências (Linux + Windows) | 3 | 3 | 1 | 1 | 5 | 4 | 5 | 2,9 | P1 | Pendente |
| R1 Pool de inferência desacoplado | 4 | 3 | 3 | 3 | 3 | 2 | 4 | 3,2 | P2 | Pendente |
| R3 Linha do tempo por sessão / relatório CSV | 4 | 3 | 4 | 4 | 3 | 3 | 4 | 3,6 | P2 | Pendente |
| Alinhamento facial pelos pontos do YuNet | 3 | 3 | 4 | 3 | 3 | 2 | 5 | 3,1 | P3 | Pendente (depende do YuNet validado) |
| `alpha` adaptativo na suavização | 2 | 4 | 3 | 2 | 3 | 3 | 5 | 2,9 | P3 | Pendente (só com evidência de atraso) |
| Kalman + húngaro (SORT) | 2 | 3 | 3 | 2 | 2 | 2 | 4 | 2,4 | P3 | Descartado por ora (benefício não medido; código de referência GPL) |

## Próximas tarefas

1. **Concluído:** YuNet validado com o modelo real (ver VALIDATION_REPORT). Próximo passo opcional: avaliar a rotação em várias escalas, ou um fallback para Haar quando o YuNet não detectar nada, só se houver dados reais com cabeças muito inclinadas.
2. **Rodar o CI no GitHub.** Critério: jobs `lint` e `pytest` verdes no PR. Depende de push autorizado.
3. **Lock de dependências.** Gerar com `pip-compile` em Linux (CI) e conferir no Windows; aceitar só se `pytest` passar em ambos.
4. **R3: relatório por sessão.** Exportar CSV com linha do tempo de rótulos suavizados por rastro.
5. **R1: pool de inferência**, com medição de FPS antes e depois em vídeo com vários rostos.

## Riscos

- O padrão `emotion_smoothing = 0.5` muda o que clientes recebem em `dominant`/`scores`. Mitigação: `raw_dominant` no contrato, CHANGELOG e `EMOTION_SMOOTHING=0` para o comportamento anterior.
- O modelo `2026may` do YuNet é o novo padrão do zoo, mas foi documentado para OpenCV 5; o projeto está em OpenCV 4 e o README recomenda o `2023mar`.

## Ferramentas e skills usadas neste ciclo

| Ferramenta/Skill | Objetivo | Fase | Custo | Disponibilidade |
|---|---|---|---|---|
| Git | Estado, histórico, diff | Todas | Gratuito | Verificada |
| `.venv` Python 3.13 + pip | Execução e instalação existente | Baseline/validação | Gratuito | Verificada |
| pytest, ruff, mypy | Testes, lint, formatação, tipos | Baseline/validação | Gratuito | Verificada |
| pip-audit | Vulnerabilidades conhecidas | Baseline/validação | Gratuito (consulta OSV/PyPI) | Verificada |
| `gh` CLI (subagente de pesquisa) | Leitura de código e licenças de referências | Pesquisa | Gratuito | Verificada |
| Navegador embutido do Claude | Verificação visual, acessibilidade, mobile | UI | Gratuito | Verificada |
| Scripts `verify_demo.py` / `measure_smoothing.py` | Smoke e medição com o modelo real | Validação | Gratuito (pesos em cache local) | Verificada |
| Docker | Build do container | Validação | Gratuito | Não usado neste ciclo |
| Playwright | Testes E2E | — | Gratuito | Listado em `requirements-dev.in`; não usado neste ciclo |
