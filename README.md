# Face Emotion Recognition Realtime

**Desenvolvido por Matheus Siqueira**

## Visão Geral do Projeto

Este projeto implementa um sistema de reconhecimento de emoções faciais em tempo real utilizando Python. Ele utiliza técnicas de Visão Computacional e modelos de Deep Learning para capturar vídeo de uma webcam, detectar rostos e classificar expressões emocionais (como felicidade, tristeza, neutralidade e raiva) com alta precisão.

O sistema foi projetado para ser robusto e de fácil implantação, contando com configuração automática de ambiente e capacidade de gravação de vídeo.

## Principais Funcionalidades

- **Detecção Facial em Tempo Real**: Utiliza Haar Cascades do OpenCV para rastreamento facial eficiente.
- **Classificação de Emoções**: Integra a biblioteca DeepFace para analisar atributos faciais e prever emoções.
- **Visualização Ao Vivo**: Exibe caixas delimitadoras, rótulos de emoção e porcentagens de confiança sobre o vídeo.
- **Gravação de Sessão**: Grava automaticamente a sessão e a salva como `output_preview.avi`.
- **Tolerância a Falhas**: Inclui mecanismos de fallback para carregamento de modelos e recursos de detecção.

## Tecnologias Utilizadas

- **Linguagem**: Python 3.11
- **Visão Computacional**: OpenCV (cv2)
- **Deep Learning**: DeepFace, TensorFlow/Keras
- **Automação**: Scripts em lote (batch) para gerenciamento de ambiente

## Pré-requisitos

- Python 3.11 é recomendado para compatibilidade ideal com OpenCV e TensorFlow em sistemas Windows.
- Uma webcam funcional.

## Instalação e Configuração

### Configuração Automática (Recomendado)

1. Navegue até a pasta do projeto.
2. Clique duas vezes no script `run.bat`.
   - Este script criará automaticamente o ambiente virtual, instalará todas as dependências e iniciará a aplicação.

### Instalação Manual

Se preferir configurar o ambiente manualmente:

1. Crie um ambiente virtual usando Python 3.11:
   ```powershell
   py -3.11 -m venv .venv
   ```

2. Ative o ambiente virtual:
   ```powershell
   .\.venv\Scripts\Activate
   ```

3. Instale as dependências necessárias:
   ```powershell
   pip install -r requirements.txt
   ```

## Uso

Para executar a aplicação via terminal:

```powershell
python main.py
```

### Controles

- **Esc**: Pressione a tecla 'Esc' para fechar a janela da aplicação e salvar a gravação.

## Saída

A aplicação gera um arquivo de vídeo chamado `output_preview.avi` no diretório raiz do projeto, contendo a sessão gravada com todas as sobreposições visuais.
