const emotionLabels = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"];
const emotionColors = {
    happy: "#f59e0b",
    sad: "#0ea5e9",
    angry: "#ef4444",
    surprise: "#f97316",
    neutral: "#64748b",
    fear: "#7c3aed",
    disgust: "#15803d",
};
const emotionWeights = {
    happy: 1.0,
    surprise: 0.5,
    neutral: 0.0,
    sad: -0.5,
    fear: -0.7,
    angry: -0.8,
    disgust: -1.0,
};

const elements = {
    canvas: document.getElementById("video-canvas"),
    statusBadge: document.getElementById("status-badge"),
    trackCount: document.getElementById("track-count"),
    fpsValue: document.getElementById("fps-value"),
    avgFpsValue: document.getElementById("avg-fps-value"),
    sentimentValue: document.getElementById("sentiment-value"),
    toggleStream: document.getElementById("toggle-stream"),
    noSignal: document.getElementById("no-signal"),
    apiToken: document.getElementById("api-token"),
    settingsForm: document.getElementById("settings-form"),
    emotionInterval: document.getElementById("emotion-interval"),
    minScore: document.getElementById("min-score"),
    maxFps: document.getElementById("max-fps"),
    jpegQuality: document.getElementById("jpeg-quality"),
    feedback: document.getElementById("config-feedback"),
    refreshMetrics: document.getElementById("refresh-metrics"),
    exportMetrics: document.getElementById("export-metrics"),
    captureFrame: document.getElementById("capture-frame"),
    eventLog: document.getElementById("event-log"),
    cameraVideo: document.getElementById("camera-video"),
    sourceModeInputs: document.querySelectorAll('input[name="source-mode"]'),
    noSignalTitle: document.getElementById("no-signal-title"),
    noSignalHint: document.getElementById("no-signal-hint"),
    privacyNote: document.getElementById("privacy-note"),
};

// Frames uploaded in camera mode are downscaled: enough for face detection, far less bandwidth and CPU.
const CAMERA_MAX_WIDTH = 640;
const CAMERA_JPEG_QUALITY = 0.7;
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const WS_CLOSE_MESSAGES = {
    1011: "O servidor falhou ao iniciar a análise.",
    1013: "Limite de sessões de câmera simultâneas atingido. Tente novamente em instantes.",
    4401: "Token ausente ou inválido.",
    4403: "Este token não tem permissão para esta operação.",
};

const SOURCE_COPY = {
    server: {
        title: "Transmissão indisponível",
        hint: "Inicie o stream para visualizar o processamento em tempo real.",
    },
    camera: {
        title: "Câmera desligada",
        hint: "Inicie a transmissão e permita o acesso à câmera para analisar seu próprio rosto.",
    },
};

const ctx = elements.canvas.getContext("2d");
const state = {
    ws: null,
    mode: "server",
    isStreaming: false,
    heartbeat: null,
    metricsTimer: null,
    frameCounter: 0,
    emotionCounts: Object.fromEntries(emotionLabels.map((item) => [item, 0])),
    sentimentSeries: Array(80).fill(0),
    eventBuffer: [],
    camera: {
        stream: null,
        tracks: [],
        sendScale: 1,
        analysisTimes: [],
        fpsSamples: [],
        rafId: null,
    },
};

const distributionChart = new Chart(document.getElementById("distributionChart"), {
    type: "doughnut",
    data: {
        labels: emotionLabels.map((value) => value.charAt(0).toUpperCase() + value.slice(1)),
        datasets: [
            {
                data: emotionLabels.map(() => 0),
                backgroundColor: emotionLabels.map((emotion) => emotionColors[emotion]),
                borderWidth: 0,
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "70%",
        plugins: {
            legend: {
                position: "bottom",
                labels: {
                    boxWidth: 10,
                    color: "#334155",
                    font: { family: "IBM Plex Sans", size: 11 },
                },
            },
        },
    },
});

const timelineChart = new Chart(document.getElementById("timelineChart"), {
    type: "line",
    data: {
        labels: Array(80).fill(""),
        datasets: [
            {
                label: "Sentimento",
                data: state.sentimentSeries,
                borderColor: "#0f766e",
                backgroundColor: "rgba(15, 118, 110, 0.12)",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.35,
                fill: true,
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
            y: {
                min: -1,
                max: 1,
                ticks: {
                    callback: (value) => Number(value).toFixed(1),
                    color: "#64748b",
                },
                grid: { color: "rgba(100, 116, 139, 0.2)" },
            },
            x: { display: false },
        },
    },
});

function buildAuthHeaders() {
    const token = elements.apiToken.value.trim();
    const headers = { "Content-Type": "application/json" };
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }
    return headers;
}

// Browsers cannot set headers on websockets, so the token travels as a query parameter.
function buildWsUrl(path) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const token = elements.apiToken.value.trim();
    const baseUrl = `${protocol}//${window.location.host}${path}`;
    if (!token) {
        return baseUrl;
    }
    return `${baseUrl}?token=${encodeURIComponent(token)}`;
}

async function readErrorMessage(response, fallback) {
    try {
        const body = await response.json();
        return body?.error?.message || fallback;
    } catch {
        return fallback;
    }
}

function setStatus(mode, text) {
    elements.statusBadge.classList.remove("status-idle", "status-live", "status-error");
    elements.statusBadge.classList.add(`status-${mode}`);
    elements.statusBadge.textContent = text;
}

function setFeedback(message, type = "info") {
    elements.feedback.textContent = message;
    if (type === "error") {
        elements.feedback.style.color = "#b91c1c";
    } else if (type === "success") {
        elements.feedback.style.color = "#15803d";
    } else {
        elements.feedback.style.color = "#64748b";
    }
}

function pushEvent(message) {
    const timestamp = new Date().toLocaleTimeString("pt-BR", { hour12: false });
    state.eventBuffer.unshift(`${timestamp} - ${message}`);
    state.eventBuffer = state.eventBuffer.slice(0, 14);

    elements.eventLog.innerHTML = "";
    state.eventBuffer.forEach((eventLine) => {
        const item = document.createElement("li");
        item.textContent = eventLine;
        elements.eventLog.appendChild(item);
    });
}

function startHeartbeat() {
    clearInterval(state.heartbeat);
    state.heartbeat = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: "ping" }));
        }
    }, 20000);
}

function stopHeartbeat() {
    clearInterval(state.heartbeat);
    state.heartbeat = null;
}

function updateButtons() {
    elements.toggleStream.textContent = state.isStreaming ? "Encerrar Transmissão" : "Iniciar Transmissão";
    elements.toggleStream.classList.toggle("btn-secondary", state.isStreaming);
    elements.toggleStream.classList.toggle("btn-primary", !state.isStreaming);
}

function drawDetections(tracks) {
    tracks.forEach((track) => {
        const { x, y, w, h } = track.box;
        const emotion = track.emotion ? String(track.emotion.dominant).toLowerCase() : null;
        const color = emotion && emotionColors[emotion] ? emotionColors[emotion] : "#14b8a6";

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        if (!track.emotion) {
            return;
        }

        const confidence = Math.round(Math.max(...Object.values(track.emotion.scores || {}), 0));
        const label = `${track.emotion.dominant.toUpperCase()} ${confidence}%`;

        ctx.font = "600 13px 'IBM Plex Sans'";
        const labelPadding = 12;
        const labelHeight = 26;
        const labelWidth = ctx.measureText(label).width + labelPadding * 2;
        const labelY = Math.max(0, y - labelHeight - 6);

        ctx.fillStyle = color;
        ctx.fillRect(x, labelY, labelWidth, labelHeight);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + labelPadding, labelY + 17);
    });
}

function renderFrame(frameBase64, tracks) {
    const image = new Image();
    image.onload = () => {
        if (state.mode !== "server") {
            return;
        }
        elements.canvas.width = image.width;
        elements.canvas.height = image.height;
        ctx.drawImage(image, 0, 0, image.width, image.height);
        drawDetections(tracks);
    };
    image.src = `data:image/jpeg;base64,${frameBase64}`;
}

function updateChartsFromTracks(tracks) {
    if (!tracks.length) {
        return;
    }

    let sentimentAccumulator = 0;
    let withEmotion = 0;

    tracks.forEach((track) => {
        if (!track.emotion) {
            return;
        }
        const dominant = String(track.emotion.dominant).toLowerCase();
        if (!state.emotionCounts[dominant] && state.emotionCounts[dominant] !== 0) {
            state.emotionCounts[dominant] = 0;
        }
        state.emotionCounts[dominant] += 1;
        sentimentAccumulator += emotionWeights[dominant] || 0;
        withEmotion += 1;
    });

    if (withEmotion === 0) {
        return;
    }

    const sentiment = sentimentAccumulator / withEmotion;
    state.sentimentSeries.push(sentiment);
    state.sentimentSeries.shift();
    timelineChart.data.datasets[0].data = state.sentimentSeries;
    timelineChart.update("none");

    const totalCounts = emotionLabels.reduce((sum, emotion) => sum + (state.emotionCounts[emotion] || 0), 0) || 1;
    distributionChart.data.datasets[0].data = emotionLabels.map(
        (emotion) => Number((((state.emotionCounts[emotion] || 0) / totalCounts) * 100).toFixed(2)),
    );
    distributionChart.update("none");

    elements.sentimentValue.textContent = sentiment.toFixed(2);
}

function handleFramePayload(payload) {
    // Frames still in flight after switching to camera mode are dropped.
    if (state.mode !== "server" || !payload || !payload.frame || !payload.data) {
        return;
    }
    const tracks = payload.data.tracks || [];

    state.frameCounter += 1;
    renderFrame(payload.frame, tracks);
    elements.noSignal.style.display = "none";
    elements.trackCount.textContent = String(tracks.length);
    elements.fpsValue.textContent = Number(payload.data.fps || 0).toFixed(1);
    updateChartsFromTracks(tracks);

    if (state.frameCounter % 30 === 0 && tracks.length > 0) {
        const withEmotion = tracks.filter((track) => track.emotion);
        if (withEmotion.length) {
            const highlights = withEmotion.slice(0, 2).map((track) => `${track.emotion.dominant} (ID ${track.id})`).join(", ");
            pushEvent(`Emoções detectadas: ${highlights}`);
        }
    }
}

function resetSessionCharts() {
    state.frameCounter = 0;
    state.emotionCounts = Object.fromEntries(emotionLabels.map((item) => [item, 0]));
    state.sentimentSeries = Array(80).fill(0);
    timelineChart.data.datasets[0].data = state.sentimentSeries;
    timelineChart.update("none");
    distributionChart.data.datasets[0].data = emotionLabels.map(() => 0);
    distributionChart.update("none");
}

function handleStreamClosed(event) {
    // A late close from a previous connection must not tear down the one that replaced it.
    if (state.ws && event.target !== state.ws) {
        return;
    }
    state.ws = null;
    state.isStreaming = false;
    updateButtons();
    stopHeartbeat();
    stopCameraCapture();
    elements.noSignal.style.display = "flex";
    setStatus("idle", "Offline");

    const reason = WS_CLOSE_MESSAGES[event.code];
    if (reason) {
        setStatus("error", "Conexão recusada");
        setFeedback(reason, "error");
        pushEvent(reason);
    } else {
        pushEvent("Transmissão encerrada.");
    }
}

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return;
    }

    resetSessionCharts();
    const ws = new WebSocket(buildWsUrl("/api/v1/ws/stream"));
    state.ws = ws;

    ws.onopen = () => {
        state.isStreaming = true;
        updateButtons();
        setStatus("live", "Online");
        pushEvent("Transmissão iniciada.");
        startHeartbeat();
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "pong") {
            return;
        }
        handleFramePayload(payload);
    };

    ws.onerror = () => {
        setStatus("error", "Erro de conexão");
        setFeedback("Falha de comunicação WebSocket. Verifique token e servidor.", "error");
    };

    ws.onclose = handleStreamClosed;
}

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        setFeedback("Este navegador não permite acesso à câmera (é preciso HTTPS ou localhost).", "error");
        return;
    }

    let stream;
    // Block a second click while the permission prompt is open (it would start a second capture).
    elements.toggleStream.disabled = true;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false,
        });
    } catch (error) {
        const message = error.name === "NotAllowedError"
            ? "Permissão de câmera negada pelo navegador."
            : `Câmera indisponível (${error.name}).`;
        setStatus("error", "Câmera indisponível");
        setFeedback(message, "error");
        pushEvent(message);
        return;
    } finally {
        elements.toggleStream.disabled = false;
    }

    state.camera.stream = stream;
    elements.cameraVideo.srcObject = stream;
    await elements.cameraVideo.play();
    resetSessionCharts();
    state.camera.analysisTimes = [];
    state.camera.fpsSamples = [];

    const ws = new WebSocket(buildWsUrl("/api/v1/ws/analyze"));
    state.ws = ws;
    setStatus("idle", "Carregando modelo…");

    ws.onopen = () => {
        state.isStreaming = true;
        updateButtons();
        startHeartbeat();
        pushEvent("Câmera conectada. Preparando o modelo de emoções…");
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "ready") {
            setStatus("live", "Online · câmera");
            elements.noSignal.style.display = "none";
            renderCameraLoop();
            sendCameraFrame();
        } else if (payload.type === "analysis") {
            handleCameraAnalysis(payload);
            sendCameraFrame();
        } else if (payload.type === "error") {
            pushEvent(`Quadro rejeitado: ${payload.message}`);
            sendCameraFrame();
        }
    };

    ws.onerror = () => {
        setStatus("error", "Erro de conexão");
        setFeedback("Falha de comunicação WebSocket. Verifique token e servidor.", "error");
    };

    ws.onclose = handleStreamClosed;
}

// Stop-and-wait: one frame in flight at a time, so a slower server lowers the frame rate instead of queueing.
function sendCameraFrame() {
    const ws = state.ws;
    const video = elements.cameraVideo;
    if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) {
        return;
    }

    const scale = Math.min(1, CAMERA_MAX_WIDTH / video.videoWidth);
    captureCanvas.width = Math.round(video.videoWidth * scale);
    captureCanvas.height = Math.round(video.videoHeight * scale);
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    state.camera.sendScale = scale;

    captureCanvas.toBlob(async (blob) => {
        if (!blob || ws.readyState !== WebSocket.OPEN) {
            return;
        }
        ws.send(await blob.arrayBuffer());
    }, "image/jpeg", CAMERA_JPEG_QUALITY);
}

function handleCameraAnalysis(payload) {
    const scale = state.camera.sendScale || 1;
    const tracks = (payload.tracks || []).map((track) => ({
        ...track,
        box: {
            x: track.box.x / scale,
            y: track.box.y / scale,
            w: track.box.w / scale,
            h: track.box.h / scale,
        },
    }));
    state.camera.tracks = tracks;

    const now = performance.now();
    const times = state.camera.analysisTimes;
    times.push(now);
    while (times.length && times[0] < now - 1000) {
        times.shift();
    }
    const fps = times.length;
    state.camera.fpsSamples.push(fps);
    state.camera.fpsSamples = state.camera.fpsSamples.slice(-120);
    const avgFps = state.camera.fpsSamples.reduce((sum, value) => sum + value, 0) / state.camera.fpsSamples.length;

    state.frameCounter += 1;
    elements.trackCount.textContent = String(tracks.length);
    elements.fpsValue.textContent = fps.toFixed(1);
    elements.avgFpsValue.textContent = avgFps.toFixed(1);
    updateChartsFromTracks(tracks);

    if (state.frameCounter % 30 === 0) {
        const withEmotion = tracks.filter((track) => track.emotion);
        if (withEmotion.length) {
            const highlights = withEmotion.slice(0, 2).map((track) => `${track.emotion.dominant} (ID ${track.id})`).join(", ");
            pushEvent(`Emoções detectadas: ${highlights} · latência ${payload.latency_ms} ms`);
        }
    }
}

// Draw the live camera at display rate and overlay the latest detections (which arrive at inference rate).
function renderCameraLoop() {
    const video = elements.cameraVideo;
    if (!state.camera.stream) {
        return;
    }
    if (video.videoWidth) {
        if (elements.canvas.width !== video.videoWidth || elements.canvas.height !== video.videoHeight) {
            elements.canvas.width = video.videoWidth;
            elements.canvas.height = video.videoHeight;
        }
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        drawDetections(state.camera.tracks);
    }
    state.camera.rafId = requestAnimationFrame(renderCameraLoop);
}

function stopCameraCapture() {
    if (state.camera.rafId) {
        cancelAnimationFrame(state.camera.rafId);
        state.camera.rafId = null;
    }
    if (state.camera.stream) {
        state.camera.stream.getTracks().forEach((track) => track.stop());
        state.camera.stream = null;
    }
    elements.cameraVideo.srcObject = null;
    state.camera.tracks = [];
}

function resetLiveView() {
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    elements.trackCount.textContent = "0";
    elements.fpsValue.textContent = "0.0";
    elements.avgFpsValue.textContent = "0.0";
    elements.sentimentValue.textContent = "0.00";
    resetSessionCharts();
}

function setSourceMode(mode) {
    if (state.isStreaming || state.ws) {
        closeWebSocket();
    }
    // Do not leave the previous source's last frame and figures on screen.
    resetLiveView();
    state.mode = mode;
    const copy = SOURCE_COPY[mode];
    elements.noSignalTitle.textContent = copy.title;
    elements.noSignalHint.textContent = copy.hint;
    elements.privacyNote.hidden = mode !== "camera";
    pushEvent(mode === "camera" ? "Fonte alterada para a câmera do navegador." : "Fonte alterada para o vídeo do servidor.");
}

function closeWebSocket() {
    if (state.ws) {
        state.ws.close(1000, "Client requested stop");
    }
}

async function fetchConfig() {
    try {
        const response = await fetch("/api/v1/config", { headers: buildAuthHeaders() });
        if (!response.ok) {
            if (response.status === 401 || response.status === 403) {
                setFeedback("Configuração protegida: informe token para carregar parâmetros.", "error");
                return;
            }
            throw new Error(await readErrorMessage(response, `Falha ao carregar configuração (${response.status})`));
        }
        const payload = await response.json();
        const config = payload.config || {};
        elements.emotionInterval.value = config.emotion_interval ?? 0.5;
        elements.minScore.value = config.min_emotion_score ?? 0;
        elements.maxFps.value = config.max_fps ?? 30;
        elements.jpegQuality.value = config.jpeg_quality ?? 80;
        setFeedback("Configuração carregada.", "success");
    } catch (error) {
        setFeedback(error.message, "error");
    }
}

async function applyConfig(event) {
    event.preventDefault();

    const payload = {
        emotion_interval: Number(elements.emotionInterval.value),
        min_emotion_score: Number(elements.minScore.value),
        max_fps: Number(elements.maxFps.value),
        jpeg_quality: Number(elements.jpegQuality.value),
    };

    try {
        const response = await fetch("/api/v1/config", {
            method: "PATCH",
            headers: buildAuthHeaders(),
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            throw new Error(await readErrorMessage(response, `Não foi possível atualizar a configuração (${response.status})`));
        }
        const data = await response.json();

        const fields = data.updated_fields?.length ? data.updated_fields.join(", ") : "nenhum campo";
        setFeedback(`Configuração aplicada (${fields}).`, "success");
        pushEvent(`Configuração atualizada: ${fields}.`);
    } catch (error) {
        setFeedback(error.message, "error");
    }
}

async function fetchMetrics() {
    try {
        // Server metrics describe the server-side stream; camera mode computes its own figures client-side.
        if (state.mode !== "server") {
            return;
        }
        const response = await fetch("/api/v1/metrics", { headers: buildAuthHeaders() });
        if (!response.ok) {
            if (response.status === 401 || response.status === 403) {
                return;
            }
            throw new Error(await readErrorMessage(response, `Métricas indisponíveis (${response.status})`));
        }
        const metrics = await response.json();
        elements.avgFpsValue.textContent = Number(metrics.average_fps || 0).toFixed(1);

        if (Array.isArray(metrics.sentiment_history) && metrics.sentiment_history.length) {
            const lastSentiment = Number(metrics.sentiment_history[metrics.sentiment_history.length - 1] || 0);
            elements.sentimentValue.textContent = lastSentiment.toFixed(2);

            const normalized = metrics.sentiment_history.slice(-80);
            while (normalized.length < 80) {
                normalized.unshift(0);
            }
            state.sentimentSeries = normalized;
            timelineChart.data.datasets[0].data = state.sentimentSeries;
            timelineChart.update("none");
        }

        if (metrics.dominant_distribution) {
            const percentages = emotionLabels.map((emotion) => Number(metrics.dominant_distribution[emotion] || 0));
            distributionChart.data.datasets[0].data = percentages;
            distributionChart.update("none");
        }
    } catch (error) {
        setFeedback(error.message, "error");
    }
}

async function exportMetrics() {
    try {
        const response = await fetch("/api/v1/metrics", { headers: buildAuthHeaders() });
        if (!response.ok) {
            throw new Error(await readErrorMessage(response, `Falha ao exportar métricas (${response.status})`));
        }
        const metrics = await response.json();
        const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        link.href = url;
        link.download = `emotion-session-${timestamp}.json`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        pushEvent("Exportação de sessão concluída.");
    } catch (error) {
        setFeedback(error.message, "error");
    }
}

function captureCurrentFrame() {
    if (!elements.canvas.width || !elements.canvas.height) {
        setFeedback("Nenhum frame disponível para captura.", "error");
        return;
    }
    elements.canvas.toBlob((blob) => {
        if (!blob) {
            setFeedback("Não foi possível capturar o quadro atual.", "error");
            return;
        }
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `emotion-frame-${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        pushEvent("Quadro capturado com sucesso.");
    }, "image/png");
}

function toggleStream() {
    if (state.isStreaming || state.ws) {
        closeWebSocket();
        return;
    }
    if (state.mode === "camera") {
        startCamera();
    } else {
        connectWebSocket();
    }
}

function setupEvents() {
    elements.toggleStream.addEventListener("click", toggleStream);
    elements.settingsForm.addEventListener("submit", applyConfig);
    elements.refreshMetrics.addEventListener("click", fetchMetrics);
    elements.exportMetrics.addEventListener("click", exportMetrics);
    elements.captureFrame.addEventListener("click", captureCurrentFrame);
    elements.apiToken.addEventListener("change", fetchConfig);
    elements.sourceModeInputs.forEach((input) => {
        input.addEventListener("change", () => setSourceMode(input.value));
    });
    window.addEventListener("beforeunload", closeWebSocket);
}

function startMetricsPolling() {
    clearInterval(state.metricsTimer);
    state.metricsTimer = setInterval(() => {
        fetchMetrics();
    }, 8000);
}

function bootstrap() {
    setupEvents();
    fetchConfig();
    fetchMetrics();
    startMetricsPolling();
    updateButtons();
}

bootstrap();
