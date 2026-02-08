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
};

const ctx = elements.canvas.getContext("2d");
const state = {
    ws: null,
    isStreaming: false,
    heartbeat: null,
    metricsTimer: null,
    frameCounter: 0,
    emotionCounts: Object.fromEntries(emotionLabels.map((item) => [item, 0])),
    sentimentSeries: Array(80).fill(0),
    eventBuffer: [],
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

function buildWsUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const token = elements.apiToken.value.trim();
    const baseUrl = `${protocol}//${window.location.host}/api/v1/ws/stream`;
    if (!token) {
        return baseUrl;
    }
    return `${baseUrl}?token=${encodeURIComponent(token)}`;
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

    distributionChart.data.datasets[0].data = emotionLabels.map((emotion) => state.emotionCounts[emotion] || 0);
    distributionChart.update("none");

    elements.sentimentValue.textContent = sentiment.toFixed(2);
}

function handleFramePayload(payload) {
    if (!payload || !payload.frame || !payload.data) {
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

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return;
    }

    const ws = new WebSocket(buildWsUrl());
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

    ws.onclose = () => {
        state.isStreaming = false;
        updateButtons();
        stopHeartbeat();
        elements.noSignal.style.display = "flex";
        setStatus("idle", "Offline");
        pushEvent("Transmissão encerrada.");
    };
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
            if (response.status === 401) {
                setFeedback("Configuração protegida: informe token para carregar parâmetros.", "error");
                return;
            }
            throw new Error(`Falha ao carregar configuração (${response.status})`);
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
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `Não foi possível atualizar a configuração (${response.status})`);
        }

        const fields = data.updated_fields?.length ? data.updated_fields.join(", ") : "nenhum campo";
        setFeedback(`Configuração aplicada (${fields}).`, "success");
        pushEvent(`Configuração atualizada: ${fields}.`);
    } catch (error) {
        setFeedback(error.message, "error");
    }
}

async function fetchMetrics() {
    try {
        const response = await fetch("/api/v1/metrics", { headers: buildAuthHeaders() });
        if (!response.ok) {
            if (response.status === 401) {
                return;
            }
            throw new Error(`Métricas indisponíveis (${response.status})`);
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
            throw new Error(`Falha ao exportar métricas (${response.status})`);
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
    if (state.isStreaming) {
        closeWebSocket();
        return;
    }
    connectWebSocket();
}

function setupEvents() {
    elements.toggleStream.addEventListener("click", toggleStream);
    elements.settingsForm.addEventListener("submit", applyConfig);
    elements.refreshMetrics.addEventListener("click", fetchMetrics);
    elements.exportMetrics.addEventListener("click", exportMetrics);
    elements.captureFrame.addEventListener("click", captureCurrentFrame);
    elements.apiToken.addEventListener("change", fetchConfig);
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
