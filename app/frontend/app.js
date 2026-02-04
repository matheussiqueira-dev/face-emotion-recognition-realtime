const canvas = document.getElementById('video-canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const trackCountEl = document.getElementById('track-count');
const fpsValueEl = document.getElementById('fps-value');
const toggleBtn = document.getElementById('toggle-stream');
const noSignalEl = document.getElementById('no-signal');

let ws = null;
let isStreaming = false;

// Initialize Charts
const distributionCtx = document.getElementById('distributionChart').getContext('2d');
const distributionChart = new Chart(distributionCtx, {
    type: 'doughnut',
    data: {
        labels: ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral', 'Fear', 'Disgust'],
        datasets: [{
            data: [0, 0, 0, 0, 0, 0, 0],
            backgroundColor: [
                '#fcd34d', '#60a5fa', '#f87171', '#c084fc', '#94a3b8', '#818cf8', '#4ade80'
            ],
            borderWidth: 0
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        cutout: '70%'
    }
});

const timelineCtx = document.getElementById('timelineChart').getContext('2d');
const timelineChart = new Chart(timelineCtx, {
    type: 'line',
    data: {
        labels: Array(50).fill(''),
        datasets: [{
            label: 'Average Sentiment',
            data: Array(50).fill(0),
            borderColor: '#6366f1',
            tension: 0.4,
            fill: true,
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            pointRadius: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            y: { display: false, min: -1, max: 1 },
            x: { display: false }
        }
    }
});

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        statusEl.textContent = 'Active';
        statusEl.classList.remove('pulse');
        statusEl.style.color = '#4ade80';
        noSignalEl.style.display = 'none';
        isStreaming = true;
        toggleBtn.textContent = 'Stop Stream';
        toggleBtn.classList.replace('btn-primary', 'btn-secondary');
    };
    
    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        const { frame, data } = payload;
        
        // Update Stats
        fpsValueEl.textContent = data.fps;
        trackCountEl.textContent = data.tracks.length;
        
        // Draw Frame
        const image = new Image();
        image.onload = () => {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
            
            // Draw Overlays
            drawDetections(data.tracks);
        };
        image.src = `data:image/jpeg;base64,${frame}`;
        
        // Update Charts
        updateCharts(data.tracks);
    };
    
    ws.onclose = () => {
        statusEl.textContent = 'Disconnected';
        statusEl.style.color = '#ef4444';
        noSignalEl.style.display = 'block';
        isStreaming = false;
        toggleBtn.textContent = 'Start Stream';
        toggleBtn.classList.replace('btn-secondary', 'btn-primary');
    };
}

function drawDetections(tracks) {
    tracks.forEach(track => {
        const { x, y, w, h } = track.box;
        
        // Draw Box
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        
        // Rounded rectangle corners (custom style)
        const r = 10;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.stroke();
        
        // Label Background
        if (track.emotion) {
            const label = `${track.emotion.dominant.toUpperCase()} ${Math.round(Math.max(...Object.values(track.emotion.scores)))}%`;
            ctx.font = 'bold 14px Inter';
            const bgWidth = ctx.measureText(label).width + 20;
            
            ctx.fillStyle = '#6366f1';
            ctx.fillRect(x, y - 30, bgWidth, 30);
            
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 10, y - 10);
        }
    });
}

const emotionWeights = {
    'happy': 1, 'surprise': 0.5, 'neutral': 0, 
    'sad': -0.5, 'fear': -0.7, 'angry': -0.8, 'disgust': -1
};

function updateCharts(tracks) {
    if (tracks.length === 0) return;
    
    // Distribution Chart
    const counts = { 'happy': 0, 'sad': 0, 'angry': 0, 'surprise': 0, 'neutral': 0, 'fear': 0, 'disgust': 0 };
    let avgSentiment = 0;
    
    tracks.forEach(t => {
        if (t.emotion) {
            counts[t.emotion.dominant]++;
            avgSentiment += emotionWeights[t.emotion.dominant] || 0;
        }
    });
    
    distributionChart.data.datasets[0].data = Object.values(counts);
    distributionChart.update('none');
    
    // Timeline Chart
    avgSentiment /= tracks.length;
    timelineChart.data.datasets[0].data.push(avgSentiment);
    timelineChart.data.datasets[0].data.shift();
    timelineChart.update('none');
}

toggleBtn.addEventListener('click', () => {
    if (isStreaming) {
        ws.close();
    } else {
        connectWebSocket();
    }
});
