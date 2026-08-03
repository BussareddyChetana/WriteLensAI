// WriteLens AI Chart.js Visualization Module

function initProbabilityChart(canvasId, aiProb, humanProb) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['AI Generated', 'Human Written'],
            datasets: [{
                data: [aiProb, humanProb],
                backgroundColor: ['#b3432f', '#3a6b4a'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { family: 'Inter', size: 12 },
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            cutout: '72%'
        }
    });
}

function initAISourceChart(canvasId, sourcesData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !sourcesData || !sourcesData.length) return;

    const labels = sourcesData.map(s => s.name);
    const probs = sourcesData.map(s => s.probability);

    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Estimated Match Probability (%)',
                data: probs,
                backgroundColor: [
                    '#1e56b0', '#3a6cc0', '#5a86cf', '#84a6de', '#b3c9ec'
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(val) { return val + '%'; }
                    }
                }
            }
        }
    });
}

function initSentenceDistributionChart(canvasId, highRiskCount, modRiskCount, humanCount) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx.getContext('2d'), {
        type: 'pie',
        data: {
            labels: ['High AI Risk', 'Moderate AI Risk', 'Human Pattern'],
            datasets: [{
                data: [highRiskCount, modRiskCount, humanCount],
                backgroundColor: ['#b3432f', '#a8791f', '#3a6b4a'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}
