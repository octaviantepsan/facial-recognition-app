document.addEventListener("DOMContentLoaded", () => {
    
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-file');
    const imagePreview = document.getElementById('image-preview');
    const featureSelect = document.getElementById('feature-select');
    const algorithmSelect = document.getElementById('algorithm-select');
    const kSelect = document.getElementById('k-select');
    const normSelect = document.getElementById('norm-select');
    const processButton = document.getElementById('processButton');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const resultsOutput = document.getElementById('results-output');
    const resultImage = document.getElementById('result-image');
    const datasetSelect = document.getElementById('dataset-select');
    const splitSelect = document.getElementById('split-select');
    const toast = document.getElementById('toast');
    
    const runStatsButton = document.getElementById('run-stats-button');
    const statsLoader = document.getElementById('stats-loader');
    const statsSection = document.getElementById('statistics-section');
    const downloadChartsButton = document.getElementById('download-charts-button');
    const accuracyChartCanvas = document.getElementById('accuracy-chart');
    const timeChartCanvas = document.getElementById('time-chart');
    const downloadCSVButton = document.getElementById('download-csv-button');
    const clearStatsButton = document.getElementById('clear-stats-button');
    const cacheMessage = document.getElementById('cache-message');

    let accuracyChart = null;
    let timeChart = null;
    let benchmarkData = null;

    imageInput.value = null;
    imagePreview.style.display = 'none';
    imagePreview.src = '';
    resultImage.style.display = 'none';
    resultImage.src = '';
    
    function updateUI() {
        const feat = featureSelect.value;
        const algo = algorithmSelect.value;

        if (feat === 'eigen_mean') {
            algorithmSelect.value = 'nn';
            algorithmSelect.disabled = true;
            kSelect.disabled = true;
        } else {
            algorithmSelect.disabled = false;
            if (algo === 'nn') {
                kSelect.disabled = true;
            } else {
                kSelect.disabled = false;
            }
        }
    }

    kSelect.disabled = true;
    algorithmSelect.addEventListener('change', updateUI);
    featureSelect.addEventListener('change', updateUI);

    // Reload Dataset
    async function reloadDataset() {
        const dataset = datasetSelect.value;
        const split = parseFloat(splitSelect.value);
        localStorage.removeItem('benchmarkData');
        resultsSection.classList.add('hidden');
        statsSection.classList.add('hidden');
        
        console.log(`Reloading dataset: ${dataset} with split ${split}`);

        try {
            const response = await fetch('http://localhost:5000/load_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: dataset, split: split })
            });
            if (response.ok) {
                toast.classList.remove('hidden');
                setTimeout(() => toast.classList.add('hidden'), 3000);
            } else { alert("Failed to load dataset."); }
        } catch (e) { console.error(e); alert("Error switching dataset."); }
    }

    datasetSelect.addEventListener('change', reloadDataset);
    splitSelect.addEventListener('change', reloadDataset);

    // Submit Form
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); 
        const file = imageInput.files[0];
        if (!file) {
            resultsOutput.textContent = "Please select an image file first.";
            resultsSection.classList.remove('hidden');
            return;
        }
        setLoading(true);

        const formData = new FormData();
        formData.append('image', file);
        formData.append('feature_mode', featureSelect.value);
        formData.append('algorithm', algorithmSelect.value);
        formData.append('k', kSelect.value);
        formData.append('normType', normSelect.value);

        const backendUrl = 'http://localhost:5000/process_image';
        try {
            const response = await fetch(backendUrl, { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                resultImage.src = "data:image/png;base64," + data.image_b64;
                resultImage.style.display = 'block';

                // GHOST LOGIC
                const ghostContainer = document.getElementById('ghost-container');
                const ghostImage = document.getElementById('ghost-image');
                if (data.ghost_b64) {
                    ghostImage.src = "data:image/png;base64," + data.ghost_b64;
                    ghostContainer.style.display = 'block';
                } else {
                    ghostContainer.style.display = 'none';
                }

                const outputText = `Algorithm Used: ${data.algorithm}
Voted Person: ${data.person_label}
Nearest Image Index: ${data.nearest_idx}`;
                resultsOutput.textContent = outputText;
            } else {
                resultImage.style.display = 'none';
                throw new Error(data.error || 'Unknown server error');
            }
        } catch (error) {
            console.error('Error:', error);
            resultsOutput.textContent = `Error connecting to backend:\n${error.message}`;
        } finally {
            setLoading(false);
            resultsSection.classList.remove('hidden');
        }
    });

    imageInput.addEventListener('change', async () => {
        const file = imageInput.files[0];
        if (!file) { imagePreview.src = ''; imagePreview.style.display = 'none'; return; }
        const formData = new FormData();
        formData.append('image', file);
        try {
            const response = await fetch('http://localhost:5000/preview', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                const imageSrc = "data:image/png;base64," + data.image_b64;
                imagePreview.src = imageSrc;
                imagePreview.style.display = 'block';
            } else { throw new Error(data.error); }
        } catch (error) { console.error(error); imagePreview.src = ''; imagePreview.style.display = 'none'; }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            processButton.disabled = true;
            processButton.textContent = 'Processing...';
            loader.classList.remove('hidden');
            resultsSection.classList.add('hidden'); 
            resultsOutput.textContent = '';
        } else {
            processButton.disabled = false;
            processButton.textContent = 'Process Image';
            loader.classList.add('hidden');
        }
    }

    // Statistics
    runStatsButton.addEventListener('click', async () => { 
        const cachedData = localStorage.getItem('benchmarkData');
        if (cachedData) {
            cacheMessage.textContent = "Displaying cached results. Click 'Clear Cache' to re-run.";
            benchmarkData = JSON.parse(cachedData);
            displayBenchmarkData(benchmarkData);
            return; 
        }
        cacheMessage.textContent = ""; 
        statsLoader.classList.remove('hidden');
        statsSection.classList.add('hidden');
        runStatsButton.disabled = true;
        clearStatsButton.disabled = true;

        try {
            const response = await fetch('http://localhost:5000/run_statistics', { method: 'POST' });
            const data = await response.json(); 
            if (response.ok) {
                benchmarkData = data;
                localStorage.setItem('benchmarkData', JSON.stringify(data));
                displayBenchmarkData(data);
            } else { throw new Error(data.error); }
        } catch (error) {
            console.error("Statistics failed:", error);
            resultsOutput.textContent = `Error running statistics:\n${error.message}`;
            resultsSection.classList.remove('hidden');
        } finally {
            statsLoader.classList.add('hidden');
            runStatsButton.disabled = false;
            clearStatsButton.disabled = false;
        }
    });

    function median(arr) {
        if (!arr.length) return 0;
        const sorted = arr.slice().sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
        return sorted[mid];
    }

    function calculateMedianTimeData(data) {
        const normTimes = {
            'cos': { times: [], name: 'Cosine' },
            '2': { times: [], name: 'Euclidean (L2)' },
            '1': { times: [], name: 'Manhattan (L1)' },
            'inf': { times: [], name: 'Chebyshev (L-inf)' }
        };

        for (const item of data) {
            if (item.name.endsWith('cos')) normTimes['cos'].times.push(item.time_ms);
            else if (item.name.endsWith('2')) normTimes['2'].times.push(item.time_ms);
            else if (item.name.endsWith('1')) normTimes['1'].times.push(item.time_ms);
            else if (item.name.endsWith('inf')) normTimes['inf'].times.push(item.time_ms);
        }

        const medianData = [];
        for (const key in normTimes) {
            medianData.push({
                name: normTimes[key].name,
                time_ms: median(normTimes[key].times)
            });
        }
        return medianData;
    }

    function displayBenchmarkData(data) {
        if (accuracyChart) accuracyChart.destroy();
        if (timeChart) timeChart.destroy();
        accuracyChart = renderChart(accuracyChartCanvas, data, 'accuracy', 'Accuracy (%)', 'rgba(75, 192, 192, 0.6)');
        const medianTimeData = calculateMedianTimeData(data);
        timeChart = renderChart(timeChartCanvas, medianTimeData, 'time_ms', 'Median Time (ms)', 'rgba(255, 99, 132, 0.6)');
        statsSection.classList.remove('hidden');
        downloadChartsButton.disabled = false;
        downloadCSVButton.disabled = false;
    }

    function renderChart(canvas, data, dataKey, label, color) {
        let processedLabels = data.map(d => d.name);
        let processedValues = data.map(d => d[dataKey]);
        
        return new Chart(canvas, {
            type: 'bar',
            data: {
                labels: processedLabels,
                datasets: [{
                    label: label,
                    data: processedValues,
                    backgroundColor: color,
                    borderColor: color.replace('0.6', '1'),
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true, ticks: { color: '#d1d5db' }, grid: { color: '#374151' } },
                    x: { ticks: { color: '#d1d5db' }, grid: { color: '#374151' } }
                },
                plugins: { legend: { labels: { color: '#d1d5db' } } }
            }
        });
    }

    function downloadChart(chart, filename) {
        if (!chart) return;
        const a = document.createElement('a');
        a.href = chart.canvas.toDataURL('image/png');
        a.download = filename;
        a.click();
    }

    downloadChartsButton.addEventListener('click', () => {
        downloadChart(accuracyChart, 'accuracy_chart.png');
        downloadChart(timeChart, 'time_chart.png');
    });

    clearStatsButton.addEventListener('click', () => {
        localStorage.removeItem('benchmarkData');
        benchmarkData = null;
        statsSection.classList.add('hidden');
        cacheMessage.textContent = "";
        if (accuracyChart) accuracyChart.destroy();
        if (timeChart) timeChart.destroy();
        downloadCSVButton.disabled = true;
        resultsOutput.textContent = "Benchmark cache cleared. Click 'Show/Run Statistics' to re-calculate.";
        resultsSection.classList.remove('hidden');
    });

    function generateCSV(data) {
        if (!data || data.length === 0) return;
        const headers = ["TestName", "Accuracy(%)", "Time(ms)"];
        let csvContent = headers.join(",") + "\n";
        data.forEach(row => {
            const name = `"${row.name}"`;
            const accuracy = row.accuracy.toFixed(2);
            const time = row.time_ms.toFixed(2);
            csvContent += [name, accuracy, time].join(",") + "\n";
        });
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "benchmark_results.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    downloadCSVButton.addEventListener('click', () => {
        generateCSV(benchmarkData);
    });
});