// Theme switcher (dropdown)
(function() {
  const themeSelect = document.getElementById('theme-select');
  const root = document.documentElement;
  const savedTheme = localStorage.getItem('runguard-theme') || 'dark';
  root.dataset.theme = savedTheme;
  themeSelect.value = savedTheme;
  themeSelect.addEventListener('change', function() {
    root.dataset.theme = this.value;
    localStorage.setItem('runguard-theme', this.value);
  });
})();

// Live Prices Table (Market page)
document.addEventListener('DOMContentLoaded', function() {
  const tableBody = document.getElementById('price-table-body');
  if (!tableBody) return;
  
  async function updatePrices() {
    tableBody.innerHTML = '<tr><td colspan="4">Refreshing prices...</td></tr>';
    try {
      const response = await fetch('/api/prices');
      if (!response.ok) throw new Error('API Error: ' + response.status);
      const data = await response.json();
      const prices = data.prices || {};
      
      let html = '';
      for (const [coinId, info] of Object.entries(prices)) {
        const price = Number(info.price).toLocaleString(undefined, { maximumFractionDigits: 8 });
        const change = info.price_change_percentage_24h;
        const changeDisplay = change !== undefined ? Number(change).toFixed(2) + '%' : 'N/A';
        const changeClass = change >= 0 ? 'positive' : 'negative';
        html += `<tr>
          <td>${info.name}</td>
          <td>${info.symbol}</td>
          <td>$${price}</td>
          <td class="${changeClass}">${changeDisplay}</td>
        </tr>`;
      }
      
      if (html === '') {
        tableBody.innerHTML = '<tr><td colspan="4">No coin data available.</td></tr>';
      } else {
        tableBody.innerHTML = html;
      }
    } catch (error) {
      console.error('Error updating prices:', error);
      tableBody.innerHTML = '<tr><td colspan="4" style="color: #ef4444;">Unable to load prices. Please refresh.</td></tr>';
    }
  }
  
  updatePrices();
  setInterval(updatePrices, 60000);
});

// Top Movers (Market page)
document.addEventListener('DOMContentLoaded', function() {
  const gainersList = document.getElementById('gainers-list');
  const losersList = document.getElementById('losers-list');
  if (!gainersList || !losersList) return;
  
  async function fetchMovers() {
    gainersList.innerHTML = '<tr><td colspan="3">Loading...</td></tr>';
    losersList.innerHTML = '<tr><td colspan="3">Loading...</td></tr>';
    try {
      const response = await fetch('/api/movers');
      if (!response.ok) throw new Error('API Error: ' + response.status);
      const data = await response.json();
      
      if (data.gainers && data.gainers.length > 0) {
        gainersList.innerHTML = data.gainers.map(coin => `
          <tr>
            <td>${coin.name} (${coin.symbol})</td>
            <td>$${Number(coin.price).toLocaleString(undefined, {maximumFractionDigits: 8})}</td>
            <td class="positive">+${Number(coin.price_change_percentage_24h).toFixed(2)}%</td>
          </tr>
        `).join('');
      } else {
        gainersList.innerHTML = '<tr><td colspan="3">No gainers data.</td></tr>';
      }
      
      if (data.losers && data.losers.length > 0) {
        losersList.innerHTML = data.losers.map(coin => `
          <tr>
            <td>${coin.name} (${coin.symbol})</td>
            <td>$${Number(coin.price).toLocaleString(undefined, {maximumFractionDigits: 8})}</td>
            <td class="negative">${Number(coin.price_change_percentage_24h).toFixed(2)}%</td>
          </tr>
        `).join('');
      } else {
        losersList.innerHTML = '<tr><td colspan="3">No losers data.</td></tr>';
      }
    } catch (error) {
      console.error('Error fetching movers:', error);
      gainersList.innerHTML = '<tr><td colspan="3" style="color: #ef4444;">Unable to load movers.</td></tr>';
      losersList.innerHTML = '<tr><td colspan="3" style="color: #ef4444;">Unable to load movers.</td></tr>';
    }
  }
  fetchMovers();
  setInterval(fetchMovers, 60000);
});

// Prediction form logic
document.addEventListener('DOMContentLoaded', function() {
  const form = document.querySelector('#predict-form');
  const horizonSelect = document.querySelector('#horizon');
  const intervalInput = document.querySelector('#interval');
  const intervalButtons = document.querySelectorAll('.interval');
  const predictBtn = document.querySelector('#predict-btn');
  
  const options = {
    hourly: [[1,'1 hour'],[2,'2 hours'],[3,'3 hours'],[6,'6 hours'],[12,'12 hours'],[24,'24 hours'],[48,'2 days'],[72,'3 days'],[168,'7 days'],[336,'14 days'],[720,'30 days']],
    daily: [[1,'1 day'],[2,'2 days'],[3,'3 days'],[7,'7 days'],[14,'14 days'],[30,'30 days']]
  };
  
  if (horizonSelect && intervalInput) {
    function populate() {
      const interval = intervalInput.value;
      const current = Number(horizonSelect.dataset.value || horizonSelect.value || 1);
      horizonSelect.innerHTML = options[interval].map(([value,label]) => `<option value="${value}">${label}</option>`).join('');
      horizonSelect.value = options[interval].some(([v]) => v === current) ? String(current) : String(options[interval][0][0]);
      horizonSelect.dataset.value = horizonSelect.value;
    }
    
    horizonSelect.dataset.value = horizonSelect.dataset.selected || '1';
    populate();
    intervalButtons.forEach(btn => btn.addEventListener('click', () => {
      intervalButtons.forEach(b => b.classList.toggle('active', b === btn));
      intervalInput.value = btn.dataset.interval;
      horizonSelect.dataset.value = '1';
      populate();
    }));
    horizonSelect.addEventListener('change', () => { horizonSelect.dataset.value = horizonSelect.value; });
    
    form?.addEventListener('submit', () => {
      if (predictBtn) { predictBtn.disabled = true; predictBtn.textContent = 'Predicting…'; }
    });
  }
});

// Live price for selected coin only (Predict page)
document.addEventListener('DOMContentLoaded', function() {
  const coinSelect = document.querySelector('#coin');
  const livePrice = document.querySelector('#current-price');
  if (!coinSelect || !livePrice) return;
  
  async function fetchLivePrice() {
    const coinId = coinSelect.value;
    try {
      const response = await fetch(`/api/prices?coin=${coinId}`);
      if (!response.ok) throw new Error('Failed to fetch price');
      const data = await response.json();
      const info = data.prices[coinId];
      if (info) {
        livePrice.textContent = `Live price: $${Number(info.price).toLocaleString(undefined, { maximumFractionDigits: 8 })}`;
      }
    } catch (error) {
      livePrice.textContent = 'Live price unavailable';
    }
  }
  
  coinSelect.addEventListener('change', fetchLivePrice);
  fetchLivePrice();
  setInterval(fetchLivePrice, 60000);
});

// Chart.js implementation
document.addEventListener('DOMContentLoaded', function() {
  const canvas = document.getElementById('price-chart');
  const coinSelect = document.getElementById('chart-coin');
  const status = document.getElementById('chart-status');
  if (!canvas || !coinSelect) return;
  
  let chart = null;
  
  async function fetchChartData(coinId) {
    status.textContent = 'Loading chart...';
    try {
      const response = await fetch(`/api/chart/${coinId}`);
      if (!response.ok) throw new Error('Failed to fetch chart data');
      const data = await response.json();
      renderChart(data);
      status.textContent = '';
    } catch (error) {
      status.textContent = 'Unable to load chart. Please try again.';
    }
  }
  
  function renderChart(data) {
    const labels = data.labels || [];
    const prices = data.prices || [];
    if (chart) chart.destroy();
    
    const ctx = canvas.getContext('2d');
    const root = document.documentElement;
    const textColor = getComputedStyle(root).getPropertyValue('--text').trim() || '#e6edf3';
    const gridColor = getComputedStyle(root).getPropertyValue('--border').trim() || '#30363d';
    const accentColor = getComputedStyle(root).getPropertyValue('--accent').trim() || '#58a6ff';
    
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Price (USD)',
          data: prices,
          borderColor: accentColor,
          backgroundColor: 'rgba(88, 166, 255, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { labels: { color: textColor, font: { size: 14 } } }
        },
        scales: {
          x: { ticks: { color: textColor, maxTicksLimit: 10 }, grid: { color: gridColor } },
          y: { ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }
  
  coinSelect.addEventListener('change', function() { fetchChartData(this.value); });
  fetchChartData(coinSelect.value);
});

// Prediction result refresh
document.addEventListener('DOMContentLoaded', function() {
  const resultPanel = document.querySelector('#result[data-prediction-id]');
  async function refreshPredictionResult() {
    if (!resultPanel) return;
    const id = resultPanel.dataset.predictionId;
    if (!id) return;
    try {
      const response = await fetch(`/api/predictions/${encodeURIComponent(id)}`);
      if (!response.ok) return;
      const prediction = await response.json();
      const status = document.querySelector('#prediction-status');
      const actual = document.querySelector('#actual-price');
      const result = document.querySelector('#prediction-result');
      if (status) status.textContent = String(prediction.status || 'pending').replace(/_/g, ' ');
      if (prediction.status === 'resolved') {
        if (actual) actual.textContent = `$${Number(prediction.actual_price).toLocaleString(undefined, {maximumFractionDigits: 8})}`;
        if (result) result.textContent = prediction.direction_correct ? 'CORRECT ✓' : 'WRONG ✗';
      }
    } catch (_) {}
  }
  if (resultPanel) {
    refreshPredictionResult();
    setInterval(refreshPredictionResult, 30000);
  }
});