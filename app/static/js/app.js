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

// Live price for selected coin only
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

// Charts page
document.addEventListener('DOMContentLoaded', function() {
  const canvas = document.getElementById('price-chart');
  const coinSelect = document.getElementById('chart-coin');
  const status = document.getElementById('chart-status');
  if (!canvas || !coinSelect) return;
  
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
    const prices = data.prices || [];
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth - 40;
    canvas.height = 400;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--panel') || '#161b22';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const range = maxPrice - minPrice || 1;
    
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = (canvas.height / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
    
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    prices.forEach((price, i) => {
      const x = (i / (prices.length - 1)) * canvas.width;
      const y = canvas.height - ((price - minPrice) / range) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  
  coinSelect.addEventListener('change', function() {
    fetchChartData(this.value);
  });
  
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