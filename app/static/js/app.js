// Theme switcher - runs immediately
(() => {
  const root = document.documentElement;
  root.dataset.theme = localStorage.getItem('runguard-theme') || 'dark';
  document.querySelectorAll('[data-theme-choice]').forEach(btn => {
    btn.addEventListener('click', () => {
      root.dataset.theme = btn.dataset.themeChoice;
      localStorage.setItem('runguard-theme', btn.dataset.themeChoice);
    });
  });
})();

// Everything else waits for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
  // Form handling
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

  // Live prices
  const coinSelect = document.querySelector('#coin');
  const livePrice = document.querySelector('#current-price');
  let currentPrices = {};
  
  async function loadLivePrices() {
    if (!coinSelect || !livePrice) return;
    try {
      // CRITICAL: Added a timeout so the UI doesn't hang forever
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10-second timeout
      const response = await fetch('/api/prices', { signal: controller.signal });
      clearTimeout(timeoutId);
      
      if (!response.ok) throw new Error('Live price service unavailable');
      const body = await response.json();
      currentPrices = body.prices || {};
      
      [...coinSelect.options].forEach(option => {
        const base = option.textContent.replace(/\s+•\s+\$[\d,\.]+$/, '');
        const info = currentPrices[option.value];
        option.textContent = info ? `${base} • $${Number(info.price).toLocaleString(undefined, {maximumFractionDigits: 8})}` : base;
      });
      
      updateSelectedPrice();
    } catch (error) {
      // Don't show "Live price unavailable" immediately - try again in 5s
      console.error('Failed to load live prices:', error);
      setTimeout(loadLivePrices, 5000);
    }
  }
  
  function formatObservedTime(value) {
    if (!value) return 'time unavailable';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'time unavailable';
    return date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  }
  
  function updateSelectedPrice() {
    if (!coinSelect || !livePrice) return;
    const info = currentPrices[coinSelect.value];
    livePrice.textContent = info
      ? `Live price: $${Number(info.price).toLocaleString(undefined, {maximumFractionDigits: 8})} · updated ${formatObservedTime(info.observed_at_utc)}`
      : 'Live price unavailable';
  }
  
  coinSelect?.addEventListener('change', updateSelectedPrice);
  
  // Load prices on page load
  loadLivePrices();
  
  // Refresh every 60 seconds
  window.setInterval(loadLivePrices, 60000);

  // Prediction result refresh
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
    } catch (_) {
      // Keep existing result
    }
  }
  
  if (resultPanel) {
    refreshPredictionResult();
    window.setInterval(refreshPredictionResult, 30000);
  }
});