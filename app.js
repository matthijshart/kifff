// ── KIFID Predictor — Application Logic ──

// ── State ──
let uitspraken = [];
let currentTab = 'predict';
let policyAnalysis = null;

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  autoLoadDataset();
});

// ── Smooth scroll ──
function scrollToTool() {
  const el = document.getElementById('tool-section');
  if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function scrollToFeatures() {
  const el = document.getElementById('features');
  if (el) el.scrollIntoView({ behavior: 'smooth' });
}

// ── Tab Switching ──
function switchTab(tab) {
  currentTab = tab;
  const tabs = ['predict','data','insights'];
  document.querySelectorAll('.tab-btn').forEach((btn, i) => {
    const isActive = tabs[i] === tab;
    btn.classList.toggle('active', isActive);
  });
  document.getElementById('tab-predict').style.display = tab === 'predict' ? 'grid' : 'none';
  document.getElementById('tab-data').style.display = tab === 'data' ? 'block' : 'none';
  document.getElementById('tab-insights').style.display = tab === 'insights' ? 'block' : 'none';
}

// ── Auto-load Dataset ──
async function autoLoadDataset() {
  try {
    const response = await fetch('data/uitspraken/dataset.json');
    if (!response.ok) return;
    const data = await response.json();
    uitspraken = data.uitspraken || [];
    if (uitspraken.length === 0) return;
    updateStats();
    updateInsights();
    updateHeroStats();
    // Update data tab
    const statusEl = document.getElementById('dataStatus');
    if (statusEl) {
      statusEl.className = 'data-status';
      statusEl.innerHTML = uitspraken.length + ' uitspraken automatisch geladen uit dataset.';
    }
    const uploadedEl = document.getElementById('uploadedFiles');
    if (uploadedEl) {
      uploadedEl.innerHTML = '<div style="display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;">Dataset — ' + uitspraken.length + ' uitspraken (automatisch geladen)</div>';
    }
    const loadBtn = document.querySelector('[onclick="loadFullDataset()"]');
    if (loadBtn) loadBtn.textContent = 'Dataset herladen (' + uitspraken.length + ' uitspraken)';
  } catch (e) {
    // Silently fail
  }
}

function updateHeroStats() {
  if (uitspraken.length === 0) return;
  const total = uitspraken.length;
  const toegewezen = uitspraken.filter(u => u.uitkomst === 'toegewezen').length;
  const deels = uitspraken.filter(u => u.uitkomst === 'deels').length;
  const afgewezen = uitspraken.filter(u => u.uitkomst === 'afgewezen').length;
  const consumerWin = Math.round((toegewezen + deels) / total * 100);
  const insurerWin = Math.round(afgewezen / total * 100);

  const el1 = document.getElementById('heroStatTotal');
  if (el1) el1.textContent = total.toLocaleString('nl-NL');
  const el2 = document.getElementById('heroStatConsumer');
  if (el2) el2.textContent = consumerWin + '%';
  const el3 = document.getElementById('heroStatInsurer');
  if (el3) el3.textContent = insurerWin + '%';
}

// ── Stats ──
function updateStats() {
  if (uitspraken.length === 0) return;
}

// ── File Upload ──
function handleFileUpload(event) {
  const files = event.target.files;
  for (const file of files) {
    if (file.name.endsWith('.csv')) readCSV(file);
    else if (file.name.endsWith('.json')) readJSON(file);
    else alert('Upload CSV of JSON bestanden.');
  }
}

function readCSV(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const lines = e.target.result.split('\n');
    const headers = lines[0].split(/[,;]/).map(h => h.trim().toLowerCase());
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(/[,;]/);
      if (vals.length < 3) continue;
      const row = {};
      headers.forEach((h, j) => row[h] = vals[j]?.trim());
      uitspraken.push(row);
    }
    updateStats();
    showUploadSuccess(file.name, lines.length - 1);
  };
  reader.readAsText(file);
}

function readJSON(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      const items = Array.isArray(data) ? data : data.uitspraken || [];
      uitspraken.push(...items);
      updateStats();
      showUploadSuccess(file.name, items.length);
    } catch (err) {
      alert('Fout bij lezen JSON: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function showUploadSuccess(name, count) {
  const el = document.getElementById('uploadedFiles');
  el.innerHTML += '<div style="display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;margin-bottom:8px;">' + name + ' — ' + count + ' uitspraken geladen</div>';
  document.getElementById('dataStatus').className = 'data-status';
  document.getElementById('dataStatus').innerHTML = uitspraken.length + ' uitspraken geladen en klaar voor analyse.';
}

// ── Full Dataset ──
async function loadFullDataset() {
  try {
    const response = await fetch('data/uitspraken/dataset.json');
    const data = await response.json();
    uitspraken = data.uitspraken || [];
    updateStats();
    document.getElementById('dataStatus').className = 'data-status';
    document.getElementById('dataStatus').innerHTML = uitspraken.length + ' uitspraken geladen uit volledige dataset.';
    document.getElementById('uploadedFiles').innerHTML = '<div style="display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;">Volledige dataset — ' + uitspraken.length + ' uitspraken</div>';
    updateInsights();
    updateHeroStats();
  } catch (err) {
    alert('Fout bij laden dataset: ' + err.message);
  }
}

// ── Demo Data ──
function loadDemoData() {
  uitspraken = [
    { uitspraaknr: "2024-0500", datum: "2024-06-15", type_verzekering: "autoverzekering", kerngeschil: "schadevrije_jaren", uitkomst: "afgewezen", bedrag_gevorderd: 2400, bedrag_toegewezen: 0, bindend: "ja", samenvatting: "Schadevrije jaren gekoppeld aan verzekeringnemer, niet aan bestuurder." },
    { uitspraaknr: "2024-1029", datum: "2024-11-20", type_verzekering: "woonhuisverzekering", kerngeschil: "uitleg_voorwaarden", uitkomst: "toegewezen", bedrag_gevorderd: 3980, bedrag_toegewezen: 3980, bindend: "nee", samenvatting: "Vervangingskosten vallen niet onder reparatiekosten-uitsluiting." },
    { uitspraaknr: "2024-1005", datum: "2024-11-15", type_verzekering: "woonhuisverzekering", kerngeschil: "dekkingsweigering", uitkomst: "afgewezen", bedrag_gevorderd: 75000, bedrag_toegewezen: 10000, bindend: "ja", samenvatting: "Neerslagcriterium KNMI niet gehaald, coulance 10k uitgekeerd." },
    { uitspraaknr: "2024-0861", datum: "2024-09-10", type_verzekering: "overlijdensrisicoverzekering", kerngeschil: "mededelingsplicht", uitkomst: "afgewezen", bedrag_gevorderd: 300000, bedrag_toegewezen: 0, bindend: "ja", samenvatting: "Fout adres opgegeven, verzekering nooit tot stand gekomen." },
    { uitspraaknr: "2024-0660", datum: "2024-07-22", type_verzekering: "autoverzekering", kerngeschil: "premiegeschil", uitkomst: "afgewezen", bedrag_gevorderd: 1600, bedrag_toegewezen: 0, bindend: "nee", samenvatting: "Geen rechtsgrond voor vordering na opzegging door consument zelf." },
    { uitspraaknr: "2024-0707", datum: "2024-08-05", type_verzekering: "autoverzekering", kerngeschil: "premiegeschil", uitkomst: "afgewezen", bedrag_gevorderd: 500, bedrag_toegewezen: 0, bindend: "nee", samenvatting: "Premieaanpassing niet onaanvaardbaar naar maatstaven R&B." },
    { uitspraaknr: "2024-0024", datum: "2024-01-18", type_verzekering: "rechtsbijstandverzekering", kerngeschil: "dekkingsweigering", uitkomst: "afgewezen", bedrag_gevorderd: 8000, bedrag_toegewezen: 0, bindend: "nee", samenvatting: "Coinbase/beleggingsfraude geen gedekte gebeurtenis." },
    { uitspraaknr: "2024-0069", datum: "2024-02-28", type_verzekering: "beleggingsverzekering", kerngeschil: "zorgplicht", uitkomst: "deels", bedrag_gevorderd: 12000, bedrag_toegewezen: 4215, bindend: "ja", samenvatting: "Adviseur tekortgeschoten in nazorgverplichting, schade deels toewijsbaar." },
    { uitspraaknr: "2023-0842", datum: "2023-10-15", type_verzekering: "woonhuisverzekering", kerngeschil: "dekkingsweigering", uitkomst: "toegewezen", bedrag_gevorderd: 22000, bedrag_toegewezen: 18500, bindend: "ja", samenvatting: "Stormschade gedekt, uitsluiting te ruim geinterpreteerd door verzekeraar." },
    { uitspraaknr: "2023-0631", datum: "2023-08-02", type_verzekering: "reisverzekering", kerngeschil: "dekkingsweigering", uitkomst: "toegewezen", bedrag_gevorderd: 4500, bedrag_toegewezen: 4500, bindend: "nee", samenvatting: "Annulering wegens ziekte gedekt, verzekeraar onvoldoende onderbouwing afwijzing." },
  ];
  updateStats();
  document.getElementById('dataStatus').className = 'data-status';
  document.getElementById('dataStatus').innerHTML = uitspraken.length + ' demo-uitspraken geladen.';
  document.getElementById('uploadedFiles').innerHTML = '<div style="display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;">Demo-dataset — ' + uitspraken.length + ' uitspraken</div>';
  updateInsights();
  updateHeroStats();
}

// ── Prediction ──
function runPrediction() {
  const btn = document.getElementById('btnPredict');
  btn.disabled = true;
  btn.textContent = 'Analyseren...';

  const resultsArea = document.getElementById('resultsArea');
  resultsArea.innerHTML = '<div class="result-hero"><div class="loading"><div class="spinner"></div><div class="loading-text">Analyseren van ' + uitspraken.length + ' uitspraken...</div></div></div>';

  const input = {
    type: document.getElementById('insuranceType').value,
    dispute: document.getElementById('coreDispute').value,
    amount: parseFloat(document.getElementById('claimAmount').value) || 0,
    binding: document.getElementById('bindingAdvice').value,
    evidence: document.getElementById('consumerEvidence').value,
    expert: document.getElementById('expertReport').value,
    goodwill: document.getElementById('goodwillOffer').value,
    context: document.getElementById('additionalContext').value,
  };

  setTimeout(() => {
    const result = analyzeCase(input);
    renderResults(result, input);
    btn.disabled = false;
    btn.textContent = 'Analyseer & Voorspel';
  }, 1200);
}

function analyzeCase(input) {
  let score = 50;
  const factors = [];

  // 1. Base rate per insurance type
  const typeMatches = uitspraken.filter(u => u.type_verzekering === input.type);
  if (typeMatches.length > 0) {
    const afgewezen = typeMatches.filter(u => u.uitkomst === 'afgewezen').length;
    const baseRate = Math.round(afgewezen / typeMatches.length * 100);
    score = baseRate;
    factors.push({ label: 'Afwijzingspercentage ' + input.type + ' (n=' + typeMatches.length + ')', value: baseRate + '%', type: baseRate > 60 ? 'pro' : 'con' });
  }

  // 2. Base rate per dispute type
  const disputeMatches = uitspraken.filter(u => u.kerngeschil === input.dispute);
  if (disputeMatches.length > 0) {
    const afgewezen = disputeMatches.filter(u => u.uitkomst === 'afgewezen').length;
    const disputeRate = Math.round(afgewezen / disputeMatches.length * 100);
    const adjustment = Math.round((disputeRate - 50) * 0.3);
    score += adjustment;
    factors.push({ label: 'Kerngeschil ' + input.dispute + ' (n=' + disputeMatches.length + ', ' + disputeRate + '% afgewezen)', value: (adjustment >= 0 ? '+' : '') + adjustment, type: adjustment > 0 ? 'pro' : 'con' });
  }

  // 3. Consumer evidence
  if (input.evidence === 'sterk') {
    const sterkBewijs = uitspraken.filter(u => u.beslisfactoren && u.beslisfactoren.bewijs_consument === 'sterk');
    const sterkToegewezen = sterkBewijs.filter(u => u.uitkomst === 'toegewezen').length;
    const pct = sterkBewijs.length ? Math.round(sterkToegewezen / sterkBewijs.length * 100) : 70;
    score -= 15;
    factors.push({ label: 'Sterk consumentenbewijs (' + pct + '% toewijzing in data)', value: '-15', type: 'con' });
  } else if (input.evidence === 'zwak') {
    const zwakBewijs = uitspraken.filter(u => u.beslisfactoren && u.beslisfactoren.bewijs_consument === 'zwak');
    const zwakAfgewezen = zwakBewijs.filter(u => u.uitkomst === 'afgewezen').length;
    const pct = zwakBewijs.length ? Math.round(zwakAfgewezen / zwakBewijs.length * 100) : 80;
    score += 15;
    factors.push({ label: 'Zwak consumentenbewijs (' + pct + '% afwijzing in data)', value: '+15', type: 'pro' });
  }

  // 4. Expert report
  if (input.expert === 'verzekeraar') {
    score += 10;
    factors.push({ label: 'Deskundigenrapport verzekeraar', value: '+10', type: 'pro' });
  } else if (input.expert === 'consument') {
    score -= 10;
    factors.push({ label: 'Deskundigenrapport consument', value: '-10', type: 'con' });
  } else if (input.expert === 'beide') {
    score -= 3;
    factors.push({ label: 'Beide partijen deskundigenrapport', value: '-3', type: 'neutral' });
  }

  // 5. Goodwill offer
  if (input.goodwill === 'ja_redelijk') {
    score += 8;
    factors.push({ label: 'Redelijk coulanceaanbod (commissie weegt mee)', value: '+8', type: 'pro' });
  } else if (input.goodwill === 'ja_laag') {
    score += 3;
    factors.push({ label: 'Laag coulanceaanbod (beperkt effect)', value: '+3', type: 'neutral' });
  } else {
    const geenCoulance = uitspraken.filter(u => u.beslisfactoren && u.beslisfactoren.coulance_aangeboden === false);
    const geenCoulanceToegewezen = geenCoulance.filter(u => u.uitkomst === 'toegewezen' || u.uitkomst === 'deels').length;
    if (geenCoulance.length > 5) {
      const pct = Math.round(geenCoulanceToegewezen / geenCoulance.length * 100);
      if (pct > 40) {
        factors.push({ label: 'Geen coulance aangeboden (' + pct + '% toewijzing zonder coulance)', value: 'risico', type: 'neutral' });
      }
    }
  }

  // 6. Claim amount
  if (input.amount > 50000) {
    score -= 3;
    factors.push({ label: 'Hoog bedrag >50k (consument meer gemotiveerd)', value: '-3', type: 'neutral' });
  } else if (input.amount > 0 && input.amount < 2000) {
    score += 3;
    factors.push({ label: 'Laag bedrag <2k (minder proceswaardig)', value: '+3', type: 'neutral' });
  }

  // 7. Binding advice
  if (input.binding === 'bindend') {
    factors.push({ label: 'Bindend advies (uitspraak is definitief)', value: 'info', type: 'neutral' });
  }

  // 8. Policy analysis
  if (policyAnalysis) {
    const riskScore = policyAnalysis.risicoscore || 0;
    if (riskScore >= 7) {
      const adj = -Math.round((riskScore - 5) * 3);
      score += adj;
      factors.push({ label: 'Polisvoorwaarden risicoscore ' + riskScore + '/10 (risicovolle clausules)', value: String(adj), type: 'con' });
    } else if (riskScore >= 4) {
      const adj = -Math.round((riskScore - 5) * 2);
      score += adj;
      factors.push({ label: 'Polisvoorwaarden risicoscore ' + riskScore + '/10', value: (adj >= 0 ? '+' : '') + adj, type: 'neutral' });
    } else if (riskScore > 0) {
      const adj = Math.round((5 - riskScore) * 2);
      score += adj;
      factors.push({ label: 'Polisvoorwaarden risicoscore ' + riskScore + '/10 (duidelijke voorwaarden)', value: '+' + adj, type: 'pro' });
    }

    const hoogClausules = (policyAnalysis.risicovolle_clausules || []).filter(c => c.ernst === 'hoog').length;
    if (hoogClausules >= 3) {
      score -= 5;
      factors.push({ label: hoogClausules + ' hoog-risico clausules in polisvoorwaarden', value: '-5', type: 'con' });
    }
  }

  score = Math.max(5, Math.min(95, score));

  // Confidence
  const relevantMatches = uitspraken.filter(u => u.type_verzekering === input.type || u.kerngeschil === input.dispute).length;
  let confidence = 'laag';
  let confidencePct = 25;
  if (relevantMatches >= 20) { confidence = 'hoog'; confidencePct = 85; }
  else if (relevantMatches >= 10) { confidence = 'gemiddeld'; confidencePct = 60; }
  else if (relevantMatches >= 5) { confidence = 'beperkt'; confidencePct = 40; }

  if (policyAnalysis) { confidencePct = Math.min(95, confidencePct + 10); }

  // Similar cases
  const similar = uitspraken
    .filter(u => u.type_verzekering === input.type || u.kerngeschil === input.dispute)
    .map(u => {
      let relevance = 0;
      if (u.type_verzekering === input.type) relevance += 40;
      if (u.kerngeschil === input.dispute) relevance += 40;
      if (u.beslisfactoren) {
        if (u.beslisfactoren.bewijs_consument === input.evidence) relevance += 10;
        if (u.beslisfactoren.deskundigenrapport === input.expert) relevance += 10;
      }
      return { nr: u.uitspraaknr, desc: u.samenvatting || 'Geen samenvatting', outcome: u.uitkomst, relevance: relevance + '%' };
    })
    .sort((a, b) => parseInt(b.relevance) - parseInt(a.relevance))
    .slice(0, 6);

  return { score, factors, similar, dataPoints: uitspraken.length, confidence, confidencePct, relevantMatches };
}

function renderResults(result, input) {
  const { score, factors, similar, confidence, confidencePct, relevantMatches, dataPoints } = result;
  const verdict = score >= 60 ? 'Procedure afwachten' : score <= 40 ? 'Overweeg uitkeren' : 'Onzeker \u2014 nader beoordelen';
  const verdictClass = score >= 60 ? 'afwachten' : score <= 40 ? 'uitkeren' : 'onzeker';
  const barClass = score >= 60 ? 'amber' : score <= 40 ? 'green' : 'amber';
  const confColor = confidencePct >= 60 ? 'var(--green)' : confidencePct >= 40 ? 'var(--amber)' : 'var(--red)';

  const resultsArea = document.getElementById('resultsArea');
  resultsArea.innerHTML =
    '<div class="result-hero animate-in">' +
      '<div class="verdict-label">Advies</div>' +
      '<div class="verdict-text ' + verdictClass + '">' + verdict + '</div>' +
      '<p style="color:var(--text-muted);font-size:15px;margin-top:6px;">' +
        'Score: <strong>' + score + '/100</strong> \u2014 ' + (score >= 60 ? 'hoge kans dat vordering wordt afgewezen' : score <= 40 ? 'aanzienlijk risico op (gedeeltelijke) toewijzing' : 'uitkomst onzeker op basis van beschikbare data') +
      '</p>' +
      '<div class="confidence-bar-wrap">' +
        '<div class="confidence-bar"><div class="confidence-fill ' + barClass + '" style="width: ' + score + '%"></div></div>' +
        '<div class="confidence-labels"><span>\u2190 Uitkeren</span><span>Afwachten \u2192</span></div>' +
      '</div>' +
      '<div style="display:flex;justify-content:center;gap:28px;margin-top:20px;font-size:13px;color:var(--text-dim);border-top:1px solid var(--border-subtle);padding-top:16px;">' +
        '<span>Betrouwbaarheid: <strong style="color:' + confColor + '">' + confidence + '</strong></span>' +
        '<span>Vergelijkbare zaken: <strong>' + relevantMatches + '</strong></span>' +
        '<span>Dataset: <strong>' + dataPoints + '</strong> uitspraken</span>' +
      '</div>' +
    '</div>' +

    '<div class="analysis-grid">' +
      '<div class="analysis-card animate-in delay-1">' +
        '<h3>Beslisfactoren</h3>' +
        '<ul class="factor-list">' +
          factors.map(f => '<li><span>' + f.label + '</span><span class="factor-tag ' + f.type + '">' + f.value + '</span></li>').join('') +
        '</ul>' +
      '</div>' +
      '<div class="analysis-card animate-in delay-2">' +
        '<h3>Aanbeveling</h3>' +
        '<p style="margin-bottom:14px;line-height:1.8;">' +
          (score >= 60
            ? 'Op basis van vergelijkbare uitspraken heeft de verzekeraar een sterke positie. De geschillencommissie wijst dit type vordering vaker af. Procedure afwachten lijkt verdedigbaar.'
            : score <= 40
            ? 'Er zijn significante risicofactoren. Vergelijkbare zaken worden regelmatig (deels) toegewezen. Overweeg een schikking of verhoogd coulanceaanbod om proceskosten en reputatieschade te beperken.'
            : 'De beschikbare data geeft geen eenduidig beeld. Overweeg aanvullende juridische analyse of het inwinnen van een deskundigenoordeel voordat u beslist.') +
        '</p>' +
        '<p style="font-size:12px;color:var(--text-dim);border-top:1px solid var(--border-subtle);padding-top:12px;margin-top:8px;">' +
          'Dit is een indicatief model. Raadpleeg altijd een juridisch specialist voor definitieve besluitvorming.' +
        '</p>' +
      '</div>' +
    '</div>' +

    (similar.length > 0 ?
    '<div class="similar-cases animate-in delay-3">' +
      '<h3>Vergelijkbare uitspraken</h3>' +
      similar.map(c =>
        '<div class="case-row">' +
          '<span class="case-nr">' + c.nr + '</span>' +
          '<span class="case-desc">' + c.desc + '</span>' +
          '<span class="case-outcome ' + c.outcome + '">' + c.outcome + '</span>' +
          '<span class="case-relevance">' + c.relevance + '</span>' +
        '</div>'
      ).join('') +
    '</div>' : '');
}

// ── Insights ──
function updateInsights() {
  if (uitspraken.length === 0) return;

  const types = [...new Set(uitspraken.map(u => u.type_verzekering))];
  const chartHTML = types.map(type => {
    const matches = uitspraken.filter(u => u.type_verzekering === type);
    const t = matches.filter(u => u.uitkomst === 'toegewezen').length;
    const d = matches.filter(u => u.uitkomst === 'deels').length;
    const a = matches.filter(u => u.uitkomst === 'afgewezen').length;
    const total = matches.length;
    return '<div style="margin-bottom:16px;">' +
      '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">' +
        '<span style="font-size:13px;color:var(--text);font-weight:500;">' + type + '</span>' +
        '<span style="font-size:12px;color:var(--text-dim);font-family:\'Source Code Pro\',monospace;">' + total + 'x</span>' +
      '</div>' +
      '<div style="display:flex;height:20px;border-radius:6px;overflow:hidden;background:#f1f5f9;">' +
        '<div style="width:' + (t/total*100) + '%;background:var(--green);"></div>' +
        '<div style="width:' + (d/total*100) + '%;background:var(--amber);"></div>' +
        '<div style="width:' + (a/total*100) + '%;background:var(--red);opacity:0.7;"></div>' +
      '</div></div>';
  }).join('');

  document.getElementById('insightChart').innerHTML = chartHTML +
    '<div style="display:flex;gap:20px;margin-top:14px;font-size:12px;color:var(--text-dim);">' +
      '<span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;background:var(--green);border-radius:3px;"></span> Toegewezen</span>' +
      '<span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;background:var(--amber);border-radius:3px;"></span> Deels</span>' +
      '<span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;background:var(--red);opacity:0.7;border-radius:3px;"></span> Afgewezen</span>' +
    '</div>';

  // Dispute stats
  const disputes = [...new Set(uitspraken.map(u => u.kerngeschil).filter(Boolean))];
  const disputeStats = disputes.map(d => {
    const matches = uitspraken.filter(u => u.kerngeschil === d);
    const afgewezen = matches.filter(u => u.uitkomst === 'afgewezen').length;
    return { name: d, count: matches.length, afwijzingsPct: Math.round(afgewezen / matches.length * 100) };
  }).filter(d => d.count >= 2).sort((a,b) => b.afwijzingsPct - a.afwijzingsPct);

  document.getElementById('insightPredictors').innerHTML =
    '<ul class="factor-list">' +
      disputeStats.slice(0, 5).map(d =>
        '<li><span>' + d.name + ' (n=' + d.count + ')</span><span class="factor-tag ' + (d.afwijzingsPct > 60 ? 'pro' : d.afwijzingsPct < 40 ? 'con' : 'neutral') + '">' + d.afwijzingsPct + '% afw.</span></li>'
      ).join('') +
    '</ul><p style="font-size:12px;color:var(--text-dim);margin-top:10px;">Gebaseerd op ' + uitspraken.length + ' uitspraken</p>';

  // Risk factors
  const metBeslisfactoren = uitspraken.filter(u => u.beslisfactoren);
  const riskFactors = [];

  if (metBeslisfactoren.length > 0) {
    const onduidelijk = metBeslisfactoren.filter(u => u.beslisfactoren.polisvoorwaarden_duidelijk === false);
    const onduidelijkToegewezen = onduidelijk.filter(u => u.uitkomst === 'toegewezen' || u.uitkomst === 'deels').length;
    if (onduidelijk.length > 0) riskFactors.push({ name: 'Onduidelijke polisvoorwaarden', pct: Math.round(onduidelijkToegewezen/onduidelijk.length*100), n: onduidelijk.length });

    const infoPlicht = metBeslisfactoren.filter(u => u.beslisfactoren.verzekeraar_informatieplicht_geschonden === true);
    const infoToegewezen = infoPlicht.filter(u => u.uitkomst === 'toegewezen' || u.uitkomst === 'deels').length;
    if (infoPlicht.length > 0) riskFactors.push({ name: 'Informatieplicht geschonden', pct: Math.round(infoToegewezen/infoPlicht.length*100), n: infoPlicht.length });

    const geenCoulance = metBeslisfactoren.filter(u => u.beslisfactoren.coulance_aangeboden === false);
    const geenCoulanceToegewezen = geenCoulance.filter(u => u.uitkomst === 'toegewezen' || u.uitkomst === 'deels').length;
    if (geenCoulance.length > 0) riskFactors.push({ name: 'Geen coulance aangeboden', pct: Math.round(geenCoulanceToegewezen/geenCoulance.length*100), n: geenCoulance.length });

    const sterkBewijs = metBeslisfactoren.filter(u => u.beslisfactoren.bewijs_consument === 'sterk');
    const sterkToegewezen = sterkBewijs.filter(u => u.uitkomst === 'toegewezen' || u.uitkomst === 'deels').length;
    if (sterkBewijs.length > 0) riskFactors.push({ name: 'Sterk bewijs consument', pct: Math.round(sterkToegewezen/sterkBewijs.length*100), n: sterkBewijs.length });
  }

  riskFactors.sort((a,b) => b.pct - a.pct);
  document.getElementById('insightRisks').innerHTML = riskFactors.length > 0 ?
    '<ul class="factor-list">' +
      riskFactors.map(r =>
        '<li><span>' + r.name + ' (n=' + r.n + ')</span><span class="factor-tag ' + (r.pct > 60 ? 'con' : r.pct > 40 ? 'neutral' : 'pro') + '">' + r.pct + '% toeg.</span></li>'
      ).join('') +
    '</ul><p style="font-size:12px;color:var(--text-dim);margin-top:10px;">% toewijzing wanneer deze factor aanwezig is</p>' :
    '<p style="font-size:14px;color:var(--text-dim);">Onvoldoende data met beslisfactoren.</p>';
}

// ── KIFID Lookup ──
function lookupKifid() {
  const nr = document.getElementById('kifidLookup').value.trim();
  const resultEl = document.getElementById('kifidLookupResult');

  if (!nr) {
    resultEl.innerHTML = '<p style="font-size:13px;color:var(--amber);">Voer een uitspraaknummer in.</p>';
    return;
  }

  const local = uitspraken.find(u => u.uitspraaknr === nr);
  if (local) {
    resultEl.innerHTML =
      '<div style="padding:14px 18px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);margin-top:10px;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">' +
          '<span style="font-family:\'Source Code Pro\',monospace;font-size:14px;color:var(--accent);font-weight:700;">' + local.uitspraaknr + '</span>' +
          '<span class="case-outcome ' + local.uitkomst + '">' + local.uitkomst + '</span>' +
        '</div>' +
        '<p style="font-size:14px;color:var(--text-muted);line-height:1.6;">' + local.samenvatting + '</p>' +
        '<div style="display:flex;gap:16px;margin-top:10px;font-size:12px;color:var(--text-dim);">' +
          '<span>Type: ' + local.type_verzekering + '</span>' +
          '<span>Gevorderd: \u20AC' + Number(local.bedrag_gevorderd).toLocaleString('nl-NL') + '</span>' +
          '<span>Toegewezen: \u20AC' + Number(local.bedrag_toegewezen).toLocaleString('nl-NL') + '</span>' +
        '</div>' +
      '</div>';
    return;
  }

  resultEl.innerHTML =
    '<div style="padding:14px 18px;background:var(--accent-subtle);border:1px solid var(--border);border-radius:var(--radius);margin-top:10px;">' +
      '<p style="font-size:14px;color:var(--text-muted);">Uitspraak <strong>' + nr + '</strong> niet gevonden in lokale data (' + uitspraken.length + ' uitspraken geladen).</p>' +
      '<p style="font-size:13px;color:var(--text-muted);margin-top:6px;">Zoek direct op KIFID: <a href="https://www.kifid.nl/uitspraken/" target="_blank" style="color:var(--primary);text-decoration:underline;">kifid.nl/uitspraken</a></p>' +
    '</div>';
}

// ── Policy Upload ──
async function handlePolicyUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const zone = document.getElementById('policyUploadZone');
  const resultEl = document.getElementById('policyAnalysisResult');
  const label = document.getElementById('policyUploadLabel');

  zone.style.borderColor = 'var(--primary-border)';
  zone.style.background = 'var(--primary-light)';
  label.innerHTML = '<strong>Analyseren...</strong><br><span style="font-size:12px;color:var(--text-dim);">Claude leest de polisvoorwaarden (15-30 sec)</span>';
  resultEl.innerHTML = '<div style="display:flex;align-items:center;gap:8px;padding:12px 0;font-size:13px;color:var(--text-dim);"><div class="spinner" style="width:16px;height:16px;border-width:2px;"></div> Polisvoorwaarden worden geanalyseerd...</div>';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/api/analyze-policy', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      resultEl.innerHTML = '<div style="padding:12px 16px;background:var(--red-bg);border:1px solid var(--red-border);border-radius:var(--radius);font-size:13px;color:var(--red);">' + data.error + (data.details ? ': ' + data.details : '') + '</div>';
      zone.style.borderColor = 'var(--border)';
      zone.style.background = '';
      label.innerHTML = '<strong>Klik om opnieuw te uploaden</strong>';
      return;
    }

    policyAnalysis = data;

    zone.style.borderColor = 'var(--green-border)';
    zone.style.background = 'var(--green-bg)';
    label.innerHTML = '<strong style="color:var(--green);">' + file.name + '</strong><br><span style="font-size:12px;color:var(--green);">Analyse compleet \u2014 risicoscore: ' + data.risicoscore + '/10</span>';

    const clausules = (data.risicovolle_clausules || []).slice(0, 4);
    resultEl.innerHTML =
      '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);padding:18px;margin-top:10px;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">' +
          '<strong style="font-size:14px;">' + (data.product_naam || data.type_verzekering || 'Polisvoorwaarden') + '</strong>' +
          '<span style="font-family:\'Source Code Pro\',monospace;font-size:12px;padding:4px 12px;border-radius:5px;font-weight:600;' +
            (data.risicoscore >= 7 ? 'background:var(--red-bg);color:var(--red);' :
             data.risicoscore >= 4 ? 'background:var(--amber-bg);color:var(--amber);' :
             'background:var(--green-bg);color:var(--green);') +
          '">Risico: ' + data.risicoscore + '/10</span>' +
        '</div>' +
        '<p style="font-size:13px;color:var(--text-muted);line-height:1.7;margin-bottom:12px;">' + (data.samenvatting || '') + '</p>' +
        (clausules.length > 0 ?
          '<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-dim);margin-bottom:8px;font-weight:600;">Risicovolle clausules</div>' +
          '<ul style="list-style:none;margin:0;padding:0;">' +
            clausules.map(c =>
              '<li style="font-size:13px;color:var(--text-secondary);padding:7px 0;border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;gap:8px;">' +
                '<span>' + (c.artikel ? '<code style="font-size:11px;">Art. ' + c.artikel + '</code> ' : '') + c.clausule + '</span>' +
                '<span style="font-size:11px;padding:2px 8px;border-radius:4px;white-space:nowrap;font-weight:500;' +
                  (c.ernst === 'hoog' ? 'background:var(--red-bg);color:var(--red);' :
                   c.ernst === 'middel' ? 'background:var(--amber-bg);color:var(--amber);' :
                   'background:var(--green-bg);color:var(--green);') +
                '">' + c.ernst + '</span>' +
              '</li>'
            ).join('') +
          '</ul>' : '') +
        ((data.aanbevelingen || []).length > 0 ?
          '<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-dim);margin-top:14px;margin-bottom:8px;font-weight:600;">Aanbevelingen</div>' +
          '<ul style="margin:0;padding-left:18px;">' +
            data.aanbevelingen.slice(0, 3).map(a => '<li style="font-size:13px;color:var(--text-muted);line-height:1.7;margin-bottom:3px;">' + a + '</li>').join('') +
          '</ul>' : '') +
      '</div>';

    const typeSelect = document.getElementById('insuranceType');
    if (!typeSelect.value && data.type_verzekering) {
      const option = [...typeSelect.options].find(o => o.value === data.type_verzekering);
      if (option) typeSelect.value = option.value;
    }

  } catch (err) {
    resultEl.innerHTML = '<div style="padding:12px 16px;background:var(--red-bg);border:1px solid var(--red-border);border-radius:var(--radius);font-size:13px;color:var(--red);">Verbinding met server mislukt. Start de server: python3 scripts/server.py</div>';
    zone.style.borderColor = 'var(--border)';
    zone.style.background = '';
    label.innerHTML = '<strong>Klik om opnieuw te uploaden</strong>';
  }
}

// ── Drag & Drop ──
document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('uploadZone');
  if (uploadZone) {
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => { uploadZone.classList.remove('drag-over'); });
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      const fileInput = document.getElementById('fileInput');
      fileInput.files = e.dataTransfer.files;
      handleFileUpload({ target: fileInput });
    });
  }
});
