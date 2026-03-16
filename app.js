// ── KIFID Predictor ──

let uitspraken = [];
let currentTab = 'predict';
let policyAnalysis = null;

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  autoLoadDataset();
  initCountAnimations();
  initTheme();
  initDragDrop();
});

// ── Theme ──
function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.setAttribute('data-theme', 'dark');
    updateThemeIcons(true);
  }
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  if (isDark) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('theme', 'light');
    updateThemeIcons(false);
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    localStorage.setItem('theme', 'dark');
    updateThemeIcons(true);
  }
}

function updateThemeIcons(isDark) {
  var sun = document.getElementById('iconSun');
  var moon = document.getElementById('iconMoon');
  if (sun) sun.style.display = isDark ? 'none' : 'block';
  if (moon) moon.style.display = isDark ? 'block' : 'none';
}

// ── Count-up Animations ──
function initCountAnimations() {
  var elements = document.querySelectorAll('[data-count]');
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting && !entry.target.dataset.counted) {
        entry.target.dataset.counted = 'true';
        animateCount(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  elements.forEach(function(el) { observer.observe(el); });
}

function animateCount(el) {
  var target = parseInt(el.dataset.count);
  var suffix = el.dataset.suffix || '';
  var prefix = el.dataset.prefix || '';
  var duration = target > 10000 ? 2200 : target > 100 ? 1600 : 1200;
  var start = performance.now();

  function ease(t) {
    return t === 1 ? 1 : 1 - Math.pow(2, -12 * t);
  }

  function update(now) {
    var elapsed = now - start;
    var progress = Math.min(elapsed / duration, 1);
    var current = Math.round(target * ease(progress));
    var formatted = target > 999 ? current.toLocaleString('nl-NL') : current;
    el.textContent = prefix + formatted + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

// ── Scroll ──
function scrollToSection(id) {
  var el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Tab Switching ──
function switchTab(tab) {
  currentTab = tab;
  var tabs = ['predict', 'data', 'insights'];
  var btns = document.querySelectorAll('.tab-btn');
  btns.forEach(function(btn, i) {
    btn.classList.toggle('active', tabs[i] === tab);
  });
  document.getElementById('tab-predict').style.display = tab === 'predict' ? 'grid' : 'none';
  document.getElementById('tab-data').style.display = tab === 'data' ? 'block' : 'none';
  document.getElementById('tab-insights').style.display = tab === 'insights' ? 'block' : 'none';
}

// ── Auto-load Dataset ──
async function autoLoadDataset() {
  try {
    var response = await fetch('data/uitspraken/dataset.json');
    if (!response.ok) return;
    var data = await response.json();
    uitspraken = data.uitspraken || [];
    if (uitspraken.length === 0) return;
    updateInsights();
    updateHeroStats();
    var statusEl = document.getElementById('dataStatus');
    if (statusEl) {
      statusEl.className = 'data-status';
      statusEl.textContent = uitspraken.length + ' uitspraken geladen.';
    }
    var uploadedEl = document.getElementById('uploadedFiles');
    if (uploadedEl) {
      uploadedEl.innerHTML = '<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;">Dataset geladen \u2014 ' + uitspraken.length + ' uitspraken</div>';
    }
    var loadBtn = document.querySelector('[onclick="loadFullDataset()"]');
    if (loadBtn) loadBtn.textContent = 'Herladen (' + uitspraken.length + ')';
  } catch (e) {}
}

function updateHeroStats() {
  if (uitspraken.length === 0) return;
  var total = uitspraken.length;
  var toegewezen = uitspraken.filter(function(u) { return u.uitkomst === 'toegewezen'; }).length;
  var deels = uitspraken.filter(function(u) { return u.uitkomst === 'deels'; }).length;
  var afgewezen = uitspraken.filter(function(u) { return u.uitkomst === 'afgewezen'; }).length;
  var consumerWin = Math.round((toegewezen + deels) / total * 100);
  var insurerWin = Math.round(afgewezen / total * 100);

  // Update metric elements that already have data-count
  var metrics = document.querySelectorAll('.metric-value');
  metrics.forEach(function(m) {
    if (m.id === 'heroStatConsumer') {
      m.dataset.count = consumerWin;
      m.textContent = consumerWin + '%';
    } else if (m.id === 'heroStatInsurer') {
      m.dataset.count = insurerWin;
      m.textContent = insurerWin + '%';
    }
  });
  // Update first metric (total analyzed)
  var firstMetric = document.querySelector('.metric-value[data-count]');
  if (firstMetric && !firstMetric.id) {
    firstMetric.dataset.count = total;
    firstMetric.textContent = total + '+';
  }
}

// ── File Upload ──
function handleFileUpload(event) {
  var files = event.target.files;
  for (var i = 0; i < files.length; i++) {
    var file = files[i];
    if (file.name.endsWith('.csv')) readCSV(file);
    else if (file.name.endsWith('.json')) readJSON(file);
  }
}

function readCSV(file) {
  var reader = new FileReader();
  reader.onload = function(e) {
    var lines = e.target.result.split('\n');
    var headers = lines[0].split(/[,;]/).map(function(h) { return h.trim().toLowerCase(); });
    for (var i = 1; i < lines.length; i++) {
      var vals = lines[i].split(/[,;]/);
      if (vals.length < 3) continue;
      var row = {};
      headers.forEach(function(h, j) { row[h] = vals[j] ? vals[j].trim() : ''; });
      uitspraken.push(row);
    }
    showUploadSuccess(file.name, lines.length - 1);
  };
  reader.readAsText(file);
}

function readJSON(file) {
  var reader = new FileReader();
  reader.onload = function(e) {
    try {
      var data = JSON.parse(e.target.result);
      var items = Array.isArray(data) ? data : data.uitspraken || [];
      uitspraken = uitspraken.concat(items);
      showUploadSuccess(file.name, items.length);
    } catch (err) {
      alert('Fout bij lezen JSON: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function showUploadSuccess(name, count) {
  var el = document.getElementById('uploadedFiles');
  el.innerHTML += '<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;margin-bottom:8px;">' + name + ' \u2014 ' + count + ' uitspraken</div>';
  document.getElementById('dataStatus').className = 'data-status';
  document.getElementById('dataStatus').textContent = uitspraken.length + ' uitspraken geladen.';
  updateInsights();
  updateHeroStats();
}

async function loadFullDataset() {
  try {
    var response = await fetch('data/uitspraken/dataset.json');
    var data = await response.json();
    uitspraken = data.uitspraken || [];
    document.getElementById('dataStatus').className = 'data-status';
    document.getElementById('dataStatus').textContent = uitspraken.length + ' uitspraken geladen.';
    document.getElementById('uploadedFiles').innerHTML = '<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;">Dataset \u2014 ' + uitspraken.length + ' uitspraken</div>';
    updateInsights();
    updateHeroStats();
  } catch (err) {
    alert('Dataset laden mislukt: ' + err.message);
  }
}

function loadDemoData() {
  uitspraken = [
    { uitspraaknr: "2024-0500", datum: "2024-06-15", type_verzekering: "autoverzekering", kerngeschil: "schadevrije_jaren", uitkomst: "afgewezen", bedrag_gevorderd: 2400, bedrag_toegewezen: 0, samenvatting: "Schadevrije jaren gekoppeld aan verzekeringnemer, niet aan bestuurder." },
    { uitspraaknr: "2024-1029", datum: "2024-11-20", type_verzekering: "woonhuisverzekering", kerngeschil: "uitleg_voorwaarden", uitkomst: "toegewezen", bedrag_gevorderd: 3980, bedrag_toegewezen: 3980, samenvatting: "Vervangingskosten vallen niet onder reparatiekosten-uitsluiting." },
    { uitspraaknr: "2024-1005", datum: "2024-11-15", type_verzekering: "woonhuisverzekering", kerngeschil: "dekkingsweigering", uitkomst: "afgewezen", bedrag_gevorderd: 75000, bedrag_toegewezen: 10000, samenvatting: "Neerslagcriterium KNMI niet gehaald, coulance 10k uitgekeerd." },
    { uitspraaknr: "2024-0861", datum: "2024-09-10", type_verzekering: "overlijdensrisicoverzekering", kerngeschil: "mededelingsplicht", uitkomst: "afgewezen", bedrag_gevorderd: 300000, bedrag_toegewezen: 0, samenvatting: "Fout adres opgegeven, verzekering nooit tot stand gekomen." },
    { uitspraaknr: "2024-0660", datum: "2024-07-22", type_verzekering: "autoverzekering", kerngeschil: "premiegeschil", uitkomst: "afgewezen", bedrag_gevorderd: 1600, bedrag_toegewezen: 0, samenvatting: "Geen rechtsgrond voor vordering na opzegging door consument zelf." },
    { uitspraaknr: "2024-0024", datum: "2024-01-18", type_verzekering: "rechtsbijstandverzekering", kerngeschil: "dekkingsweigering", uitkomst: "afgewezen", bedrag_gevorderd: 8000, bedrag_toegewezen: 0, samenvatting: "Beleggingsfraude geen gedekte gebeurtenis." },
    { uitspraaknr: "2024-0069", datum: "2024-02-28", type_verzekering: "beleggingsverzekering", kerngeschil: "zorgplicht", uitkomst: "deels", bedrag_gevorderd: 12000, bedrag_toegewezen: 4215, samenvatting: "Adviseur tekortgeschoten in nazorgverplichting." },
    { uitspraaknr: "2023-0842", datum: "2023-10-15", type_verzekering: "woonhuisverzekering", kerngeschil: "dekkingsweigering", uitkomst: "toegewezen", bedrag_gevorderd: 22000, bedrag_toegewezen: 18500, samenvatting: "Stormschade gedekt, uitsluiting te ruim geinterpreteerd." },
    { uitspraaknr: "2023-0631", datum: "2023-08-02", type_verzekering: "reisverzekering", kerngeschil: "dekkingsweigering", uitkomst: "toegewezen", bedrag_gevorderd: 4500, bedrag_toegewezen: 4500, samenvatting: "Annulering wegens ziekte gedekt." },
  ];
  document.getElementById('dataStatus').className = 'data-status';
  document.getElementById('dataStatus').textContent = uitspraken.length + ' demo-uitspraken geladen.';
  document.getElementById('uploadedFiles').innerHTML = '<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);font-size:13px;color:var(--green);font-weight:500;">Demo \u2014 ' + uitspraken.length + ' uitspraken</div>';
  updateInsights();
  updateHeroStats();
}

// ── Prediction ──
function runPrediction() {
  var btn = document.getElementById('btnPredict');
  btn.disabled = true;
  btn.textContent = 'Analyseren...';

  var area = document.getElementById('resultsArea');
  area.innerHTML = '<div class="result-hero"><div class="loading"><div class="spinner"></div><div class="loading-text">Analyseren van ' + uitspraken.length + ' uitspraken...</div></div></div>';

  var input = {
    type: document.getElementById('insuranceType').value,
    dispute: document.getElementById('coreDispute').value,
    amount: parseFloat(document.getElementById('claimAmount').value) || 0,
    binding: document.getElementById('bindingAdvice').value,
    evidence: document.getElementById('consumerEvidence').value,
    expert: document.getElementById('expertReport').value,
    goodwill: document.getElementById('goodwillOffer').value,
    context: document.getElementById('additionalContext').value,
  };

  setTimeout(function() {
    var result = analyzeCase(input);
    renderResults(result, input);
    btn.disabled = false;
    btn.textContent = 'Analyseer & Voorspel';
  }, 1200);
}

// ── Helper: compute afwijzingspercentage for a subset ──
function afwPct(subset) {
  if (subset.length === 0) return 50;
  return Math.round(subset.filter(function(u) { return u.uitkomst === 'afgewezen'; }).length / subset.length * 100);
}

// ── Helper: format score delta for display ──
function fmtDelta(val) { return (val >= 0 ? '+' : '') + val; }

function analyzeCase(input) {
  var score = 50;
  var factors = [];
  var bfAll = uitspraken.filter(function(u) { return u.beslisfactoren; });
  var overallAfwRate = afwPct(uitspraken);

  // ── 1. Base rate: verzekeringtype ──
  var typeMatches = uitspraken.filter(function(u) { return u.type_verzekering === input.type; });
  if (typeMatches.length > 0) {
    var typeRate = afwPct(typeMatches);
    score = typeRate;
    factors.push({ label: 'Afwijzingspercentage ' + input.type + ' (n=' + typeMatches.length + ')', value: typeRate + '%', type: typeRate > 60 ? 'pro' : 'con' });
  }

  // ── 2. Kerngeschil correctie ──
  var disputeMatches = uitspraken.filter(function(u) { return u.kerngeschil === input.dispute; });
  if (disputeMatches.length > 0) {
    var dRate = afwPct(disputeMatches);
    var adj = Math.round((dRate - 50) * 0.3);
    score += adj;
    factors.push({ label: 'Kerngeschil ' + input.dispute + ' (' + dRate + '% afw, n=' + disputeMatches.length + ')', value: fmtDelta(adj), type: adj > 0 ? 'pro' : adj < 0 ? 'con' : 'neutral' });
  }

  // ── 3. Combinatie type + geschil (meest specifiek) ──
  var combiMatches = uitspraken.filter(function(u) { return u.type_verzekering === input.type && u.kerngeschil === input.dispute; });
  if (combiMatches.length >= 3) {
    var combiRate = afwPct(combiMatches);
    var combiAdj = Math.round((combiRate - score) * 0.4);
    score += combiAdj;
    factors.push({ label: 'Specifiek ' + input.type + ' + ' + input.dispute + ' (n=' + combiMatches.length + ')', value: fmtDelta(combiAdj), type: combiAdj > 0 ? 'pro' : combiAdj < 0 ? 'con' : 'neutral' });
  }

  // ── 4. Beslisfactoren: bewijs consument (data-driven) ──
  if (input.evidence && bfAll.length >= 5) {
    var sameEvidence = bfAll.filter(function(u) { return u.beslisfactoren.bewijs_consument === input.evidence; });
    if (sameEvidence.length >= 3) {
      var evRate = afwPct(sameEvidence);
      var evDelta = Math.round((evRate - overallAfwRate) * 0.4);
      score += evDelta;
      var evLabel = 'Bewijs \u201C' + input.evidence + '\u201D \u2192 ' + evRate + '% afw. (n=' + sameEvidence.length + ')';
      factors.push({ label: evLabel, value: fmtDelta(evDelta), type: evDelta > 5 ? 'pro' : evDelta < -5 ? 'con' : 'neutral' });
    } else {
      // Fallback to heuristic if not enough data
      var evFallback = input.evidence === 'sterk' ? -12 : input.evidence === 'zwak' ? 12 : 0;
      if (evFallback !== 0) {
        score += evFallback;
        factors.push({ label: 'Bewijs consument: ' + input.evidence, value: fmtDelta(evFallback), type: evFallback > 0 ? 'pro' : 'con' });
      }
    }
  }

  // ── 5. Beslisfactoren: deskundigenrapport (data-driven) ──
  if (input.expert && input.expert !== 'geen' && bfAll.length >= 5) {
    var sameExpert = bfAll.filter(function(u) { return u.beslisfactoren.deskundigenrapport === input.expert; });
    if (sameExpert.length >= 3) {
      var exRate = afwPct(sameExpert);
      var exDelta = Math.round((exRate - overallAfwRate) * 0.35);
      score += exDelta;
      factors.push({ label: 'Deskundigenrapport \u201C' + input.expert + '\u201D \u2192 ' + exRate + '% afw. (n=' + sameExpert.length + ')', value: fmtDelta(exDelta), type: exDelta > 3 ? 'pro' : exDelta < -3 ? 'con' : 'neutral' });
    } else {
      var exFallback = input.expert === 'verzekeraar' ? 8 : input.expert === 'consument' ? -8 : -2;
      score += exFallback;
      factors.push({ label: 'Deskundigenrapport: ' + input.expert, value: fmtDelta(exFallback), type: exFallback > 0 ? 'pro' : 'con' });
    }
  }

  // ── 6. Beslisfactoren: polisvoorwaarden duidelijkheid (data-driven) ──
  if (bfAll.length >= 5) {
    var onduidelijk = bfAll.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === false; });
    var duidelijk = bfAll.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === true; });
    if (onduidelijk.length >= 3 && duidelijk.length >= 3) {
      var ondRate = afwPct(onduidelijk);
      var duiRate = afwPct(duidelijk);
      var verschil = duiRate - ondRate;
      if (Math.abs(verschil) >= 10) {
        factors.push({ label: 'Onduidelijke voorwaarden \u2192 ' + ondRate + '% afw. vs. duidelijk ' + duiRate + '% (n=' + onduidelijk.length + '/' + duidelijk.length + ')', value: '\u0394' + Math.abs(verschil) + '%', type: verschil > 0 ? 'con' : 'pro' });
      }
    }
  }

  // ── 7. Beslisfactoren: consument nalatig (data-driven) ──
  if (bfAll.length >= 5) {
    var nalatig = bfAll.filter(function(u) { return u.beslisfactoren.consument_nalatig === true; });
    if (nalatig.length >= 3) {
      var nalRate = afwPct(nalatig);
      var nalDelta = Math.round((nalRate - overallAfwRate) * 0.25);
      if (Math.abs(nalDelta) >= 3) {
        factors.push({ label: 'Nalatigheid consument \u2192 ' + nalRate + '% afw. (n=' + nalatig.length + ')', value: fmtDelta(nalDelta), type: 'neutral' });
      }
    }
  }

  // ── 8. Beslisfactoren: informatieplicht geschonden (data-driven) ──
  if (bfAll.length >= 5) {
    var infoSchending = bfAll.filter(function(u) { return u.beslisfactoren.verzekeraar_informatieplicht_geschonden === true; });
    if (infoSchending.length >= 2) {
      var infoRate = afwPct(infoSchending);
      var infoDelta = Math.round((infoRate - overallAfwRate) * 0.3);
      score += infoDelta;
      factors.push({ label: 'Informatieplicht geschonden \u2192 ' + infoRate + '% afw. (n=' + infoSchending.length + ')', value: fmtDelta(infoDelta), type: infoDelta < -3 ? 'con' : infoDelta > 3 ? 'pro' : 'neutral' });
    }
  }

  // ── 9. Coulance (data-driven + input) ──
  if (bfAll.length >= 5) {
    var coulYes = bfAll.filter(function(u) { return u.beslisfactoren.coulance_aangeboden === true; });
    var coulNo = bfAll.filter(function(u) { return u.beslisfactoren.coulance_aangeboden === false; });
    if (coulYes.length >= 3 && coulNo.length >= 3) {
      var coulYesRate = afwPct(coulYes);
      var coulNoRate = afwPct(coulNo);
      if (input.goodwill === 'ja_redelijk' || input.goodwill === 'ja_laag') {
        var coulAdj = Math.round((coulYesRate - overallAfwRate) * 0.3);
        score += coulAdj;
        factors.push({ label: 'Met coulance \u2192 ' + coulYesRate + '% afw. vs. zonder ' + coulNoRate + '% (n=' + coulYes.length + '/' + coulNo.length + ')', value: fmtDelta(coulAdj), type: coulAdj > 3 ? 'pro' : coulAdj < -3 ? 'con' : 'neutral' });
      } else {
        var noCoulAdj = Math.round((coulNoRate - overallAfwRate) * 0.2);
        if (Math.abs(noCoulAdj) >= 3) {
          score += noCoulAdj;
          factors.push({ label: 'Geen coulance \u2192 ' + coulNoRate + '% afw. (n=' + coulNo.length + ')', value: fmtDelta(noCoulAdj), type: noCoulAdj > 3 ? 'pro' : 'con' });
        }
      }
    }
  }

  // ── 10. Bedragrange-analyse ──
  if (input.amount > 0) {
    var lo = input.amount * 0.3;
    var hi = input.amount * 3;
    var amountPeers = uitspraken.filter(function(u) { var b = u.bedrag_gevorderd || 0; return b >= lo && b <= hi && b > 0; });
    if (amountPeers.length >= 5) {
      var amtRate = afwPct(amountPeers);
      var amtAdj = Math.round((amtRate - overallAfwRate) * 0.2);
      if (Math.abs(amtAdj) >= 2) {
        score += amtAdj;
        factors.push({ label: 'Bedragrange \u20AC' + Math.round(lo).toLocaleString('nl-NL') + '\u2013' + Math.round(hi).toLocaleString('nl-NL') + ' (' + amtRate + '% afw, n=' + amountPeers.length + ')', value: fmtDelta(amtAdj), type: amtAdj > 3 ? 'pro' : amtAdj < -3 ? 'con' : 'neutral' });
      }
    }
  }

  // ── 11. Juridische grondslag & context-matching ──
  if (input.context && input.context.length > 3) {
    var keywords = input.context.toLowerCase().split(/[\s,;.:()\-]+/).filter(function(w) { return w.length > 3; });
    if (keywords.length > 0) {
      var contextMatched = uitspraken.filter(function(u) {
        var haystack = ((u.juridische_grondslag || []).join(' ') + ' ' + (u.tags || []).join(' ') + ' ' + (u.samenvatting || '') + ' ' + (u.argumenten_consument || []).join(' ') + ' ' + (u.argumenten_verzekeraar || []).join(' ')).toLowerCase();
        var hits = 0;
        keywords.forEach(function(kw) { if (haystack.indexOf(kw) !== -1) hits++; });
        return hits >= Math.max(1, Math.floor(keywords.length * 0.3));
      });
      if (contextMatched.length >= 2) {
        var ctxRate = afwPct(contextMatched);
        var ctxAdj = Math.round((ctxRate - overallAfwRate) * 0.25);
        score += ctxAdj;
        var topGrondslagen = findTopGrondslagen(contextMatched);
        factors.push({ label: 'Contexmatch: ' + contextMatched.length + ' zaken' + (topGrondslagen ? ' (' + topGrondslagen + ')' : '') + ' \u2192 ' + ctxRate + '% afw.', value: fmtDelta(ctxAdj), type: ctxAdj > 3 ? 'pro' : ctxAdj < -3 ? 'con' : 'neutral' });
      }
    }
  }

  // ── 12. Bindend advies info ──
  if (input.binding === 'bindend') {
    var bindend = uitspraken.filter(function(u) { return u.bindend === true; });
    var nietBindend = uitspraken.filter(function(u) { return u.bindend === false; });
    if (bindend.length >= 3 && nietBindend.length >= 3) {
      var bRate = afwPct(bindend);
      var nbRate = afwPct(nietBindend);
      if (Math.abs(bRate - nbRate) >= 5) {
        var bAdj = Math.round((bRate - nbRate) * 0.15);
        score += bAdj;
        factors.push({ label: 'Bindend advies (' + bRate + '% afw.) vs. niet-bindend (' + nbRate + '%)', value: fmtDelta(bAdj), type: 'neutral' });
      } else {
        factors.push({ label: 'Bindend advies', value: 'info', type: 'neutral' });
      }
    }
  }

  // ── 13. Polisvoorwaarden-upload analyse ──
  if (policyAnalysis) {
    var rs = policyAnalysis.risicoscore || 0;
    if (rs >= 7) {
      var pa = -Math.round((rs - 5) * 3);
      score += pa;
      factors.push({ label: 'Polisvoorwaarden risico ' + rs + '/10', value: String(pa), type: 'con' });
    } else if (rs >= 4) {
      var pa2 = -Math.round((rs - 5) * 2);
      score += pa2;
      factors.push({ label: 'Polisvoorwaarden risico ' + rs + '/10', value: fmtDelta(pa2), type: 'neutral' });
    } else if (rs > 0) {
      var pa3 = Math.round((5 - rs) * 2);
      score += pa3;
      factors.push({ label: 'Polisvoorwaarden risico ' + rs + '/10', value: '+' + pa3, type: 'pro' });
    }
    var hoog = (policyAnalysis.risicovolle_clausules || []).filter(function(c) { return c.ernst === 'hoog'; }).length;
    if (hoog >= 3) {
      score -= 5;
      factors.push({ label: hoog + ' hoog-risico clausules', value: '-5', type: 'con' });
    }
  }

  score = Math.max(5, Math.min(95, score));

  // ── Betrouwbaarheid ──
  var relevantMatches = uitspraken.filter(function(u) { return u.type_verzekering === input.type || u.kerngeschil === input.dispute; }).length;
  var bfRelevant = bfAll.filter(function(u) { return u.type_verzekering === input.type || u.kerngeschil === input.dispute; }).length;
  var confidence = 'laag';
  var confidencePct = 20;
  // Meer data = meer betrouwbaar
  if (relevantMatches >= 30) { confidence = 'hoog'; confidencePct = 85; }
  else if (relevantMatches >= 15) { confidence = 'hoog'; confidencePct = 75; }
  else if (relevantMatches >= 8) { confidence = 'gemiddeld'; confidencePct = 60; }
  else if (relevantMatches >= 4) { confidence = 'beperkt'; confidencePct = 40; }
  // Bonus voor beschikbare beslisfactoren
  if (bfRelevant >= 10) confidencePct = Math.min(95, confidencePct + 10);
  else if (bfRelevant >= 5) confidencePct = Math.min(95, confidencePct + 5);
  // Bonus voor type+geschil combo
  if (combiMatches.length >= 5) confidencePct = Math.min(95, confidencePct + 8);
  if (policyAnalysis) confidencePct = Math.min(95, confidencePct + 10);

  // ── Vergelijkbare uitspraken (verbeterde relevantie) ──
  var similar = uitspraken
    .filter(function(u) { return u.type_verzekering === input.type || u.kerngeschil === input.dispute; })
    .map(function(u) {
      var rel = 0;
      // Type en geschil matches
      if (u.type_verzekering === input.type) rel += 25;
      if (u.kerngeschil === input.dispute) rel += 25;
      if (u.type_verzekering === input.type && u.kerngeschil === input.dispute) rel += 15; // combi-bonus
      // Beslisfactoren matches
      if (u.beslisfactoren) {
        if (u.beslisfactoren.bewijs_consument === input.evidence) rel += 10;
        if (u.beslisfactoren.deskundigenrapport === input.expert) rel += 8;
        if (input.goodwill !== 'nee' && u.beslisfactoren.coulance_aangeboden === true) rel += 5;
        if (input.goodwill === 'nee' && u.beslisfactoren.coulance_aangeboden === false) rel += 5;
      }
      // Bedrag proximity
      if (input.amount > 0 && u.bedrag_gevorderd > 0) {
        var ratio = Math.min(input.amount, u.bedrag_gevorderd) / Math.max(input.amount, u.bedrag_gevorderd);
        rel += Math.round(ratio * 7);
      }
      // Context/juridische grondslag match
      if (input.context && input.context.length > 3) {
        var ctx = input.context.toLowerCase();
        var uText = ((u.juridische_grondslag || []).join(' ') + ' ' + (u.tags || []).join(' ') + ' ' + (u.samenvatting || '')).toLowerCase();
        var ctxWords = ctx.split(/[\s,;.:()\-]+/).filter(function(w) { return w.length > 3; });
        var ctxHits = 0;
        ctxWords.forEach(function(w) { if (uText.indexOf(w) !== -1) ctxHits++; });
        if (ctxWords.length > 0) rel += Math.round(ctxHits / ctxWords.length * 10);
      }
      rel = Math.min(100, rel);
      return { nr: u.uitspraaknr, desc: u.samenvatting || '', outcome: u.uitkomst, relevance: rel + '%', pdfUrl: u.bron_url || '', grondslag: (u.juridische_grondslag || []).slice(0, 2).join(', ') };
    })
    .sort(function(a, b) { return parseInt(b.relevance) - parseInt(a.relevance); })
    .slice(0, 8);

  return { score: score, factors: factors, similar: similar, dataPoints: uitspraken.length, confidence: confidence, confidencePct: confidencePct, relevantMatches: relevantMatches };
}

// ── Helper: vind meest voorkomende juridische grondslagen in een set uitspraken ──
function findTopGrondslagen(matches) {
  var counts = {};
  matches.forEach(function(u) {
    (u.juridische_grondslag || []).forEach(function(g) {
      counts[g] = (counts[g] || 0) + 1;
    });
  });
  var sorted = Object.keys(counts).sort(function(a, b) { return counts[b] - counts[a]; });
  return sorted.slice(0, 2).join(', ');
}

function renderResults(result, input) {
  var s = result.score, f = result.factors, sim = result.similar;
  var verdict = s >= 60 ? 'Procedure afwachten' : s <= 40 ? 'Overweeg uitkeren' : 'Onzeker \u2014 nader beoordelen';
  var vc = s >= 60 ? 'afwachten' : s <= 40 ? 'uitkeren' : 'onzeker';
  var bc = s >= 60 ? 'amber' : s <= 40 ? 'green' : 'amber';
  var cc = result.confidencePct >= 60 ? 'var(--green)' : result.confidencePct >= 40 ? 'var(--amber)' : 'var(--red)';

  var html =
    '<div class="result-hero animate-in">' +
      '<div class="verdict-label">Advies</div>' +
      '<div class="verdict-text ' + vc + '">' + verdict + '</div>' +
      '<p style="color:var(--text-muted);font-size:15px;margin-top:8px;">Score: <strong>' + s + '/100</strong> \u2014 ' +
        (s >= 60 ? 'hoge kans dat vordering wordt afgewezen' : s <= 40 ? 'aanzienlijk risico op (gedeeltelijke) toewijzing' : 'uitkomst onzeker') +
      '</p>' +
      '<div class="confidence-bar-wrap">' +
        '<div class="confidence-bar"><div class="confidence-fill ' + bc + '" style="width:' + s + '%"></div></div>' +
        '<div class="confidence-labels"><span>\u2190 Uitkeren</span><span>Afwachten \u2192</span></div>' +
      '</div>' +
      '<div style="display:flex;justify-content:center;gap:32px;margin-top:24px;font-size:13px;color:var(--text-dim);border-top:1px solid var(--border-subtle);padding-top:18px;">' +
        '<span>Betrouwbaarheid: <strong style="color:' + cc + '">' + result.confidence + '</strong></span>' +
        '<span>Vergelijkbaar: <strong>' + result.relevantMatches + '</strong></span>' +
        '<span>Dataset: <strong>' + result.dataPoints + '</strong></span>' +
      '</div>' +
    '</div>';

  html += '<div class="analysis-grid">' +
    '<div class="analysis-card animate-in delay-1"><h3>Beslisfactoren</h3><ul class="factor-list">' +
    f.map(function(x) { return '<li><span>' + x.label + '</span><span class="factor-tag ' + x.type + '">' + x.value + '</span></li>'; }).join('') +
    '</ul></div>' +
    '<div class="analysis-card animate-in delay-2"><h3>Aanbeveling</h3><p style="line-height:1.8;">' +
    (s >= 60 ? 'Op basis van vergelijkbare uitspraken heeft de verzekeraar een sterke positie. Procedure afwachten lijkt verdedigbaar.' :
     s <= 40 ? 'Significante risicofactoren aanwezig. Overweeg een schikking of verhoogd coulanceaanbod om proceskosten te beperken.' :
     'Geen eenduidig beeld. Overweeg aanvullende juridische analyse voordat u beslist.') +
    '</p><p style="font-size:12px;color:var(--text-dim);border-top:1px solid var(--border-subtle);padding-top:14px;margin-top:14px;">Indicatief model. Raadpleeg een specialist voor definitieve besluitvorming.</p></div></div>';

  if (sim.length > 0) {
    html += '<div class="similar-cases animate-in delay-3"><h3>Vergelijkbare uitspraken</h3>' +
      sim.map(function(c) {
        var kifidUrl = c.pdfUrl || ('https://www.kifid.nl/kifid-kennis-en-uitspraken/uitspraken/?SearchTerm=' + encodeURIComponent('uitspraak-' + c.nr));
        var grondslagHtml = c.grondslag ? '<span class="case-grondslag" style="font-size:11px;color:var(--text-dim);display:block;margin-top:2px;">' + c.grondslag + '</span>' : '';
        return '<a href="' + kifidUrl + '" target="_blank" rel="noopener" class="case-row case-row-link"><span class="case-nr">' + c.nr + '</span><span class="case-desc">' + c.desc + grondslagHtml + '</span><span class="case-outcome ' + c.outcome + '">' + c.outcome + '</span><span class="case-relevance">' + c.relevance + ' <svg style="width:14px;height:14px;vertical-align:middle;opacity:0.5;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></span></a>';
      }).join('') + '</div>';
  }

  document.getElementById('resultsArea').innerHTML = html;
}

// ── Insights ──
function updateInsights() {
  if (uitspraken.length === 0) return;

  var types = [];
  uitspraken.forEach(function(u) { if (types.indexOf(u.type_verzekering) === -1) types.push(u.type_verzekering); });

  var chartHTML = types.map(function(type) {
    var m = uitspraken.filter(function(u) { return u.type_verzekering === type; });
    var t = m.filter(function(u) { return u.uitkomst === 'toegewezen'; }).length;
    var d = m.filter(function(u) { return u.uitkomst === 'deels'; }).length;
    var a = m.filter(function(u) { return u.uitkomst === 'afgewezen'; }).length;
    var tot = m.length;
    return '<div style="margin-bottom:16px;">' +
      '<div style="display:flex;justify-content:space-between;margin-bottom:6px;">' +
        '<span style="font-size:13px;font-weight:600;">' + type + '</span>' +
        '<span style="font-size:12px;color:var(--text-dim);font-variant-numeric:tabular-nums;">' + tot + 'x</span>' +
      '</div>' +
      '<div style="display:flex;height:20px;border-radius:6px;overflow:hidden;background:var(--bg-alt);">' +
        '<div style="width:' + (t/tot*100) + '%;background:var(--green);transition:width 0.8s ease;"></div>' +
        '<div style="width:' + (d/tot*100) + '%;background:var(--amber);transition:width 0.8s ease;"></div>' +
        '<div style="width:' + (a/tot*100) + '%;background:var(--red);opacity:0.8;transition:width 0.8s ease;"></div>' +
      '</div></div>';
  }).join('');

  document.getElementById('insightChart').innerHTML = chartHTML +
    '<div style="display:flex;gap:24px;margin-top:16px;font-size:12px;color:var(--text-dim);font-weight:500;">' +
      '<span style="display:flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:var(--green);border-radius:3px;"></span>Toegewezen</span>' +
      '<span style="display:flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:var(--amber);border-radius:3px;"></span>Deels</span>' +
      '<span style="display:flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;background:var(--red);opacity:0.8;border-radius:3px;"></span>Afgewezen</span>' +
    '</div>';

  var disputes = [];
  uitspraken.forEach(function(u) { if (u.kerngeschil && disputes.indexOf(u.kerngeschil) === -1) disputes.push(u.kerngeschil); });
  var dStats = disputes.map(function(d) {
    var m = uitspraken.filter(function(u) { return u.kerngeschil === d; });
    var a = m.filter(function(u) { return u.uitkomst === 'afgewezen'; }).length;
    return { name: d, count: m.length, pct: Math.round(a / m.length * 100) };
  }).filter(function(d) { return d.count >= 2; }).sort(function(a, b) { return b.pct - a.pct; });

  document.getElementById('insightPredictors').innerHTML =
    '<ul class="factor-list">' +
    dStats.slice(0, 5).map(function(d) {
      return '<li><span>' + d.name + ' (n=' + d.count + ')</span><span class="factor-tag ' + (d.pct > 60 ? 'pro' : d.pct < 40 ? 'con' : 'neutral') + '">' + d.pct + '% afw.</span></li>';
    }).join('') +
    '</ul><p style="font-size:12px;color:var(--text-dim);margin-top:12px;">' + uitspraken.length + ' uitspraken</p>';

  var mBf = uitspraken.filter(function(u) { return u.beslisfactoren; });
  var risks = [];
  if (mBf.length > 0) {
    var onduidelijk = mBf.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === false; });
    var ondT = onduidelijk.filter(function(u) { return u.uitkomst === 'toegewezen' || u.uitkomst === 'deels'; }).length;
    if (onduidelijk.length > 0) risks.push({ name: 'Onduidelijke voorwaarden', pct: Math.round(ondT/onduidelijk.length*100), n: onduidelijk.length });

    var info = mBf.filter(function(u) { return u.beslisfactoren.verzekeraar_informatieplicht_geschonden === true; });
    var infoT = info.filter(function(u) { return u.uitkomst === 'toegewezen' || u.uitkomst === 'deels'; }).length;
    if (info.length > 0) risks.push({ name: 'Informatieplicht geschonden', pct: Math.round(infoT/info.length*100), n: info.length });

    var gc = mBf.filter(function(u) { return u.beslisfactoren.coulance_aangeboden === false; });
    var gcT = gc.filter(function(u) { return u.uitkomst === 'toegewezen' || u.uitkomst === 'deels'; }).length;
    if (gc.length > 0) risks.push({ name: 'Geen coulance', pct: Math.round(gcT/gc.length*100), n: gc.length });

    var sb = mBf.filter(function(u) { return u.beslisfactoren.bewijs_consument === 'sterk'; });
    var sbT = sb.filter(function(u) { return u.uitkomst === 'toegewezen' || u.uitkomst === 'deels'; }).length;
    if (sb.length > 0) risks.push({ name: 'Sterk bewijs consument', pct: Math.round(sbT/sb.length*100), n: sb.length });
  }

  risks.sort(function(a, b) { return b.pct - a.pct; });
  document.getElementById('insightRisks').innerHTML = risks.length > 0 ?
    '<ul class="factor-list">' +
    risks.map(function(r) {
      return '<li><span>' + r.name + ' (n=' + r.n + ')</span><span class="factor-tag ' + (r.pct > 60 ? 'con' : r.pct > 40 ? 'neutral' : 'pro') + '">' + r.pct + '% toeg.</span></li>';
    }).join('') +
    '</ul><p style="font-size:12px;color:var(--text-dim);margin-top:12px;">% toewijzing bij aanwezigheid factor</p>' :
    '<p style="font-size:14px;color:var(--text-dim);">Onvoldoende data.</p>';
}

// ── KIFID Lookup ──
function lookupKifid() {
  var nr = document.getElementById('kifidLookup').value.trim();
  var el = document.getElementById('kifidLookupResult');
  if (!nr) { el.innerHTML = '<p style="font-size:13px;color:var(--amber);">Voer een uitspraaknummer in.</p>'; return; }

  var local = uitspraken.find(function(u) { return u.uitspraaknr === nr; });
  if (local) {
    el.innerHTML =
      '<div style="padding:16px 20px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);margin-top:12px;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">' +
          '<span style="font-size:14px;font-weight:700;">' + local.uitspraaknr + '</span>' +
          '<span class="case-outcome ' + local.uitkomst + '">' + local.uitkomst + '</span>' +
        '</div>' +
        '<p style="font-size:14px;color:var(--text-muted);line-height:1.7;">' + local.samenvatting + '</p>' +
        '<div style="display:flex;gap:16px;margin-top:10px;font-size:12px;color:var(--text-dim);">' +
          '<span>Type: ' + local.type_verzekering + '</span>' +
          '<span>Gevorderd: \u20AC' + Number(local.bedrag_gevorderd || 0).toLocaleString('nl-NL') + '</span>' +
        '</div>' +
      '</div>';
  } else {
    el.innerHTML =
      '<div style="padding:16px 20px;background:var(--bg-alt);border:1px solid var(--border);border-radius:var(--radius);margin-top:12px;">' +
        '<p style="font-size:14px;color:var(--text-muted);">Niet gevonden in lokale data (' + uitspraken.length + ' uitspraken). <a href="https://www.kifid.nl/kifid-kennis-en-uitspraken/uitspraken/?SearchTerm=' + encodeURIComponent('uitspraak-' + nr) + '" target="_blank" style="color:var(--primary);">Zoek op kifid.nl</a></p>' +
      '</div>';
  }
}

// ── Policy Upload ──
async function handlePolicyUpload(event) {
  var file = event.target.files[0];
  if (!file) return;

  var zone = document.getElementById('policyUploadZone');
  var resultEl = document.getElementById('policyAnalysisResult');
  var label = document.getElementById('policyUploadLabel');

  zone.style.borderColor = 'var(--primary)';
  zone.style.background = 'var(--primary-soft)';
  label.innerHTML = '<strong>Analyseren...</strong><br><span style="font-size:12px;color:var(--text-dim);">Dit kan 15\u201330 seconden duren</span>';
  resultEl.innerHTML = '<div style="display:flex;align-items:center;gap:10px;padding:14px 0;font-size:13px;color:var(--text-dim);"><div class="spinner" style="width:16px;height:16px;border-width:2px;"></div>Polisvoorwaarden worden geanalyseerd...</div>';

  var formData = new FormData();
  formData.append('file', file);

  try {
    var resp = await fetch('/api/analyze-policy', { method: 'POST', body: formData });
    var data = await resp.json();

    if (data.error) {
      resultEl.innerHTML = '<div style="padding:12px 16px;background:var(--red-bg);border:1px solid var(--red-border);border-radius:var(--radius);font-size:13px;color:var(--red);">' + data.error + '</div>';
      zone.style.borderColor = 'var(--border)';
      zone.style.background = '';
      label.innerHTML = '<strong style="color:var(--text-secondary);">Klik om opnieuw te uploaden</strong>';
      return;
    }

    policyAnalysis = data;
    zone.style.borderColor = 'var(--green-border)';
    zone.style.background = 'var(--green-bg)';
    label.innerHTML = '<strong style="color:var(--green);">' + file.name + '</strong><br><span style="font-size:12px;color:var(--green);">Risicoscore: ' + data.risicoscore + '/10</span>';

    var clausules = (data.risicovolle_clausules || []).slice(0, 4);
    resultEl.innerHTML =
      '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;margin-top:12px;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">' +
          '<strong style="font-size:14px;">' + (data.product_naam || data.type_verzekering || 'Polisvoorwaarden') + '</strong>' +
          '<span style="font-size:12px;padding:4px 12px;border-radius:6px;font-weight:600;' +
            (data.risicoscore >= 7 ? 'background:var(--red-bg);color:var(--red);' : data.risicoscore >= 4 ? 'background:var(--amber-bg);color:var(--amber);' : 'background:var(--green-bg);color:var(--green);') +
          '">Risico: ' + data.risicoscore + '/10</span>' +
        '</div>' +
        (data.samenvatting ? '<p style="font-size:13px;color:var(--text-muted);line-height:1.7;margin-bottom:14px;">' + data.samenvatting + '</p>' : '') +
        (clausules.length > 0 ?
          '<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-dim);margin-bottom:8px;font-weight:600;">Risicovolle clausules</div>' +
          '<ul style="list-style:none;">' +
          clausules.map(function(c) {
            return '<li style="font-size:13px;color:var(--text-secondary);padding:8px 0;border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;gap:8px;">' +
              '<span>' + (c.artikel ? '<code>Art. ' + c.artikel + '</code> ' : '') + c.clausule + '</span>' +
              '<span style="font-size:11px;padding:2px 8px;border-radius:4px;white-space:nowrap;font-weight:600;' +
                (c.ernst === 'hoog' ? 'background:var(--red-bg);color:var(--red);' : c.ernst === 'middel' ? 'background:var(--amber-bg);color:var(--amber);' : 'background:var(--green-bg);color:var(--green);') +
              '">' + c.ernst + '</span></li>';
          }).join('') + '</ul>' : '') +
      '</div>';

    var typeSelect = document.getElementById('insuranceType');
    if (!typeSelect.value && data.type_verzekering) {
      var opt = Array.from(typeSelect.options).find(function(o) { return o.value === data.type_verzekering; });
      if (opt) typeSelect.value = opt.value;
    }
  } catch (err) {
    resultEl.innerHTML = '<div style="padding:12px 16px;background:var(--red-bg);border:1px solid var(--red-border);border-radius:var(--radius);font-size:13px;color:var(--red);">Server niet bereikbaar. Start: python3 scripts/server.py</div>';
    zone.style.borderColor = 'var(--border)';
    zone.style.background = '';
    label.innerHTML = '<strong style="color:var(--text-secondary);">Klik om opnieuw te uploaden</strong>';
  }
}

// ── Drag & Drop ──
function initDragDrop() {
  var zone = document.getElementById('uploadZone');
  if (!zone) return;
  zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', function() { zone.classList.remove('drag-over'); });
  zone.addEventListener('drop', function(e) {
    e.preventDefault();
    zone.classList.remove('drag-over');
    var input = document.getElementById('fileInput');
    input.files = e.dataTransfer.files;
    handleFileUpload({ target: input });
  });
}
