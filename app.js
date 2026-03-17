// ── KIFID Predictor ──

let uitspraken = [];
let currentTab = 'predict';
let policyAnalysis = null;
let trainedModel = null;

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  autoLoadDataset();
  autoLoadModel();
  initCountAnimations();
  initTheme();
  initDragDrop();
});

// ── Load trained model ──
async function autoLoadModel() {
  try {
    var response = await fetch('data/model.json');
    if (!response.ok) return;
    trainedModel = await response.json();
    var n = (trainedModel.meta || {}).totaal_uitspraken || 0;
    var versie = (trainedModel.meta || {}).versie || '1.0';
    var hasEnsemble = !!(trainedModel.ensemble);
    console.log('Model geladen v' + versie + ': ' + n + ' uitspraken' + (hasEnsemble ? ', ensemble (LR+NB+TF-IDF)' : ''));
    if (hasEnsemble && trainedModel.ensemble.cross_validation) {
      console.log('  Cross-val accuracy: ' + (trainedModel.ensemble.cross_validation.accuracy * 100).toFixed(1) + '%');
    }
    // Show model status in UI
    var statusEl = document.getElementById('dataStatus');
    if (statusEl && uitspraken.length > 0) {
      statusEl.textContent = uitspraken.length + ' uitspraken geladen, ensemble model v' + versie + ' (' + n + ' zaken).';
    }
  } catch (e) {
    console.log('Geen getraind model gevonden, runtime analyse wordt gebruikt.');
  }
}

// ── Ensemble Predictor (browser-side inference) ──

var ENSEMBLE_JURIDISCHE_TERMEN = [
  "bewijslast", "redelijkheid en billijkheid", "zorgplicht", "contra proferentem",
  "art. 6:248 bw", "art. 7:940 bw", "avg", "onaanvaardbaar", "haviltex",
  "mededelingsplicht", "informatieplicht", "klachtplicht", "schadebeperkingsplicht",
  "dekkingsomvang", "eigen schuld", "eigen gebrek", "merkelijke schuld",
  "grove nalatigheid", "opzet", "molest", "polisvoorwaarden", "uitsluitingsclausule",
  "kernbeding", "algemene voorwaarden", "verjaringstermijn", "schending",
  "onredelijk bezwarend", "dwingend recht", "art. 7:941 bw", "art. 7:943 bw",
  "art. 7:952 bw", "art. 7:953 bw", "art. 6:233 bw", "art. 6:236 bw",
  "art. 6:237 bw", "coulance", "bindend advies", "geschillencommissie",
  "commissie van beroep", "wft", "bgfo"
];

var ENSEMBLE_INSURANCE_TYPES = [
  "autoverzekering", "woonhuisverzekering", "inboedelverzekering",
  "reisverzekering", "aansprakelijkheidsverzekering", "rechtsbijstandverzekering",
  "levensverzekering", "arbeidsongeschiktheidsverzekering", "zorgverzekering",
  "beleggingsverzekering", "overlijdensrisicoverzekering", "opstalverzekering",
  "bromfietsverzekering", "brandverzekering", "transportverzekering", "overig"
];

var ENSEMBLE_DISPUTE_TYPES = [
  "dekkingsweigering", "uitleg_voorwaarden", "schadevaststelling",
  "premiegeschil", "mededelingsplicht", "opzegging", "zorgplicht",
  "informatievoorziening", "clausule", "vertraging", "fraude",
  "eigen_gebrek", "overig"
];

var ENSEMBLE_EXPERT_TYPES = ["geen", "consument", "verzekeraar", "beide", "onafhankelijk"];
var ENSEMBLE_OUTCOME_NAMES = ["afgewezen", "toegewezen", "deels"];

function ensembleBuildFeatureVector(input, textForInput) {
  var features = [];
  var text = textForInput.toLowerCase();

  // 1. Type verzekering (one-hot)
  ENSEMBLE_INSURANCE_TYPES.forEach(function(t) { features.push(input.type === t ? 1 : 0); });

  // 2. Kerngeschil (one-hot)
  ENSEMBLE_DISPUTE_TYPES.forEach(function(d) { features.push(input.dispute === d ? 1 : 0); });

  // 3. Bedrag features
  var bedrag = input.amount || 0;
  features.push(Math.log1p(bedrag));
  features.push(bedrag > 50000 ? 1 : 0);
  features.push(bedrag > 100000 ? 1 : 0);
  features.push(bedrag > 0 && bedrag <= 5000 ? 1 : 0);

  // 4. Bindend advies
  features.push(input.binding === 'bindend' ? 1 : 0);

  // 5. Commissie type (default geschillencommissie)
  features.push(0);

  // 6. Jaar (genormaliseerd)
  var jaar = new Date().getFullYear();
  features.push((jaar - 2000) / 26.0);

  // 7. Beslisfactoren
  var bewijsMap = { sterk: 3, gemiddeld: 2, zwak: 1, geen: 0 };
  var bewijsScore = (bewijsMap[input.evidence] || 2) / 3.0;
  features.push(bewijsScore);

  // Deskundigenrapport (one-hot)
  ENSEMBLE_EXPERT_TYPES.forEach(function(e) { features.push(input.expert === e ? 1 : 0); });

  // Boolean beslisfactoren
  features.push(1); // polisvoorwaarden_duidelijk (default ja)
  features.push(0); // consument_nalatig (default nee)
  features.push(0); // verzekeraar_informatieplicht_geschonden (default nee)
  features.push(input.goodwill !== 'nee' ? 1 : 0); // coulance

  // 8. Juridische termen (counts)
  ENSEMBLE_JURIDISCHE_TERMEN.forEach(function(term) {
    var count = 0;
    var idx = 0;
    while ((idx = text.indexOf(term, idx)) !== -1) { count++; idx += term.length; }
    features.push(count);
  });

  // 9. Tekststatistieken
  var words = text.split(/\s+/).filter(function(w) { return w.length > 0; });
  var nWords = words.length;
  var avgWordLen = nWords > 0 ? words.reduce(function(s, w) { return s + w.length; }, 0) / nWords : 0;
  var sentences = text.split(/[.!?]+/).filter(function(s) { return s.trim().length > 0; });
  var nSentences = Math.max(1, sentences.length);
  var uniqueWords = {};
  words.forEach(function(w) { if (w.length > 2) uniqueWords[w.toLowerCase()] = true; });
  var diversity = nWords > 0 ? Object.keys(uniqueWords).length / nWords : 0;

  features.push(nWords / 100.0);
  features.push(avgWordLen / 10.0);
  features.push(nWords / nSentences / 30.0);
  features.push(diversity);

  // 10. Interactie-features
  var bewijsVal = bewijsMap[input.evidence] || 2;
  features.push(bewijsVal / 3.0 * Math.log1p(bedrag) / 15.0);
  var typeIdx = ENSEMBLE_INSURANCE_TYPES.indexOf(input.type || 'overig');
  if (typeIdx < 0) typeIdx = ENSEMBLE_INSURANCE_TYPES.length - 1;
  var dispIdx = ENSEMBLE_DISPUTE_TYPES.indexOf(input.dispute || 'overig');
  if (dispIdx < 0) dispIdx = ENSEMBLE_DISPUTE_TYPES.length - 1;
  features.push(typeIdx * ENSEMBLE_DISPUTE_TYPES.length + dispIdx);
  features.push(0 * bewijsVal / 3.0); // nalatig × bewijs (default nalatig=0)
  features.push((input.goodwill !== 'nee' ? 1 : 0) * Math.log1p(bedrag) / 15.0);
  features.push(0); // info_geschonden × pv_onduidelijk

  // 11. Tags
  features.push(0); // heeft_tags
  features.push(0); // n_tags

  // 12. Totaal juridische termen
  var totalJur = 0;
  ENSEMBLE_JURIDISCHE_TERMEN.forEach(function(term) {
    var idx2 = 0;
    while ((idx2 = text.indexOf(term, idx2)) !== -1) { totalJur++; idx2 += term.length; }
  });
  features.push(totalJur / 5.0);

  return features;
}

function ensemblePredict(input) {
  var tm = trainedModel;
  if (!tm || !tm.ensemble) return null;
  var ens = tm.ensemble;
  var lr = ens.logreg;
  var nb = ens.naive_bayes;
  var tfidf = ens.tfidf;
  var weights = ens.ensemble_weights;

  if (!lr || !nb || !tfidf || !weights) return null;

  // Build text from input context
  var text = (input.context || '') + ' ' + (input.type || '') + ' ' + (input.dispute || '');

  // ── Model 1: Logistic Regression ──
  var features = ensembleBuildFeatureVector(input, text);

  // Standardize
  var scaled = features.map(function(val, i) {
    var mean = lr.scaler_mean[i] || 0;
    var std = lr.scaler_std[i] || 1;
    return std > 0 ? (val - mean) / std : 0;
  });

  // Logits per klasse
  var lrProbs = [];
  for (var c = 0; c < lr.weights.length; c++) {
    var logit = lr.bias[c];
    for (var j = 0; j < scaled.length; j++) {
      logit += lr.weights[c][j] * scaled[j];
    }
    lrProbs.push(logit);
  }

  // Softmax
  var maxLogit = Math.max.apply(null, lrProbs);
  var expSum = 0;
  lrProbs = lrProbs.map(function(l) { var e = Math.exp(l - maxLogit); expSum += e; return e; });
  lrProbs = lrProbs.map(function(e) { return e / expSum; });

  // Map naar [afgewezen, toegewezen, deels] op basis van classes
  var lrProbsFull = [0, 0, 0];
  lr.classes.forEach(function(cls, i) { lrProbsFull[cls] = lrProbs[i]; });

  // ── Model 2: Naive Bayes (via TF-IDF features) ──
  // Tokenize
  var tokens = text.toLowerCase().split(/[^a-z\u00e0-\u00ff0-9]+/).filter(function(t) { return t.length > 1; });

  // Bouw TF-IDF vector
  var vocab = tfidf.vocabulary;
  var idf = tfidf.idf;
  var termFreqs = {};
  tokens.forEach(function(t) { if (vocab[t] !== undefined) { termFreqs[vocab[t]] = (termFreqs[vocab[t]] || 0) + 1; } });

  // Bigrams
  for (var bi = 0; bi < tokens.length - 1; bi++) {
    var bigram = tokens[bi] + ' ' + tokens[bi + 1];
    if (vocab[bigram] !== undefined) { termFreqs[vocab[bigram]] = (termFreqs[vocab[bigram]] || 0) + 1; }
  }

  // TF-IDF vector (sparse)
  var tfidfVec = {};
  var tfidfNorm = 0;
  for (var termIdx in termFreqs) {
    var tf = 1 + Math.log(termFreqs[termIdx]); // sublinear_tf
    var tfidfVal = tf * (idf[termIdx] || 0);
    tfidfVec[termIdx] = tfidfVal;
    tfidfNorm += tfidfVal * tfidfVal;
  }
  tfidfNorm = Math.sqrt(tfidfNorm) || 1;

  // NB log probabilities
  var nbProbs = [0, 0, 0];
  nb.classes.forEach(function(cls, clsIdx) {
    var logProb = nb.log_priors[clsIdx];
    for (var tIdx in termFreqs) {
      var count = termFreqs[tIdx];
      if (nb.feature_log_probs[clsIdx] && nb.feature_log_probs[clsIdx][tIdx] !== undefined) {
        logProb += count * nb.feature_log_probs[clsIdx][tIdx];
      }
    }
    nbProbs[cls] = logProb;
  });

  // Softmax NB
  var nbMax = Math.max.apply(null, nbProbs);
  var nbExpSum = 0;
  nbProbs = nbProbs.map(function(l) { var e = Math.exp(l - nbMax); nbExpSum += e; return e; });
  nbProbs = nbProbs.map(function(e) { return e / nbExpSum; });

  // ── Model 3: TF-IDF Cosine Similarity ──
  var tfidfProbs = [0, 0, 0];
  ENSEMBLE_OUTCOME_NAMES.forEach(function(cls, clsIdx) {
    var centroid = tfidf.centroids[cls];
    if (!centroid) return;
    var dot = 0;
    var centNorm = 0;
    for (var ci in centroid) {
      centNorm += centroid[ci] * centroid[ci];
      if (tfidfVec[ci]) {
        dot += (tfidfVec[ci] / tfidfNorm) * centroid[ci];
      }
    }
    centNorm = Math.sqrt(centNorm) || 1;
    tfidfProbs[clsIdx] = Math.max(0, dot / centNorm);
  });

  // Normaliseer tfidf probs
  var tfidfSum = tfidfProbs.reduce(function(a, b) { return a + b; }, 0) || 1;
  tfidfProbs = tfidfProbs.map(function(p) { return p / tfidfSum; });

  // ── Ensemble ──
  var wLr = weights.logreg || 0.33;
  var wNb = weights.nb || 0.34;
  var wTfidf = weights.tfidf || 0.33;

  var ensembleProbs = [
    wLr * lrProbsFull[0] + wNb * nbProbs[0] + wTfidf * tfidfProbs[0],
    wLr * lrProbsFull[1] + wNb * nbProbs[1] + wTfidf * tfidfProbs[1],
    wLr * lrProbsFull[2] + wNb * nbProbs[2] + wTfidf * tfidfProbs[2],
  ];

  // Normaliseer
  var ensSum = ensembleProbs.reduce(function(a, b) { return a + b; }, 0) || 1;
  ensembleProbs = ensembleProbs.map(function(p) { return p / ensSum; });

  var predicted = ensembleProbs.indexOf(Math.max.apply(null, ensembleProbs));

  return {
    probs: {
      afgewezen: Math.round(ensembleProbs[0] * 1000) / 10,
      toegewezen: Math.round(ensembleProbs[1] * 1000) / 10,
      deels: Math.round(ensembleProbs[2] * 1000) / 10,
    },
    predicted: ENSEMBLE_OUTCOME_NAMES[predicted],
    modelProbs: {
      logreg: lrProbsFull,
      nb: nbProbs,
      tfidf: tfidfProbs,
    },
    cvAccuracy: (ens.cross_validation || {}).accuracy || 0,
  };
}

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
  var allMetrics = document.querySelectorAll('.metric-value[data-count]:not([id])');
  if (allMetrics.length >= 1) {
    allMetrics[0].dataset.count = total;
    allMetrics[0].textContent = total + '+';
  }
  // Update last metric (woonhuisverzekering count)
  if (allMetrics.length >= 2) {
    var woonhuis = uitspraken.filter(function(u) { return u.type_verzekering === 'woonhuisverzekering'; }).length;
    allMetrics[1].dataset.count = woonhuis;
    allMetrics[1].textContent = woonhuis;
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

// ── Helper: lookup beslisfactor in trained model ──
function modelBfLookup(typeModel, factor, value) {
  // Try type-specific model first, then global model
  var sources = [];
  if (typeModel && typeModel.beslisfactoren && typeModel.beslisfactoren[factor]) {
    sources.push(typeModel.beslisfactoren[factor]);
  }
  if (trainedModel && trainedModel.beslisfactoren && trainedModel.beslisfactoren[factor]) {
    sources.push(trainedModel.beslisfactoren[factor]);
  }
  for (var i = 0; i < sources.length; i++) {
    var data = sources[i][value] || sources[i][String(value)];
    if (data && data.n >= 3) return data;
  }
  return null;
}

function analyzeCase(input) {
  var score = 50;
  var factors = [];
  var bfAll = uitspraken.filter(function(u) { return u.beslisfactoren; });
  var overallAfwRate = afwPct(uitspraken);

  // ── Gebruik getraind model voor betere voorspellingen ──
  var tm = trainedModel; // shorthand
  var typeModel = tm && tm.per_type ? tm.per_type[input.type] : null;
  var geschilModel = null;
  if (typeModel && typeModel.per_kerngeschil) {
    geschilModel = typeModel.per_kerngeschil[input.dispute] || null;
  }

  // ── 1. Base rate: verzekeringtype (model of runtime) ──
  var typeN = 0;
  if (typeModel) {
    score = typeModel.afw_pct;
    typeN = typeModel.n;
    factors.push({ label: 'Afwijzingspercentage ' + input.type + ' (n=' + typeN + ', model)', value: typeModel.afw_pct + '%', type: typeModel.afw_pct > 60 ? 'pro' : 'con' });
  } else {
    var typeMatches = uitspraken.filter(function(u) { return u.type_verzekering === input.type; });
    typeN = typeMatches.length;
    if (typeN > 0) {
      var typeRate = afwPct(typeMatches);
      score = typeRate;
      factors.push({ label: 'Afwijzingspercentage ' + input.type + ' (n=' + typeN + ')', value: typeRate + '%', type: typeRate > 60 ? 'pro' : 'con' });
    }
  }

  // ── 2. Kerngeschil correctie (model of runtime) ──
  var disputeN = 0;
  if (geschilModel) {
    var gmRate = geschilModel.afw_pct;
    disputeN = geschilModel.n;
    var gmAdj = Math.round((gmRate - score) * 0.35);
    score += gmAdj;
    factors.push({ label: input.type + ' + ' + input.dispute + ' (' + gmRate + '% afw, n=' + disputeN + ', model)', value: fmtDelta(gmAdj), type: gmAdj > 0 ? 'pro' : gmAdj < 0 ? 'con' : 'neutral' });
    // Toewijzingsratio context
    if (geschilModel.toewijzings_ratio > 0) {
      factors.push({ label: 'Gem. toewijzingsratio bij dit geschiltype', value: geschilModel.toewijzings_ratio + '%', type: geschilModel.toewijzings_ratio > 30 ? 'con' : 'neutral' });
    }
  } else {
    var disputeMatches = uitspraken.filter(function(u) { return u.kerngeschil === input.dispute; });
    disputeN = disputeMatches.length;
    if (disputeN > 0) {
      var dRate = afwPct(disputeMatches);
      var adj = Math.round((dRate - 50) * 0.3);
      score += adj;
      factors.push({ label: 'Kerngeschil ' + input.dispute + ' (' + dRate + '% afw, n=' + disputeN + ')', value: fmtDelta(adj), type: adj > 0 ? 'pro' : adj < 0 ? 'con' : 'neutral' });
    }
  }

  // ── 3. Combinatie type + geschil (runtime fallback als geen model) ──
  var combiMatches = uitspraken.filter(function(u) { return u.type_verzekering === input.type && u.kerngeschil === input.dispute; });
  if (!geschilModel && combiMatches.length >= 3) {
    var combiRate = afwPct(combiMatches);
    var combiAdj = Math.round((combiRate - score) * 0.4);
    score += combiAdj;
    factors.push({ label: 'Specifiek ' + input.type + ' + ' + input.dispute + ' (n=' + combiMatches.length + ')', value: fmtDelta(combiAdj), type: combiAdj > 0 ? 'pro' : combiAdj < 0 ? 'con' : 'neutral' });
  }

  // ── 4. Beslisfactoren: bewijs consument (model of runtime) ──
  if (input.evidence) {
    var evData = modelBfLookup(typeModel, 'bewijs_consument', input.evidence);
    if (evData) {
      var evBase = typeModel ? typeModel.afw_pct : overallAfwRate;
      var evDelta = Math.round((evData.afw_pct - evBase) * 0.4);
      score += evDelta;
      factors.push({ label: 'Bewijs \u201C' + input.evidence + '\u201D bij ' + (typeModel ? input.type : 'alle') + ' \u2192 ' + evData.afw_pct + '% afw. (n=' + evData.n + ', model)', value: fmtDelta(evDelta), type: evDelta > 5 ? 'pro' : evDelta < -5 ? 'con' : 'neutral' });
    } else if (bfAll.length >= 5) {
      var sameEvidence = bfAll.filter(function(u) { return u.beslisfactoren.bewijs_consument === input.evidence; });
      if (sameEvidence.length >= 3) {
        var evRate = afwPct(sameEvidence);
        var evDelta2 = Math.round((evRate - overallAfwRate) * 0.4);
        score += evDelta2;
        factors.push({ label: 'Bewijs \u201C' + input.evidence + '\u201D \u2192 ' + evRate + '% afw. (n=' + sameEvidence.length + ')', value: fmtDelta(evDelta2), type: evDelta2 > 5 ? 'pro' : evDelta2 < -5 ? 'con' : 'neutral' });
      }
    }
  }

  // ── 5. Beslisfactoren: deskundigenrapport (model of runtime) ──
  if (input.expert && input.expert !== 'geen') {
    var exData = modelBfLookup(typeModel, 'deskundigenrapport', input.expert);
    if (exData) {
      var exBase = typeModel ? typeModel.afw_pct : overallAfwRate;
      var exDelta = Math.round((exData.afw_pct - exBase) * 0.35);
      score += exDelta;
      factors.push({ label: 'Deskundigenrapport \u201C' + input.expert + '\u201D bij ' + (typeModel ? input.type : 'alle') + ' \u2192 ' + exData.afw_pct + '% afw. (n=' + exData.n + ', model)', value: fmtDelta(exDelta), type: exDelta > 3 ? 'pro' : exDelta < -3 ? 'con' : 'neutral' });
    } else if (bfAll.length >= 5) {
      var sameExpert = bfAll.filter(function(u) { return u.beslisfactoren.deskundigenrapport === input.expert; });
      if (sameExpert.length >= 3) {
        var exRate = afwPct(sameExpert);
        var exDelta2 = Math.round((exRate - overallAfwRate) * 0.35);
        score += exDelta2;
        factors.push({ label: 'Deskundigenrapport \u201C' + input.expert + '\u201D \u2192 ' + exRate + '% afw. (n=' + sameExpert.length + ')', value: fmtDelta(exDelta2), type: exDelta2 > 3 ? 'pro' : exDelta2 < -3 ? 'con' : 'neutral' });
      }
    }
  }

  // ── 6. Beslisfactoren: polisvoorwaarden duidelijkheid (model of runtime) ──
  var pvdData = modelBfLookup(typeModel, 'polisvoorwaarden_duidelijk', 'true');
  var pvuData = modelBfLookup(typeModel, 'polisvoorwaarden_duidelijk', 'false');
  if (pvdData && pvuData) {
    var verschil = pvdData.afw_pct - pvuData.afw_pct;
    if (Math.abs(verschil) >= 5) {
      factors.push({ label: 'Onduidelijke voorwaarden \u2192 ' + pvuData.afw_pct + '% afw. vs. duidelijk ' + pvdData.afw_pct + '% (n=' + pvuData.n + '/' + pvdData.n + ', model)', value: '\u0394' + Math.abs(Math.round(verschil)) + '%', type: verschil > 0 ? 'con' : 'pro' });
    }
  } else if (bfAll.length >= 5) {
    var onduidelijk = bfAll.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === false; });
    var duidelijk = bfAll.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === true; });
    if (onduidelijk.length >= 3 && duidelijk.length >= 3) {
      var ondRate = afwPct(onduidelijk);
      var duiRate = afwPct(duidelijk);
      var verschil2 = duiRate - ondRate;
      if (Math.abs(verschil2) >= 10) {
        factors.push({ label: 'Onduidelijke voorwaarden \u2192 ' + ondRate + '% afw. vs. duidelijk ' + duiRate + '% (n=' + onduidelijk.length + '/' + duidelijk.length + ')', value: '\u0394' + Math.abs(verschil2) + '%', type: verschil2 > 0 ? 'con' : 'pro' });
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

  // ── 14. Factor-combinaties uit model ──
  var modelCombos = typeModel ? typeModel.factor_combinaties : (tm ? tm.factor_combinaties : []);
  if (modelCombos && modelCombos.length > 0) {
    // Vind combinaties die matchen met de input
    var matchedCombo = modelCombos.filter(function(c) {
      var f = c.factoren || '';
      var matches = true;
      if (f.indexOf('bewijs_consument=' + input.evidence) !== -1) matches = true;
      else if (f.indexOf('bewijs_consument=') !== -1) matches = false;
      if (f.indexOf('deskundigenrapport=' + input.expert) !== -1) matches = matches && true;
      else if (f.indexOf('deskundigenrapport=') !== -1) matches = false;
      if (input.goodwill !== 'nee' && f.indexOf('coulance_aangeboden=ja') !== -1) matches = matches && true;
      else if (input.goodwill === 'nee' && f.indexOf('coulance_aangeboden=nee') !== -1) matches = matches && true;
      else if (f.indexOf('coulance_aangeboden=') !== -1) matches = false;
      return matches;
    });
    if (matchedCombo.length > 0) {
      var best = matchedCombo[0]; // Highest impact
      var cmbAdj = Math.round(best.impact * 0.25);
      if (Math.abs(cmbAdj) >= 3) {
        score += cmbAdj;
        factors.push({ label: 'Factor-combinatie: ' + best.factoren + ' \u2192 ' + best.afw_pct + '% afw. (n=' + best.n + ', model)', value: fmtDelta(cmbAdj), type: cmbAdj > 3 ? 'pro' : cmbAdj < -3 ? 'con' : 'neutral' });
      }
    }
  }

  score = Math.max(5, Math.min(95, score));

  // ── 15. Ensemble model integratie ──
  var ensembleResult = ensemblePredict(input);
  if (ensembleResult) {
    // Converteer ensemble kans naar score (0=uitkeren, 100=afwachten)
    // afgewezen = verzekeraar wint = hoge score
    var ensScore = Math.round(ensembleResult.probs.afgewezen + ensembleResult.probs.deels * 0.5);

    // Blend statistisch model met ensemble (60% ensemble, 40% statistisch)
    var oldScore = score;
    score = Math.round(0.6 * ensScore + 0.4 * score);
    score = Math.max(5, Math.min(95, score));

    var ensAdj = score - oldScore;
    if (Math.abs(ensAdj) >= 2) {
      factors.push({
        label: 'Ensemble model (LR+NB+TF-IDF, cv=' + Math.round(ensembleResult.cvAccuracy * 100) + '%): ' + ensembleResult.probs.afgewezen + '% afw. / ' + ensembleResult.probs.toegewezen + '% toeg. / ' + ensembleResult.probs.deels + '% deels',
        value: fmtDelta(ensAdj),
        type: ensAdj > 3 ? 'pro' : ensAdj < -3 ? 'con' : 'neutral'
      });
    } else {
      factors.push({
        label: 'Ensemble model bevestigt statistische analyse (' + ensembleResult.probs.afgewezen + '% afw.)',
        value: 'bevestigd',
        type: 'neutral'
      });
    }
  }

  // ── Betrouwbaarheid ──
  var relevantMatches = uitspraken.filter(function(u) { return u.type_verzekering === input.type || u.kerngeschil === input.dispute; }).length;
  var bfRelevant = bfAll.filter(function(u) { return u.type_verzekering === input.type || u.kerngeschil === input.dispute; }).length;
  var confidence = 'laag';
  var confidencePct = 20;
  // Model geeft flinke boost
  if (typeModel) {
    confidencePct = 50;
    if (typeModel.n >= 100) confidencePct = 70;
    if (typeModel.n >= 300) confidencePct = 80;
    if (geschilModel && geschilModel.n >= 10) confidencePct = Math.min(95, confidencePct + 10);
  }
  // Meer data = meer betrouwbaar
  if (relevantMatches >= 30) { confidence = 'hoog'; confidencePct = Math.max(confidencePct, 85); }
  else if (relevantMatches >= 15) { confidence = 'hoog'; confidencePct = Math.max(confidencePct, 75); }
  else if (relevantMatches >= 8) { confidence = 'gemiddeld'; confidencePct = Math.max(confidencePct, 60); }
  else if (relevantMatches >= 4) { confidence = 'beperkt'; confidencePct = Math.max(confidencePct, 40); }
  // Bonus voor beschikbare beslisfactoren
  if (bfRelevant >= 10) confidencePct = Math.min(95, confidencePct + 5);
  // Confidence label
  if (confidencePct >= 75) confidence = 'hoog';
  else if (confidencePct >= 55) confidence = 'gemiddeld';
  else if (confidencePct >= 35) confidence = 'beperkt';
  if (policyAnalysis) confidencePct = Math.min(95, confidencePct + 10);
  // Ensemble model boost
  if (ensembleResult) confidencePct = Math.min(95, confidencePct + 10);

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

  return { score: score, factors: factors, similar: similar, dataPoints: uitspraken.length, confidence: confidence, confidencePct: confidencePct, relevantMatches: relevantMatches, ensemble: ensembleResult };
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
  var verdict = s >= 60 ? 'Procedure afwachten' : s <= 40 ? 'Overweeg uitkering' : 'Nader beoordelen';
  var verdictSub = s >= 60 ? 'Op basis van historische KIFID-uitspraken is de kans groot dat de vordering wordt afgewezen.' :
    s <= 40 ? 'Er zijn significante risicofactoren. Overweeg een schikking of verhoogd coulanceaanbod.' :
    'De uitkomst is onzeker. Aanvullende juridische analyse wordt aanbevolen.';
  var vc = s >= 60 ? 'afwachten' : s <= 40 ? 'uitkeren' : 'onzeker';
  var bc = s >= 60 ? 'amber' : s <= 40 ? 'green' : 'amber';
  var cc = result.confidencePct >= 60 ? 'var(--green)' : result.confidencePct >= 40 ? 'var(--amber)' : 'var(--red)';
  var confLabel = result.confidencePct >= 75 ? 'Hoog' : result.confidencePct >= 55 ? 'Gemiddeld' : result.confidencePct >= 35 ? 'Beperkt' : 'Laag';
  var now = new Date();
  var dateStr = now.toLocaleDateString('nl-NL', { day: 'numeric', month: 'long', year: 'numeric' });
  var timeStr = now.toLocaleTimeString('nl-NL', { hour: '2-digit', minute: '2-digit' });
  var typeLabel = input.type ? input.type.replace(/_/g, ' ') : 'niet opgegeven';
  var disputeLabel = input.dispute ? input.dispute.replace(/_/g, ' ') : 'niet opgegeven';

  // ── Report header ──
  var html =
    '<div class="report-container animate-in">' +
      '<div class="report-header">' +
        '<div class="report-header-top">' +
          '<div class="report-badge">' +
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>' +
            '<span>Risico-inschatting</span>' +
          '</div>' +
          '<div class="report-meta">' + dateStr + ' &middot; ' + timeStr + '</div>' +
        '</div>' +
        '<div class="report-case-summary">' +
          '<span class="report-case-tag">' + typeLabel + '</span>' +
          '<span class="report-case-tag">' + disputeLabel + '</span>' +
          (input.amount > 0 ? '<span class="report-case-tag">&euro; ' + Number(input.amount).toLocaleString('nl-NL') + '</span>' : '') +
        '</div>' +
      '</div>';

  // ── Verdict card ──
  html +=
      '<div class="report-verdict">' +
        '<div class="report-verdict-inner">' +
          '<div class="report-verdict-score">' +
            '<div class="report-score-ring ' + vc + '">' +
              '<svg viewBox="0 0 120 120">' +
                '<circle cx="60" cy="60" r="52" fill="none" stroke="var(--border)" stroke-width="8"/>' +
                '<circle cx="60" cy="60" r="52" fill="none" stroke-width="8" stroke-linecap="round" class="report-score-arc ' + vc + '" style="stroke-dasharray: ' + Math.round(s * 3.267) + ' 326.7; transform: rotate(-90deg); transform-origin: center;"/>' +
              '</svg>' +
              '<div class="report-score-value">' + s + '</div>' +
            '</div>' +
          '</div>' +
          '<div class="report-verdict-text">' +
            '<div class="report-verdict-label">Advies</div>' +
            '<div class="report-verdict-title ' + vc + '">' + verdict + '</div>' +
            '<p class="report-verdict-desc">' + verdictSub + '</p>' +
          '</div>' +
        '</div>' +
        '<div class="report-verdict-bar">' +
          '<div class="report-bar-track"><div class="report-bar-fill ' + bc + '" style="width:' + s + '%"></div></div>' +
          '<div class="report-bar-labels"><span>Uitkeren</span><span>Afwachten</span></div>' +
        '</div>' +
        '<div class="report-verdict-stats">' +
          '<div class="report-stat">' +
            '<div class="report-stat-icon" style="color:' + cc + '"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div>' +
            '<div><div class="report-stat-value" style="color:' + cc + '">' + confLabel + ' (' + result.confidencePct + '%)</div><div class="report-stat-label">Betrouwbaarheid</div></div>' +
          '</div>' +
          '<div class="report-stat">' +
            '<div class="report-stat-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>' +
            '<div><div class="report-stat-value">' + result.relevantMatches + ' zaken</div><div class="report-stat-label">Vergelijkbaar</div></div>' +
          '</div>' +
          '<div class="report-stat">' +
            '<div class="report-stat-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>' +
            '<div><div class="report-stat-value">' + result.dataPoints.toLocaleString('nl-NL') + '</div><div class="report-stat-label">Uitspraken in dataset</div></div>' +
          '</div>' +
        '</div>' +
      '</div>';

  // ── Beslisfactoren ──
  var proFactors = f.filter(function(x) { return x.type === 'pro'; });
  var conFactors = f.filter(function(x) { return x.type === 'con'; });
  var neutralFactors = f.filter(function(x) { return x.type === 'neutral'; });

  html += '<div class="report-grid">' +
    '<div class="report-card animate-in delay-1">' +
      '<div class="report-card-header">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>' +
        '<h3>Beslisfactoren (' + f.length + ')</h3>' +
      '</div>' +
      '<p class="report-card-desc">Elke factor is gewogen op basis van historische KIFID-uitspraken met vergelijkbare kenmerken.</p>';

  if (proFactors.length > 0) {
    html += '<div class="report-factor-group"><div class="report-factor-group-label pro">Positie verzekeraar versterkt</div><ul class="factor-list">' +
      proFactors.map(function(x) { return '<li><span>' + x.label + '</span><span class="factor-tag pro">' + x.value + '</span></li>'; }).join('') +
      '</ul></div>';
  }
  if (conFactors.length > 0) {
    html += '<div class="report-factor-group"><div class="report-factor-group-label con">Risicoverhogende factoren</div><ul class="factor-list">' +
      conFactors.map(function(x) { return '<li><span>' + x.label + '</span><span class="factor-tag con">' + x.value + '</span></li>'; }).join('') +
      '</ul></div>';
  }
  if (neutralFactors.length > 0) {
    html += '<div class="report-factor-group"><div class="report-factor-group-label neutral">Contextfactoren</div><ul class="factor-list">' +
      neutralFactors.map(function(x) { return '<li><span>' + x.label + '</span><span class="factor-tag neutral">' + x.value + '</span></li>'; }).join('') +
      '</ul></div>';
  }
  html += '</div>';

  // ── Aanbeveling ──
  html +=
    '<div class="report-card animate-in delay-2">' +
      '<div class="report-card-header">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>' +
        '<h3>Aanbeveling</h3>' +
      '</div>' +
      '<div class="report-recommendation ' + vc + '">' +
        '<div class="report-rec-verdict">' + verdict + '</div>' +
        '<p>' +
        (s >= 60
          ? 'De verzekeraar heeft op basis van vergelijkbare uitspraken een sterke positie. Het afwijzingspercentage bij soortgelijke zaken is hoog. Procedure afwachten lijkt verdedigbaar, mits de polisvoorwaarden helder zijn geformuleerd.'
          : s <= 40
          ? 'Er zijn substantiele risicofactoren aanwezig die wijzen op een mogelijke (gedeeltelijke) toewijzing. Overweeg proactief een schikking of verhoogd coulanceaanbod om proceskosten en reputatierisico te beperken.'
          : 'De uitkomst van vergelijkbare zaken laat geen eenduidig patroon zien. Aanvullende juridische analyse is aanbevolen voordat een definitieve strategie wordt bepaald.') +
        '</p>' +
      '</div>' +
      '<div class="report-methodology">' +
        '<div class="report-methodology-title">' +
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>' +
          'Methodologie' +
        '</div>' +
        '<p>Deze inschatting combineert een <strong>ensemble van drie modellen</strong> (Logistic Regression, Naive Bayes, TF-IDF similarity) met statistische analyse van ' + result.dataPoints.toLocaleString('nl-NL') + ' openbare KIFID-uitspraken. ' +
        'Het model is gevalideerd met 5-fold cross-validatie' + (result.ensemble ? ' (accuracy: ' + Math.round(result.ensemble.cvAccuracy * 100) + '%)' : '') + '. ' +
        '99 gestructureerde features (verzekeringstype, geschiltype, 41 juridische termen, interactie-features) en 800 tekstfeatures worden gewogen.</p>' +
      '</div>' +
    '</div>' +
  '</div>';

  // ── Ensemble model details ──
  if (result.ensemble) {
    var ep = result.ensemble.probs;
    html += '<div class="report-card animate-in delay-2" style="grid-column:span 1;">' +
      '<div class="report-card-header">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>' +
        '<h3>Ensemble Model Analyse</h3>' +
      '</div>' +
      '<p class="report-card-desc">Probabiliteiten berekend door drie onafhankelijke ML-modellen, gewogen gecombineerd.</p>' +
      '<div class="report-ensemble-probs">' +
        '<div class="report-prob-row">' +
          '<span class="report-prob-label">Afgewezen</span>' +
          '<div class="report-prob-bar"><div class="report-prob-fill" style="width:' + ep.afgewezen + '%;background:var(--red);"></div></div>' +
          '<span class="report-prob-value">' + ep.afgewezen + '%</span>' +
        '</div>' +
        '<div class="report-prob-row">' +
          '<span class="report-prob-label">Toegewezen</span>' +
          '<div class="report-prob-bar"><div class="report-prob-fill" style="width:' + ep.toegewezen + '%;background:var(--green);"></div></div>' +
          '<span class="report-prob-value">' + ep.toegewezen + '%</span>' +
        '</div>' +
        '<div class="report-prob-row">' +
          '<span class="report-prob-label">Deels</span>' +
          '<div class="report-prob-bar"><div class="report-prob-fill" style="width:' + ep.deels + '%;background:var(--amber);"></div></div>' +
          '<span class="report-prob-value">' + ep.deels + '%</span>' +
        '</div>' +
      '</div>' +
      '<div style="margin-top:16px;padding-top:14px;border-top:1px solid var(--border-subtle);font-size:12px;color:var(--text-dim);">' +
        '<div style="display:flex;gap:16px;flex-wrap:wrap;">' +
          '<span>Logistic Regression: <strong>' + Math.round(result.ensemble.modelProbs.logreg[0]*100) + '/' + Math.round(result.ensemble.modelProbs.logreg[1]*100) + '/' + Math.round(result.ensemble.modelProbs.logreg[2]*100) + '%</strong></span>' +
          '<span>Naive Bayes: <strong>' + Math.round(result.ensemble.modelProbs.nb[0]*100) + '/' + Math.round(result.ensemble.modelProbs.nb[1]*100) + '/' + Math.round(result.ensemble.modelProbs.nb[2]*100) + '%</strong></span>' +
          '<span>TF-IDF: <strong>' + Math.round(result.ensemble.modelProbs.tfidf[0]*100) + '/' + Math.round(result.ensemble.modelProbs.tfidf[1]*100) + '/' + Math.round(result.ensemble.modelProbs.tfidf[2]*100) + '%</strong></span>' +
        '</div>' +
      '</div>' +
    '</div>';
  }

  // ── Vergelijkbare uitspraken ──
  if (sim.length > 0) {
    html += '<div class="report-cases animate-in delay-3">' +
      '<div class="report-card-header">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>' +
        '<h3>Vergelijkbare KIFID-uitspraken (' + sim.length + ')</h3>' +
      '</div>' +
      '<p class="report-card-desc">Gesorteerd op relevantie. Klik op een uitspraak om de volledige tekst te bekijken op kifid.nl.</p>' +
      sim.map(function(c) {
        var kifidUrl = c.pdfUrl || ('https://www.kifid.nl/kifid-kennis-en-uitspraken/uitspraken/?SearchTerm=' + encodeURIComponent('uitspraak-' + c.nr));
        var grondslagHtml = c.grondslag ? '<span class="case-grondslag" style="font-size:11px;color:var(--text-dim);display:block;margin-top:2px;">' + c.grondslag + '</span>' : '';
        return '<a href="' + kifidUrl + '" target="_blank" rel="noopener" class="case-row case-row-link"><span class="case-nr">' + c.nr + '</span><span class="case-desc">' + c.desc + grondslagHtml + '</span><span class="case-outcome ' + c.outcome + '">' + c.outcome + '</span><span class="case-relevance">' + c.relevance + ' <svg style="width:14px;height:14px;vertical-align:middle;opacity:0.5;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></span></a>';
      }).join('') + '</div>';
  }

  // ── Disclaimer footer ──
  html += '<div class="report-disclaimer animate-in delay-3">' +
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>' +
    '<div>' +
      '<strong>Disclaimer</strong>' +
      '<p>Dit rapport is een indicatieve risico-inschatting op basis van statistische analyse van historische KIFID-uitspraken. ' +
      'Het vormt geen juridisch advies. Raadpleeg een specialist voor definitieve besluitvorming. ' +
      'Historische uitkomsten bieden geen garantie voor toekomstige resultaten.</p>' +
    '</div>' +
  '</div>';

  html += '</div>'; // close report-container

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
