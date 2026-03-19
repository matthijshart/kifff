// ── ClaimWise ──

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
  initLang();
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
  features.push(bedrag > 0 ? 1 : 0); // bedrag_bekend flag

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

  // Calibratie-boost: corrigeer voor class imbalance (minority classes boosten)
  var calBoost = ens.calibration_boost || [1, 1, 1];
  ensembleProbs = ensembleProbs.map(function(p, i) { return p * calBoost[i]; });

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

// ── Language Toggle / i18n ──
var I18N = {
  en: {
    // Nav
    nav_benefits: 'Benefits', nav_risk: 'Risk Assessment', nav_insights: 'Insights',
    nav_cta: 'Start analysis',
    // Hero
    hero_title: 'Predict KIFID rulings with <span class="gradient" data-i18n="hero_gradient">data-driven precision</span>',
    hero_gradient: 'data-driven precision',
    hero_sub: 'Save litigation costs by knowing in advance how KIFID will rule. Compare with historical rulings and receive an evidence-based recommendation instantly.',
    hero_cta_html: 'Free analysis <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
    hero_cta2: 'View benefits',
    // Metrics
    metric_analyzed: 'Analyzed rulings', metric_consumer: 'Consumer wins',
    metric_insurer: 'Insurer wins', metric_home: 'Home insurance',
    // Benefits
    benefits_eyebrow: 'Why insurers use this', benefits_title: 'Faster, sharper, better substantiated',
    benefits_desc: 'From reactive and intuitive to proactive and data-driven. Risk management and reputation protection, not just time savings.',
    benefit1_title: 'Faster complaint assessment', benefit1_desc: 'Find comparable KIFID rulings and relevant arguments within seconds.',
    benefit2_title: 'Lower legal costs', benefit2_desc: 'Prevent unnecessary proceedings by foreseeing how KIFID will likely rule.',
    benefit3_title: 'Better risk assessment', benefit3_desc: 'See the probability of a consumer winning based on prior rulings.',
    benefit4_title: 'Consistent decision-making', benefit4_desc: 'Decisions based on case law rather than individual interpretation.',
    benefit5_title: 'Better policy terms', benefit5_desc: 'Analysis of rulings reveals which clauses frequently lead to disputes.',
    // How it works
    how_eyebrow: 'How it works', how_title: '4 steps to an evidence-based recommendation',
    step1_title: 'Enter case details', step1_desc: 'Insurance type, core dispute, evidence position and other relevant details.',
    step2_title: 'Upload policy terms', step2_desc: 'Optional: let AI scan the terms for high-risk clauses.',
    step3_title: 'AI analyzes', step3_desc: 'The model compares your case with 5,922 historical KIFID rulings and trained patterns.',
    step4_title: 'Receive recommendation', step4_desc: 'Pay out or hold, with decision factors and confidence score.',
    // Model
    model_eyebrow: 'The model', model_title: 'Trained on 5,922 KIFID rulings',
    model_desc: 'An ensemble of three machine learning models, validated with 5-fold cross-validation.',
    arch_input: 'Case features', arch_input_sub: '99 features + text',
    arch_lr_sub: 'Structured features', arch_nb_sub: 'TF-IDF text', arch_tfidf_sub: 'Cosine distance',
    arch_output: 'Risk assessment', arch_output_sub: 'Weighted ensemble',
    trend_title: 'Rejection trend', trend_desc: 'Rejection rate per year. Declining trend: consumers win (partially) more often.',
    fi_title: 'Strongest predictors', fi_desc: 'Top features from the Logistic Regression model, ranked by weight.',
    fi_negligent: 'Consumer negligent', fi_neg_evidence: 'Negligence × evidence', fi_year: 'Year (trend)',
    fi_complaint: 'Legal: complaint duty', fi_life: 'Life insurance', fi_limitation: 'Legal: limitation period',
    fi_report: 'Independent report', fi_gdpr: 'Legal: GDPR',
    fi_pro_consumer: 'Pro consumer', fi_pro_insurer: 'Pro insurer',
    stat_cv: 'Cross-validation accuracy', stat_fold: 'Stratified cross-validation',
    stat_features: 'Features (99 + 800 text)', stat_legal: 'Legal terms',
    // Tool section
    tool_eyebrow: 'Risk Assessment', tool_title: 'Assess the risk.',
    tool_desc: 'Enter the case details and receive an evidence-based risk assessment instantly.',
    tab_predict: 'Risk Assessment', tab_data: 'Data', tab_insights: 'Insights',
    panel_case: 'Case Details',
    // Form labels
    label_type: 'Insurance type', label_dispute: 'Core dispute', label_amount: 'Claimed amount',
    label_binding: 'Binding advice?', label_evidence: 'Consumer evidence', label_expert: 'Expert report',
    label_goodwill: 'Goodwill offer', label_context: 'Additional context', label_upload: 'Upload policy terms',
    // Select options
    opt_select_type: 'Select type', opt_auto: 'Car insurance', opt_home: 'Home insurance',
    opt_contents: 'Contents insurance', opt_travel: 'Travel insurance', opt_liability: 'Liability insurance',
    opt_legal: 'Legal expenses insurance', opt_life: 'Life insurance', opt_disability: 'Disability insurance',
    opt_health: 'Health insurance', opt_investment: 'Investment insurance', opt_death: 'Term life insurance',
    opt_building: 'Building insurance', opt_moped: 'Moped insurance', opt_fire: 'Fire insurance',
    opt_transport: 'Transport insurance', opt_other: 'Other',
    opt_select_dispute: 'Select dispute type', opt_coverage: 'Coverage denial',
    opt_terms: 'Policy terms interpretation', opt_damage: 'Damage assessment / amount',
    opt_premium: 'Premium dispute', opt_disclosure: 'Disclosure breach',
    opt_cancel: 'Policy cancellation', opt_duty: 'Duty of care breach',
    opt_info: 'Inadequate information', opt_exclusion: 'Exclusion clause invoked',
    opt_delay: 'Processing delay', opt_fraud: 'Fraud', opt_defect: 'Inherent defect', opt_other2: 'Other',
    opt_binding_yes: 'Yes, binding', opt_binding_no: 'No, non-binding',
    opt_select_evidence: 'Select evidence strength',
    opt_strong: 'Strong (photos, reports, witnesses)', opt_medium: 'Medium (own statement + some evidence)',
    opt_weak: 'Weak (own statement only)',
    opt_no_report: 'No report', opt_by_consumer: 'Submitted by consumer',
    opt_by_insurer: 'Submitted by insurer', opt_both: 'Both parties',
    opt_no_goodwill: 'No goodwill offered', opt_low_goodwill: 'Yes, low amount',
    opt_fair_goodwill: 'Yes, fair amount',
    placeholder_context: 'Optional: relevant details about the case...',
    upload_click: 'Click to upload PDF', upload_scan: 'AI scans for high-risk clauses',
    btn_predict: 'Analyze & Predict',
    empty_state_html: 'Enter the case details and click <strong style="color:var(--text-muted);">Analyze & Predict</strong>',
    // Data tab
    panel_registry: 'KIFID Rulings Registry', data_analyzed: 'Analyzed rulings',
    data_period: 'Period', data_public: 'Public', data_accessible: 'Freely accessible',
    placeholder_lookup: 'Search by ruling number, e.g. 2024-1029', btn_lookup: 'Look up',
    panel_dataset: 'Manage dataset', data_no_data: 'No data loaded yet.',
    upload_files: 'Click to upload files', upload_formats: 'CSV or JSON files of KIFID rulings',
    btn_load_full: 'Load full dataset', btn_demo: 'Demo data',
    // Insights
    insight_dist_title: 'Outcome distribution by insurance type',
    insight_dist_desc: 'Stacked bars show the ratio of granted, partial and rejected per type.',
    insight_load_first: 'Load data first via the Data tab.',
    insight_predictors_title: 'Strongest predictors',
    insight_predictors_desc: 'Core disputes with the highest rejection rate.',
    insight_available: 'Available after loading data.',
    insight_risk_title: 'Risk factors',
    insight_risk_desc: 'Factors that increase the chance of (partial) granting.',
    insight_available2: 'Available after loading data.',
    // Trust
    trust_no_data: 'No data stored', trust_indicative: 'Indicative model, not legal advice',
    trust_public: 'Public KIFID data',
    // Footer
    footer_rulings: 'KIFID Rulings', footer_about: 'About KIFID',
    // Pricing / Solutions
    nav_pricing: 'Solutions',
    pricing_eyebrow: 'Solutions', pricing_title: 'Tailored to your organisation',
    pricing_desc: 'Every engagement starts with a conversation. We tailor the solution to your workflows, systems and volume.',
    pricing_pilot_name: 'Pilot', pricing_pilot_desc: 'Experience the value with a guided trial within your claims team',
    pricing_pilot_price: 'Free trial period',
    pricing_pilot_f1: '4-week guided pilot', pricing_pilot_f2: 'Risk analyses & comparable rulings',
    pricing_pilot_f3: 'Evaluation report with ROI analysis', pricing_pilot_f4: 'Personal onboarding',
    pricing_pilot_cta: 'Request pilot',
    pricing_integ_name: 'Integration', pricing_integ_desc: 'Seamlessly connected to your claims and complaint systems',
    pricing_integ_price: 'By arrangement', pricing_integ_note: 'License based on volume and scope',
    pricing_integ_f1: 'API integration with your existing systems', pricing_integ_f2: 'Unlimited analyses for your entire team',
    pricing_integ_f3: 'Policy terms scanner (PDF/API)', pricing_integ_f4: 'Insurer analytics & benchmarks',
    pricing_integ_f5: 'SSO and role-based access',
    pricing_integ_cta: 'Schedule a call',
    pricing_ent_name: 'Custom solution', pricing_ent_desc: 'For organisations wanting maximum control and custom models',
    pricing_ent_price: 'By arrangement',
    pricing_ent_f1: 'Custom models trained on your own data', pricing_ent_f2: 'On-premise or private cloud hosting',
    pricing_ent_f3: 'Dedicated customer success manager', pricing_ent_f4: 'SLA with guaranteed uptime',
    pricing_ent_f5: 'Legal compliance support',
    pricing_ent_cta: 'Contact us',
    pricing_popular: 'Most chosen',
    pricing_step1_title: 'Introduction', pricing_step1_desc: 'We discuss your situation and needs',
    pricing_step2_title: 'Pilot or demo', pricing_step2_desc: 'Experience the value in practice',
    pricing_step3_title: 'Custom proposal', pricing_step3_desc: 'Pricing based on volume and integration',
    pricing_step4_title: 'Implementation', pricing_step4_desc: 'System integration and training',
    // Insurer tab
    nav_insurer: 'My Insurer', tab_insurer: 'My Insurer',
    insurer_title: 'Insurer Analytics',
    insurer_search_placeholder: 'Search for an insurer...',
    insurer_select_prompt: 'Select an insurer',
    insurer_select_desc: 'Search and select an insurer to view detailed analytics, benchmarks and risk areas.',
    insurer_total_cases: 'Total cases',
    insurer_rejection_rate: 'Rejection rate',
    insurer_grant_rate: 'Grant rate',
    insurer_partial_rate: 'Partial %',
    insurer_vs_market: 'vs. market avg',
    insurer_benchmark_title: 'Benchmark: rejection rate vs. market average',
    insurer_this_insurer: 'This insurer',
    insurer_market_avg: 'Market average',
    insurer_type_breakdown: 'Breakdown by insurance type',
    insurer_col_type: 'Type',
    insurer_col_n: 'N',
    insurer_col_rejected: 'Rej%',
    insurer_col_granted: 'Grant%',
    insurer_col_partial: 'Partial%',
    insurer_top_disputes: 'Top dispute types',
    insurer_trend_title: 'Trend over years',
    insurer_risk_areas: 'Risk areas',
    insurer_risk_desc: 'Areas where this insurer performs worse than market average',
    insurer_no_risk: 'No risk areas identified - this insurer performs at or above market average.',
    insurer_cases_period: 'Cases in period',
    insurer_worse_than_avg: 'worse than avg',
    insurer_better_than_avg: 'better than avg',
    insurer_year: 'Year',
    insurer_rej_pct: 'Rej%',
    insurer_n_cases: 'N'
  },
  nl: {
    nav_benefits: 'Voordelen', nav_risk: 'Risico-inschatting', nav_insights: 'Inzichten',
    nav_cta: 'Start analyse',
    hero_title: 'Voorspel KIFID-uitspraken met <span class="gradient" data-i18n="hero_gradient">datagedreven precisie</span>',
    hero_gradient: 'datagedreven precisie',
    hero_sub: 'Bespaar proceskosten door vooraf te weten hoe KIFID oordeelt. Vergelijk met historische uitspraken en ontvang direct een onderbouwd advies.',
    hero_cta_html: 'Gratis analyse starten <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
    hero_cta2: 'Bekijk voordelen',
    metric_analyzed: 'Geanalyseerde uitspraken', metric_consumer: 'Consument krijgt gelijk',
    metric_insurer: 'Verzekeraar krijgt gelijk', metric_home: 'Woonhuisverzekering',
    benefits_eyebrow: 'Waarom verzekeraars dit gebruiken', benefits_title: 'Sneller, scherper en beter onderbouwd',
    benefits_desc: 'Van reactief en intuitief naar proactief en datagedreven. Risicobeheersing en reputatiebescherming, niet alleen tijdsbesparing.',
    benefit1_title: 'Snellere beoordeling van klachten', benefit1_desc: 'Vind binnen seconden vergelijkbare KiFiD-uitspraken en relevante argumenten.',
    benefit2_title: 'Minder juridische kosten', benefit2_desc: 'Voorkom onnodige procedures door vooraf te zien hoe KiFiD waarschijnlijk zal oordelen.',
    benefit3_title: 'Betere risicobeoordeling', benefit3_desc: 'Zie direct de kans dat een consument wint op basis van eerdere uitspraken.',
    benefit4_title: 'Consistente besluitvorming', benefit4_desc: 'Beslissingen gebaseerd op jurisprudentie in plaats van individuele interpretatie.',
    benefit5_title: 'Betere polisvoorwaarden', benefit5_desc: 'Analyse van uitspraken laat zien welke clausules vaak tot geschillen leiden.',
    how_eyebrow: 'Hoe het werkt', how_title: 'In 4 stappen naar een onderbouwd advies',
    step1_title: 'Casuskenmerken invullen', step1_desc: 'Type verzekering, kerngeschil, bewijspositie en overige relevante details.',
    step2_title: 'Polisvoorwaarden uploaden', step2_desc: 'Optioneel: laat AI de voorwaarden scannen op risicovolle clausules.',
    step3_title: 'AI analyseert', step3_desc: 'Het model vergelijkt uw casus met 5.922 historische KIFID-uitspraken en getrainde patronen.',
    step4_title: 'Advies ontvangen', step4_desc: 'Uitkeren of afwachten, met beslisfactoren en betrouwbaarheidsscore.',
    model_eyebrow: 'Het model', model_title: 'Getraind op 5.922 KIFID-uitspraken',
    model_desc: 'Een ensemble van drie machine learning modellen, gevalideerd met 5-fold cross-validatie.',
    arch_input: 'Casuskenmerken', arch_input_sub: '99 features + tekst',
    arch_lr_sub: 'Gestructureerde features', arch_nb_sub: 'TF-IDF tekst', arch_tfidf_sub: 'Cosine afstand',
    arch_output: 'Risico-inschatting', arch_output_sub: 'Gewogen ensemble',
    trend_title: 'Afwijzingstrend', trend_desc: 'Afwijzingspercentage per jaar. Dalende trend: consumenten krijgen vaker (deels) gelijk.',
    fi_title: 'Sterkste voorspellers', fi_desc: 'Top features uit het Logistic Regression model, gerangschikt op gewicht.',
    fi_negligent: 'Consument nalatig', fi_neg_evidence: 'Nalatigheid \u00d7 bewijs', fi_year: 'Jaar (trend)',
    fi_complaint: 'Jur: klachtplicht', fi_life: 'Overlijdensrisicoverz.', fi_limitation: 'Jur: verjaringstermijn',
    fi_report: 'Onafhankelijk rapport', fi_gdpr: 'Jur: AVG',
    fi_pro_consumer: 'Pro consument', fi_pro_insurer: 'Pro verzekeraar',
    stat_cv: 'Cross-validatie accuracy', stat_fold: 'Stratified cross-validatie',
    stat_features: 'Features (99 + 800 tekst)', stat_legal: 'Juridische termen',
    tool_eyebrow: 'Risico-inschatting', tool_title: 'Schat het risico in.',
    tool_desc: 'Voer de casuskenmerken in en ontvang direct een onderbouwde risico-inschatting.',
    tab_predict: 'Risico-inschatting', tab_data: 'Data', tab_insights: 'Inzichten',
    panel_case: 'Casuskenmerken',
    label_type: 'Type verzekering', label_dispute: 'Kerngeschil', label_amount: 'Gevorderd bedrag',
    label_binding: 'Bindend advies?', label_evidence: 'Bewijs consument', label_expert: 'Deskundigenrapport',
    label_goodwill: 'Coulanceaanbod', label_context: 'Aanvullende context', label_upload: 'Polisvoorwaarden uploaden',
    opt_select_type: 'Selecteer type', opt_auto: 'Autoverzekering', opt_home: 'Woonhuisverzekering',
    opt_contents: 'Inboedelverzekering', opt_travel: 'Reisverzekering', opt_liability: 'Aansprakelijkheidsverzekering',
    opt_legal: 'Rechtsbijstandverzekering', opt_life: 'Levensverzekering', opt_disability: 'Arbeidsongeschiktheidsverzekering',
    opt_health: 'Zorgverzekering', opt_investment: 'Beleggingsverzekering', opt_death: 'Overlijdensrisicoverzekering',
    opt_building: 'Opstalverzekering', opt_moped: 'Bromfietsverzekering', opt_fire: 'Brandverzekering',
    opt_transport: 'Transportverzekering', opt_other: 'Overig',
    opt_select_dispute: 'Selecteer geschiltype', opt_coverage: 'Dekkingsweigering',
    opt_terms: 'Uitleg polisvoorwaarden', opt_damage: 'Schadevaststelling / schadehoogte',
    opt_premium: 'Premiegeschil', opt_disclosure: 'Schending mededelingsplicht',
    opt_cancel: 'Opzegging verzekering', opt_duty: 'Schending zorgplicht',
    opt_info: 'Gebrekkige informatievoorziening', opt_exclusion: 'Beroep op uitsluiting/clausule',
    opt_delay: 'Vertraging afhandeling', opt_fraud: 'Fraude', opt_defect: 'Eigen gebrek', opt_other2: 'Overig',
    opt_binding_yes: 'Ja, bindend', opt_binding_no: 'Nee, niet-bindend',
    opt_select_evidence: 'Selecteer bewijskracht',
    opt_strong: 'Sterk (foto\'s, rapporten, getuigen)', opt_medium: 'Gemiddeld (eigen verklaring + enig bewijs)',
    opt_weak: 'Zwak (alleen eigen verklaring)',
    opt_no_report: 'Geen rapport', opt_by_consumer: 'Ingebracht door consument',
    opt_by_insurer: 'Ingebracht door verzekeraar', opt_both: 'Beide partijen',
    opt_no_goodwill: 'Geen coulance aangeboden', opt_low_goodwill: 'Ja, laag bedrag',
    opt_fair_goodwill: 'Ja, redelijk bedrag',
    placeholder_context: 'Optioneel: relevante details over de casus...',
    upload_click: 'Klik om PDF te uploaden', upload_scan: 'AI scant op risicovolle clausules',
    btn_predict: 'Analyseer & Voorspel',
    empty_state_html: 'Vul de casuskenmerken in en klik op <strong style="color:var(--text-muted);">Analyseer & Voorspel</strong>',
    panel_registry: 'KIFID Uitsprakenregister', data_analyzed: 'Geanalyseerde uitspraken',
    data_period: 'Periode', data_public: 'Publiek', data_accessible: 'Vrij toegankelijk',
    placeholder_lookup: 'Zoek op uitspraaknummer, bijv. 2024-1029', btn_lookup: 'Opzoeken',
    panel_dataset: 'Dataset beheren', data_no_data: 'Nog geen data geladen.',
    upload_files: 'Klik om bestanden te uploaden', upload_formats: 'CSV of JSON bestanden van KIFID-uitspraken',
    btn_load_full: 'Volledige dataset laden', btn_demo: 'Demo-data',
    insight_dist_title: 'Uitkomstverdeling per verzekeringstype',
    insight_dist_desc: 'Stacked bars tonen de verhouding tussen toegewezen, deels en afgewezen per type.',
    insight_load_first: 'Laad eerst data via het Data-tabblad.',
    insight_predictors_title: 'Sterkste voorspellers',
    insight_predictors_desc: 'Kerngeschillen met het hoogste afwijzingspercentage.',
    insight_available: 'Beschikbaar na laden data.',
    insight_risk_title: 'Risicofactoren',
    insight_risk_desc: 'Factoren die de kans op toewijzing verhogen.',
    insight_available2: 'Beschikbaar na laden data.',
    trust_no_data: 'Geen data opgeslagen', trust_indicative: 'Indicatief model, geen juridisch advies',
    trust_public: 'Openbare KIFID-data',
    footer_rulings: 'KIFID Uitspraken', footer_about: 'Over KIFID',
    // Pricing / Oplossingen
    nav_pricing: 'Oplossingen',
    pricing_eyebrow: 'Oplossingen', pricing_title: 'Passend bij uw organisatie',
    pricing_desc: 'Elk traject begint met een kennismaking. Wij stemmen de oplossing af op uw werkprocessen, systemen en volume.',
    pricing_pilot_name: 'Pilot', pricing_pilot_desc: 'Ervaar de meerwaarde met een begeleid proeftraject binnen uw schadeteam',
    pricing_pilot_price: 'Gratis proefperiode',
    pricing_pilot_f1: '4 weken begeleide pilot', pricing_pilot_f2: 'Risico-analyses & vergelijkbare uitspraken',
    pricing_pilot_f3: 'Evaluatierapport met ROI-analyse', pricing_pilot_f4: 'Persoonlijke onboarding',
    pricing_pilot_cta: 'Pilot aanvragen',
    pricing_integ_name: 'Integratie', pricing_integ_desc: 'Naadloos gekoppeld aan uw schadebeheer- en klachtsystemen',
    pricing_integ_price: 'In overleg', pricing_integ_note: 'Licentie op basis van volume en scope',
    pricing_integ_f1: 'API-koppeling met uw bestaande systemen', pricing_integ_f2: 'Onbeperkte analyses voor uw hele team',
    pricing_integ_f3: 'Polisvoorwaarden-scanner (PDF/API)', pricing_integ_f4: 'Verzekeraar-analytics & benchmarks',
    pricing_integ_f5: 'SSO en rolgebaseerde toegang',
    pricing_integ_cta: 'Plan een gesprek',
    pricing_ent_name: 'Maatwerkoplossing', pricing_ent_desc: 'Voor organisaties die maximale controle en eigen modellen willen',
    pricing_ent_price: 'In overleg',
    pricing_ent_f1: 'Custom modellen getraind op uw eigen data', pricing_ent_f2: 'On-premise of private cloud hosting',
    pricing_ent_f3: 'Dedicated customer success manager', pricing_ent_f4: 'SLA met gegarandeerde uptime',
    pricing_ent_f5: 'Juridische compliance-ondersteuning',
    pricing_ent_cta: 'Neem contact op',
    pricing_popular: 'Meest gekozen',
    pricing_step1_title: 'Kennismakingsgesprek', pricing_step1_desc: 'We bespreken uw situatie en wensen',
    pricing_step2_title: 'Pilot of demo', pricing_step2_desc: 'Ervaar de meerwaarde in de praktijk',
    pricing_step3_title: 'Voorstel op maat', pricing_step3_desc: 'Prijs afgestemd op volume en integratie',
    pricing_step4_title: 'Implementatie', pricing_step4_desc: 'Koppeling met uw systemen en training',
    // Insurer tab
    nav_insurer: 'Mijn Verzekeraar', tab_insurer: 'Mijn Verzekeraar',
    insurer_title: 'Verzekeraar Analytics',
    insurer_search_placeholder: 'Zoek een verzekeraar...',
    insurer_select_prompt: 'Selecteer een verzekeraar',
    insurer_select_desc: 'Zoek en selecteer een verzekeraar om gedetailleerde analyses, benchmarks en risicogebieden te bekijken.',
    insurer_total_cases: 'Totaal zaken',
    insurer_rejection_rate: 'Afwijzingspercentage',
    insurer_grant_rate: 'Toewijzingspercentage',
    insurer_partial_rate: 'Deels toegewezen %',
    insurer_vs_market: 'vs. marktgemiddelde',
    insurer_benchmark_title: 'Benchmark: afwijzingspercentage vs. marktgemiddelde',
    insurer_this_insurer: 'Deze verzekeraar',
    insurer_market_avg: 'Marktgemiddelde',
    insurer_type_breakdown: 'Uitsplitsing per verzekeringstype',
    insurer_col_type: 'Type',
    insurer_col_n: 'N',
    insurer_col_rejected: 'Afw%',
    insurer_col_granted: 'Toeg%',
    insurer_col_partial: 'Deels%',
    insurer_top_disputes: 'Top geschiltypen',
    insurer_trend_title: 'Trend over jaren',
    insurer_risk_areas: 'Risicogebieden',
    insurer_risk_desc: 'Gebieden waar deze verzekeraar slechter presteert dan het marktgemiddelde',
    insurer_no_risk: 'Geen risicogebieden gevonden - deze verzekeraar presteert op of boven het marktgemiddelde.',
    insurer_cases_period: 'Zaken in periode',
    insurer_worse_than_avg: 'slechter dan gem.',
    insurer_better_than_avg: 'beter dan gem.',
    insurer_year: 'Jaar',
    insurer_rej_pct: 'Afw%',
    insurer_n_cases: 'N'
  }
};

function currentLang() {
  return document.documentElement.getAttribute('data-lang') || 'nl';
}

function t(key) {
  var lang = currentLang();
  return (I18N[lang] && I18N[lang][key]) || (I18N.nl[key]) || key;
}

function applyLang(lang) {
  document.documentElement.setAttribute('data-lang', lang);
  document.documentElement.setAttribute('lang', lang);
  localStorage.setItem('lang', lang);
  var btn = document.getElementById('langToggle');
  if (btn) btn.querySelector('.lang-label').textContent = lang === 'nl' ? 'EN' : 'NL';

  var dict = I18N[lang] || I18N.nl;

  // Update all elements with data-i18n
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    if (!dict[key]) return;
    // Keys ending in _html contain HTML
    if (key.endsWith('_html') || el.tagName === 'H1' || el.querySelector('span,svg,strong')) {
      el.innerHTML = dict[key];
    } else {
      el.textContent = dict[key];
    }
  });

  // Update placeholders
  document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el) {
    var key = el.getAttribute('data-i18n-placeholder');
    if (dict[key]) el.placeholder = dict[key];
  });

  // Update page title
  document.title = 'ClaimWise \u2014 ' + (lang === 'en' ? 'Assess the risk' : 'Schat het risico in');
}

function toggleLang() {
  var lang = currentLang() === 'nl' ? 'en' : 'nl';
  applyLang(lang);
}

function initLang() {
  var lang = localStorage.getItem('lang') || 'nl';
  applyLang(lang);
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
  var duration = target > 10000 ? 4000 : target > 100 ? 3200 : 2400;
  var start = performance.now();

  function ease(t) {
    // Cubic ease-out: snelle start, elegante vertraging
    return 1 - Math.pow(1 - t, 3);
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
  var tabs = ['predict', 'data', 'insights', 'insurer'];
  var btns = document.querySelectorAll('.tab-btn');
  btns.forEach(function(btn, i) {
    btn.classList.toggle('active', tabs[i] === tab);
  });
  document.getElementById('tab-predict').style.display = tab === 'predict' ? 'grid' : 'none';
  document.getElementById('tab-data').style.display = tab === 'data' ? 'block' : 'none';
  document.getElementById('tab-insights').style.display = tab === 'insights' ? 'block' : 'none';
  document.getElementById('tab-insurer').style.display = tab === 'insurer' ? 'block' : 'none';
  if (tab === 'insurer') {
    initInsurerTab();
  }
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
  // Duidelijke voorwaarden → meer afwijzingen → pro (goed voor verzekeraar)
  // Onduidelijke voorwaarden → minder afwijzingen → con (slecht voor verzekeraar)
  var pvdData = modelBfLookup(typeModel, 'polisvoorwaarden_duidelijk', 'true');
  var pvuData = modelBfLookup(typeModel, 'polisvoorwaarden_duidelijk', 'false');
  if (pvdData && pvuData) {
    var verschil = pvdData.afw_pct - pvuData.afw_pct;
    if (Math.abs(verschil) >= 5) {
      factors.push({ label: 'Onduidelijke voorwaarden \u2192 ' + pvuData.afw_pct + '% afw. vs. duidelijk ' + pvdData.afw_pct + '% (n=' + pvuData.n + '/' + pvdData.n + ', model)', value: '\u0394' + Math.abs(Math.round(verschil)) + '%', type: verschil > 0 ? 'pro' : 'con' });
    }
  } else if (bfAll.length >= 5) {
    var onduidelijk = bfAll.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === false; });
    var duidelijk = bfAll.filter(function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === true; });
    if (onduidelijk.length >= 3 && duidelijk.length >= 3) {
      var ondRate = afwPct(onduidelijk);
      var duiRate = afwPct(duidelijk);
      var verschil2 = duiRate - ondRate;
      if (Math.abs(verschil2) >= 10) {
        factors.push({ label: 'Onduidelijke voorwaarden \u2192 ' + ondRate + '% afw. vs. duidelijk ' + duiRate + '% (n=' + onduidelijk.length + '/' + duidelijk.length + ')', value: '\u0394' + Math.abs(verschil2) + '%', type: verschil2 > 0 ? 'pro' : 'con' });
      }
    }
  }

  // ── 7. Beslisfactoren: consument nalatig (gunstig voor verzekeraar) ──
  if (bfAll.length >= 5) {
    var nalatig = bfAll.filter(function(u) { return u.beslisfactoren.consument_nalatig === true; });
    if (nalatig.length >= 3) {
      var nalRate = afwPct(nalatig);
      var nalDelta = Math.round((nalRate - overallAfwRate) * 0.25);
      if (Math.abs(nalDelta) >= 3) {
        score += nalDelta;
        factors.push({ label: 'Nalatigheid consument \u2192 ' + nalRate + '% afw. (n=' + nalatig.length + ')', value: fmtDelta(nalDelta), type: nalDelta > 3 ? 'pro' : nalDelta < -3 ? 'con' : 'neutral' });
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
    var amountPeers = uitspraken.filter(function(u) { var b = u.bedrag_gevorderd; return b && b >= lo && b <= hi; });
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
        var kifidUrl = c.pdfUrl || ('https://www.kifid.nl/kifid-kennis-en-uitspraken/uitspraken/?SearchTerm=' + encodeURIComponent(c.nr));
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

  // ── Uitkomstverdeling per type (gesorteerd op volume) ──
  var typeCounts = {};
  uitspraken.forEach(function(u) {
    var t = u.type_verzekering || 'overig';
    if (!typeCounts[t]) typeCounts[t] = { total: 0, toegewezen: 0, deels: 0, afgewezen: 0 };
    typeCounts[t].total++;
    typeCounts[t][u.uitkomst || 'afgewezen']++;
  });

  var sortedTypes = Object.keys(typeCounts).sort(function(a, b) { return typeCounts[b].total - typeCounts[a].total; });

  var chartHTML = '<div class="ds-chart">' + sortedTypes.map(function(type) {
    var c = typeCounts[type];
    var tPct = (c.toegewezen / c.total * 100).toFixed(1);
    var dPct = (c.deels / c.total * 100).toFixed(1);
    var aPct = (c.afgewezen / c.total * 100).toFixed(1);
    return '<div class="ds-row">' +
      '<span class="ds-label">' + type + ' <span class="ds-n">' + c.total + '</span></span>' +
      '<div class="ds-bar-wrap">' +
        '<div class="ds-bar-seg ds-green" style="width:' + tPct + '%" title="Toegewezen: ' + tPct + '%"></div>' +
        '<div class="ds-bar-seg ds-amber" style="width:' + dPct + '%" title="Deels: ' + dPct + '%"></div>' +
        '<div class="ds-bar-seg ds-red" style="width:' + aPct + '%" title="Afgewezen: ' + aPct + '%"></div>' +
      '</div></div>';
  }).join('') + '</div>' +
  '<div class="ds-legend">' +
    '<span><span class="ds-leg-dot ds-green"></span>Toegewezen</span>' +
    '<span><span class="ds-leg-dot ds-amber"></span>Deels</span>' +
    '<span><span class="ds-leg-dot ds-red"></span>Afgewezen</span>' +
  '</div>';

  document.getElementById('insightChart').innerHTML = chartHTML;

  // ── Sterkste voorspellers (per kerngeschil) ──
  var disputes = {};
  uitspraken.forEach(function(u) {
    var kg = u.kerngeschil || 'overig';
    if (!disputes[kg]) disputes[kg] = { total: 0, afgewezen: 0 };
    disputes[kg].total++;
    if (u.uitkomst === 'afgewezen') disputes[kg].afgewezen++;
  });

  var dStats = Object.keys(disputes).map(function(d) {
    return { name: d, count: disputes[d].total, pct: Math.round(disputes[d].afgewezen / disputes[d].total * 100) };
  }).filter(function(d) { return d.count >= 5; }).sort(function(a, b) { return b.pct - a.pct; });

  document.getElementById('insightPredictors').innerHTML =
    '<ul class="factor-list">' +
    dStats.slice(0, 6).map(function(d) {
      return '<li><span>' + d.name.replace(/_/g, ' ') + ' <span style="color:var(--text-dim);font-size:11px;">(n=' + d.count + ')</span></span><span class="factor-tag ' + (d.pct > 85 ? 'pro' : d.pct < 80 ? 'con' : 'neutral') + '">' + d.pct + '%</span></li>';
    }).join('') +
    '</ul>' +
    '<p style="font-size:11px;color:var(--text-dim);margin-top:12px;padding-top:10px;border-top:1px solid var(--border-subtle);">Afwijzingspercentage per geschiltype. Gebaseerd op ' + uitspraken.length + ' uitspraken.</p>';

  // ── Risicofactoren ──
  var mBf = uitspraken.filter(function(u) { return u.beslisfactoren; });
  var risks = [];
  if (mBf.length > 0) {
    var riskTests = [
      { name: 'Onduidelijke polisvoorwaarden', filter: function(u) { return u.beslisfactoren.polisvoorwaarden_duidelijk === false; } },
      { name: 'Informatieplicht geschonden', filter: function(u) { return u.beslisfactoren.verzekeraar_informatieplicht_geschonden === true; } },
      { name: 'Geen coulance aangeboden', filter: function(u) { return u.beslisfactoren.coulance_aangeboden === false; } },
      { name: 'Sterk bewijs consument', filter: function(u) { return u.beslisfactoren.bewijs_consument === 'sterk'; } },
      { name: 'Consument nalatig', filter: function(u) { return u.beslisfactoren.consument_nalatig === true; } },
    ];

    riskTests.forEach(function(test) {
      var subset = mBf.filter(test.filter);
      if (subset.length >= 5) {
        var toewijzing = subset.filter(function(u) { return u.uitkomst === 'toegewezen' || u.uitkomst === 'deels'; }).length;
        risks.push({ name: test.name, pct: Math.round(toewijzing / subset.length * 100), n: subset.length });
      }
    });
  }

  risks.sort(function(a, b) { return b.pct - a.pct; });
  document.getElementById('insightRisks').innerHTML = risks.length > 0 ?
    '<ul class="factor-list">' +
    risks.map(function(r) {
      return '<li><span>' + r.name + ' <span style="color:var(--text-dim);font-size:11px;">(n=' + r.n + ')</span></span><span class="factor-tag ' + (r.pct > 20 ? 'con' : r.pct > 10 ? 'neutral' : 'pro') + '">' + r.pct + '%</span></li>';
    }).join('') +
    '</ul>' +
    '<p style="font-size:11px;color:var(--text-dim);margin-top:12px;padding-top:10px;border-top:1px solid var(--border-subtle);">% (gedeeltelijke) toewijzing bij aanwezigheid van deze factor.</p>' :
    '<p style="font-size:14px;color:var(--text-dim);">Onvoldoende data voor risicofactoren.</p>';
}

// ── KIFID Lookup ──
function lookupKifid() {
  var nr = document.getElementById('kifidLookup').value.trim();
  var el = document.getElementById('kifidLookupResult');
  if (!nr) { el.innerHTML = '<p style="font-size:13px;color:var(--amber);">Voer een uitspraaknummer in.</p>'; return; }

  var local = uitspraken.find(function(u) { return u.uitspraaknr === nr; });
  if (local) {
    var lookupUrl = local.bron_url || ('https://www.kifid.nl/kifid-kennis-en-uitspraken/uitspraken/?SearchTerm=' + encodeURIComponent(nr));
    el.innerHTML =
      '<div style="padding:16px 20px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:var(--radius);margin-top:12px;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">' +
          '<a href="' + lookupUrl + '" target="_blank" rel="noopener" style="font-size:14px;font-weight:700;color:var(--primary);">' + local.uitspraaknr + ' \u2197</a>' +
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
        '<p style="font-size:14px;color:var(--text-muted);">Niet gevonden in lokale data (' + uitspraken.length + ' uitspraken). <a href="https://www.kifid.nl/kifid-kennis-en-uitspraken/uitspraken/?SearchTerm=' + encodeURIComponent(nr) + '" target="_blank" style="color:var(--primary);">Zoek op kifid.nl</a></p>' +
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

// ══════════════════════════════════════════════════════
// ── Insurer Analytics Tab ──
// ══════════════════════════════════════════════════════

var VERZEKERAAR_GROUP_MAP = {
  'asr': 'ASR',
  'a.s.r.': 'ASR',
  'asr schadeverzekering': 'ASR',
  'asr levensverzekering': 'ASR',
  'asr basis ziektekostenverzekeringen': 'ASR',
  'asr aanvullende ziektekostenverzekeringen': 'ASR',
  'asr zorgverzekeringen': 'ASR',
  'ditzo': 'ASR',
  'achmea': 'Achmea',
  'achmea schadeverzekeringen': 'Achmea',
  'achmea pensioen- en levensverzekeringen': 'Achmea',
  'achmea zorgverzekeringen': 'Achmea',
  'interpolis': 'Achmea',
  'centraal beheer': 'Achmea',
  'centraal beheer achmea': 'Achmea',
  'fbto': 'Achmea',
  'zilveren kruis': 'Achmea',
  'zilveren kruis zorgverzekeringen': 'Achmea',
  'avero achmea': 'Achmea',
  'aegon': 'Aegon',
  'aegon schadeverzekering': 'Aegon',
  'aegon levensverzekering': 'Aegon',
  'aegon spaarkas': 'Aegon',
  'nationale-nederlanden': 'Nationale-Nederlanden',
  'nationale nederlanden': 'Nationale-Nederlanden',
  'nn': 'Nationale-Nederlanden',
  'nn schadeverzekering': 'Nationale-Nederlanden',
  'nn levensverzekering': 'Nationale-Nederlanden',
  'das': 'DAS',
  'das rechtsbijstand': 'DAS',
  'das dutch legal protection': 'DAS',
  'das nederlandse rechtsbijstand verzekeringmaatschappij': 'DAS',
  'abn amro': 'ABN AMRO',
  'abn amro bank': 'ABN AMRO',
  'abn amro schadeverzekering': 'ABN AMRO',
  'abn amro levensverzekering': 'ABN AMRO',
  'abn amro verzekeringen': 'ABN AMRO',
  'ing': 'ING',
  'ing bank': 'ING',
  'nn groep': 'Nationale-Nederlanden',
  'delta lloyd': 'Nationale-Nederlanden',
  'vivat': 'Vivat',
  'reaal': 'Vivat',
  'zwitserleven': 'Vivat',
  'unigarant': 'Unigarant',
  'allianz': 'Allianz',
  'allianz nederland': 'Allianz',
  'allianz benelux': 'Allianz',
  'generali': 'Generali',
  'generali nederland': 'Generali',
  'generali schadeverzekering': 'Generali',
  'chubb': 'Chubb',
  'chubb european group': 'Chubb',
  'arag': 'ARAG',
  'arag rechtsbijstand': 'ARAG',
  'arag se': 'ARAG',
  'klaverblad': 'Klaverblad',
  'klaverblad verzekeringen': 'Klaverblad',
  'univé': 'Unive',
  'unive': 'Unive',
  'univé verzekeringen': 'Unive',
  'bovemij': 'Bovemij',
  'bovemij verzekeringen': 'Bovemij',
  'de goudse': 'De Goudse',
  'goudse verzekeringen': 'De Goudse',
  'de goudse schadeverzekeringen': 'De Goudse',
  'de goudse levensverzekeringen': 'De Goudse',
  'menzis': 'Menzis',
  'anderzorg': 'Menzis',
  'cz': 'CZ',
  'cz groep': 'CZ',
  'cz zorgverzekeringen': 'CZ',
  'ohra': 'CZ',
  'ohra zorgverzekeringen': 'CZ',
  'vgz': 'VGZ',
  'cooperatie vgz': 'VGZ',
  'iza zorgverzekeraar': 'VGZ',
  'snp': 'SNP',
  'tvm': 'TVM',
  'tvm verzekeringen': 'TVM',
  'zürich': 'Zurich',
  'zurich': 'Zurich',
  'zurich insurance': 'Zurich',
  'hdi': 'HDI',
  'hdi global': 'HDI'
};

function normalizeVerzekeraarnaam(name) {
  if (!name) return '';
  var cleaned = name.trim();

  // Strip ", gevestigd te..." suffix
  cleaned = cleaned.replace(/,\s*gevestigd\s+te\s+.*/i, '');

  // Strip " h.o.d.n." and everything after
  cleaned = cleaned.replace(/\s+h\.o\.d\.n\..*/i, '');

  // Strip common legal suffixes for matching
  var normalized = cleaned.replace(/\s+(n\.v\.|b\.v\.|nv|bv)\.?$/i, '').trim();

  // Try lowercase lookup in group map
  var lower = normalized.toLowerCase();
  if (VERZEKERAAR_GROUP_MAP[lower]) return VERZEKERAAR_GROUP_MAP[lower];

  // Try partial match: check if normalized name starts with a known key
  var keys = Object.keys(VERZEKERAAR_GROUP_MAP);
  for (var i = 0; i < keys.length; i++) {
    if (lower.indexOf(keys[i]) === 0 || keys[i].indexOf(lower) === 0) {
      return VERZEKERAAR_GROUP_MAP[keys[i]];
    }
  }

  // Return cleaned name (with original casing from cleaned version)
  return cleaned;
}

// Cache for insurer data
var insurerDataCache = null;
var insurerListCache = null;
var selectedInsurer = null;

function buildInsurerData() {
  if (insurerDataCache && insurerDataCache._len === uitspraken.length) return insurerDataCache;

  var data = {};
  var marketTotals = { total: 0, afgewezen: 0, toegewezen: 0, deels: 0 };

  uitspraken.forEach(function(u) {
    if (!u.verzekeraar) return;
    var name = normalizeVerzekeraarnaam(u.verzekeraar);
    if (!name) return;

    if (!data[name]) {
      data[name] = { name: name, cases: [], total: 0, afgewezen: 0, toegewezen: 0, deels: 0 };
    }
    data[name].cases.push(u);
    data[name].total++;
    data[name][u.uitkomst || 'afgewezen']++;

    marketTotals.total++;
    marketTotals[u.uitkomst || 'afgewezen']++;
  });

  insurerDataCache = { insurers: data, market: marketTotals, _len: uitspraken.length };
  insurerListCache = Object.keys(data).sort(function(a, b) { return data[b].total - data[a].total; });
  return insurerDataCache;
}

function initInsurerTab() {
  if (uitspraken.length === 0) return;
  buildInsurerData();
  // If already selected, re-render
  if (selectedInsurer) {
    renderInsurerAnalytics(selectedInsurer);
  }
}

// ── Insurer Dropdown ──
function showInsurerDropdown() {
  filterInsurerDropdown(document.getElementById('insurerSearch').value);
}

function filterInsurerDropdown(query) {
  buildInsurerData();
  var dropdown = document.getElementById('insurerDropdown');
  if (!insurerListCache || insurerListCache.length === 0) {
    dropdown.innerHTML = '<div class="insurer-dropdown-item" style="color:var(--text-dim);">Geen data geladen</div>';
    dropdown.style.display = 'block';
    return;
  }

  var q = (query || '').toLowerCase().trim();
  var filtered = insurerListCache.filter(function(name) {
    return !q || name.toLowerCase().indexOf(q) !== -1;
  }).slice(0, 20);

  if (filtered.length === 0) {
    dropdown.innerHTML = '<div class="insurer-dropdown-item" style="color:var(--text-dim);">Geen resultaten</div>';
    dropdown.style.display = 'block';
    return;
  }

  var data = insurerDataCache.insurers;
  dropdown.innerHTML = filtered.map(function(name) {
    var d = data[name];
    var rejPct = d.total > 0 ? Math.round(d.afgewezen / d.total * 100) : 0;
    return '<div class="insurer-dropdown-item" onclick="selectInsurer(\'' + name.replace(/'/g, "\\'") + '\')">' +
      '<span class="insurer-dropdown-name">' + name + '</span>' +
      '<span class="insurer-dropdown-meta">' + d.total + ' ' + t('insurer_n_cases') + ' &middot; ' + rejPct + '% ' + t('insurer_rej_pct') + '</span>' +
    '</div>';
  }).join('');
  dropdown.style.display = 'block';
}

function selectInsurer(name) {
  selectedInsurer = name;
  document.getElementById('insurerSearch').value = name;
  document.getElementById('insurerDropdown').style.display = 'none';
  renderInsurerAnalytics(name);
}

// Close dropdown on outside click
document.addEventListener('click', function(e) {
  var wrap = document.querySelector('.insurer-search-wrap');
  var dropdown = document.getElementById('insurerDropdown');
  if (wrap && dropdown && !wrap.contains(e.target)) {
    dropdown.style.display = 'none';
  }
});

// ── Render full insurer analytics ──
function renderInsurerAnalytics(name) {
  var cache = buildInsurerData();
  var d = cache.insurers[name];
  var market = cache.market;
  if (!d) return;

  var contentEl = document.getElementById('insurerContent');
  var rejPct = d.total > 0 ? Math.round(d.afgewezen / d.total * 100) : 0;
  var grantPct = d.total > 0 ? Math.round(d.toegewezen / d.total * 100) : 0;
  var partialPct = d.total > 0 ? Math.round(d.deels / d.total * 100) : 0;
  var marketRejPct = market.total > 0 ? Math.round(market.afgewezen / market.total * 100) : 0;

  // Date range
  var dates = d.cases.map(function(u) { return u.datum || ''; }).filter(function(x) { return x; }).sort();
  var dateRange = dates.length > 0 ? dates[0].substring(0, 4) + ' - ' + dates[dates.length - 1].substring(0, 4) : '-';

  var rejDiff = rejPct - marketRejPct;
  var rejDiffClass = rejDiff > 0 ? 'pro' : rejDiff < 0 ? 'con' : 'neutral';
  var rejDiffLabel = rejDiff > 0 ? '+' + rejDiff + '% ' + t('insurer_worse_than_avg') : rejDiff < 0 ? rejDiff + '% ' + t('insurer_better_than_avg') : '= ' + t('insurer_market_avg');

  var html = '';

  // ── Header ──
  html += '<div class="insurer-header">' +
    '<div class="insurer-header-name">' + name + '</div>' +
    '<div class="insurer-header-meta">' + d.total + ' ' + t('insurer_cases_period') + ' &middot; ' + dateRange + '</div>' +
  '</div>';

  // ── KPI cards ──
  html += '<div class="insurer-kpi-grid">' +
    '<div class="insurer-kpi">' +
      '<div class="insurer-kpi-value">' + d.total + '</div>' +
      '<div class="insurer-kpi-label">' + t('insurer_total_cases') + '</div>' +
    '</div>' +
    '<div class="insurer-kpi">' +
      '<div class="insurer-kpi-value">' + rejPct + '%</div>' +
      '<div class="insurer-kpi-label">' + t('insurer_rejection_rate') + '</div>' +
      '<div class="insurer-kpi-delta ' + rejDiffClass + '">' + rejDiffLabel + '</div>' +
    '</div>' +
    '<div class="insurer-kpi">' +
      '<div class="insurer-kpi-value" style="color:var(--green);">' + grantPct + '%</div>' +
      '<div class="insurer-kpi-label">' + t('insurer_grant_rate') + '</div>' +
    '</div>' +
    '<div class="insurer-kpi">' +
      '<div class="insurer-kpi-value" style="color:var(--amber);">' + partialPct + '%</div>' +
      '<div class="insurer-kpi-label">' + t('insurer_partial_rate') + '</div>' +
    '</div>' +
  '</div>';

  // ── Benchmark bar ──
  html += '<div class="analysis-card insurer-section">' +
    '<div class="analysis-card-header">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>' +
      '<h3>' + t('insurer_benchmark_title') + '</h3>' +
    '</div>' +
    '<div class="insurer-benchmark">' +
      '<div class="insurer-bench-row">' +
        '<span class="insurer-bench-label">' + t('insurer_this_insurer') + '</span>' +
        '<div class="insurer-bench-bar-wrap">' +
          '<div class="insurer-bench-bar" style="width:' + rejPct + '%;background:' + (rejDiff > 5 ? 'var(--red)' : rejDiff < -5 ? 'var(--green)' : 'var(--primary)') + ';"></div>' +
        '</div>' +
        '<span class="insurer-bench-value">' + rejPct + '%</span>' +
      '</div>' +
      '<div class="insurer-bench-row">' +
        '<span class="insurer-bench-label">' + t('insurer_market_avg') + '</span>' +
        '<div class="insurer-bench-bar-wrap">' +
          '<div class="insurer-bench-bar" style="width:' + marketRejPct + '%;background:var(--text-dim);opacity:0.5;"></div>' +
        '</div>' +
        '<span class="insurer-bench-value">' + marketRejPct + '%</span>' +
      '</div>' +
    '</div>' +
  '</div>';

  // ── Per verzekeringtype breakdown ──
  var typeBreakdown = {};
  d.cases.forEach(function(u) {
    var tp = u.type_verzekering || 'overig';
    if (!typeBreakdown[tp]) typeBreakdown[tp] = { total: 0, afgewezen: 0, toegewezen: 0, deels: 0 };
    typeBreakdown[tp].total++;
    typeBreakdown[tp][u.uitkomst || 'afgewezen']++;
  });
  var sortedTypes = Object.keys(typeBreakdown).sort(function(a, b) { return typeBreakdown[b].total - typeBreakdown[a].total; });

  // Market breakdown for comparison
  var marketTypeBreakdown = {};
  uitspraken.forEach(function(u) {
    if (!u.verzekeraar) return;
    var tp = u.type_verzekering || 'overig';
    if (!marketTypeBreakdown[tp]) marketTypeBreakdown[tp] = { total: 0, afgewezen: 0 };
    marketTypeBreakdown[tp].total++;
    if (u.uitkomst === 'afgewezen') marketTypeBreakdown[tp].afgewezen++;
  });

  html += '<div class="analysis-card insurer-section">' +
    '<div class="analysis-card-header">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>' +
      '<h3>' + t('insurer_type_breakdown') + '</h3>' +
    '</div>' +
    '<div class="insurer-table-wrap">' +
    '<table class="insurer-table">' +
      '<thead><tr>' +
        '<th>' + t('insurer_col_type') + '</th>' +
        '<th>' + t('insurer_col_n') + '</th>' +
        '<th>' + t('insurer_col_rejected') + '</th>' +
        '<th>' + t('insurer_col_granted') + '</th>' +
        '<th>' + t('insurer_col_partial') + '</th>' +
      '</tr></thead><tbody>' +
      sortedTypes.map(function(tp) {
        var c = typeBreakdown[tp];
        var tRej = c.total > 0 ? Math.round(c.afgewezen / c.total * 100) : 0;
        var tGrant = c.total > 0 ? Math.round(c.toegewezen / c.total * 100) : 0;
        var tPartial = c.total > 0 ? Math.round(c.deels / c.total * 100) : 0;
        var mktData = marketTypeBreakdown[tp];
        var mktRej = mktData && mktData.total > 0 ? Math.round(mktData.afgewezen / mktData.total * 100) : 0;
        var worse = tRej > mktRej + 5;
        return '<tr' + (worse ? ' class="insurer-row-risk"' : '') + '>' +
          '<td>' + tp + '</td>' +
          '<td>' + c.total + '</td>' +
          '<td>' + tRej + '%</td>' +
          '<td>' + tGrant + '%</td>' +
          '<td>' + tPartial + '%</td>' +
        '</tr>';
      }).join('') +
    '</tbody></table></div>' +
  '</div>';

  // ── Top geschiltypen ──
  var disputeBreakdown = {};
  d.cases.forEach(function(u) {
    var kg = u.kerngeschil || 'overig';
    if (!disputeBreakdown[kg]) disputeBreakdown[kg] = { total: 0, afgewezen: 0 };
    disputeBreakdown[kg].total++;
    if (u.uitkomst === 'afgewezen') disputeBreakdown[kg].afgewezen++;
  });
  var sortedDisputes = Object.keys(disputeBreakdown).sort(function(a, b) { return disputeBreakdown[b].total - disputeBreakdown[a].total; }).slice(0, 8);

  html += '<div class="analysis-card insurer-section">' +
    '<div class="analysis-card-header">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' +
      '<h3>' + t('insurer_top_disputes') + '</h3>' +
    '</div>' +
    '<ul class="factor-list">' +
    sortedDisputes.map(function(kg) {
      var c = disputeBreakdown[kg];
      var pct = c.total > 0 ? Math.round(c.afgewezen / c.total * 100) : 0;
      return '<li><span>' + kg.replace(/_/g, ' ') + ' <span style="color:var(--text-dim);font-size:11px;">(n=' + c.total + ')</span></span><span class="factor-tag ' + (pct > 85 ? 'pro' : pct < 75 ? 'con' : 'neutral') + '">' + pct + '% afw.</span></li>';
    }).join('') +
    '</ul>' +
  '</div>';

  // ── Trend over jaren ──
  var yearData = {};
  d.cases.forEach(function(u) {
    var y = (u.datum || '').substring(0, 4);
    if (!y || y.length !== 4) return;
    if (!yearData[y]) yearData[y] = { total: 0, afgewezen: 0 };
    yearData[y].total++;
    if (u.uitkomst === 'afgewezen') yearData[y].afgewezen++;
  });
  var years = Object.keys(yearData).sort();

  if (years.length >= 2) {
    var maxYearN = 0;
    years.forEach(function(y) { if (yearData[y].total > maxYearN) maxYearN = yearData[y].total; });

    html += '<div class="analysis-card insurer-section">' +
      '<div class="analysis-card-header">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>' +
        '<h3>' + t('insurer_trend_title') + '</h3>' +
      '</div>' +
      '<div class="insurer-table-wrap"><table class="insurer-table">' +
        '<thead><tr><th>' + t('insurer_year') + '</th><th>' + t('insurer_n_cases') + '</th><th>' + t('insurer_rej_pct') + '</th><th></th></tr></thead>' +
        '<tbody>' +
        years.map(function(y) {
          var yd = yearData[y];
          var yRej = yd.total > 0 ? Math.round(yd.afgewezen / yd.total * 100) : 0;
          var barW = yd.total > 0 ? Math.round(yd.total / maxYearN * 100) : 0;
          return '<tr><td>' + y + '</td><td>' + yd.total + '</td><td>' + yRej + '%</td>' +
            '<td><div class="insurer-mini-bar" style="width:' + barW + '%;background:' + (yRej > marketRejPct + 5 ? 'var(--red)' : yRej < marketRejPct - 5 ? 'var(--green)' : 'var(--primary)') + ';"></div></td></tr>';
        }).join('') +
      '</tbody></table></div>' +
    '</div>';
  }

  // ── Risicogebieden ──
  var riskAreas = [];
  sortedTypes.forEach(function(tp) {
    var c = typeBreakdown[tp];
    if (c.total < 3) return;
    var insRej = Math.round(c.afgewezen / c.total * 100);
    var mktData = marketTypeBreakdown[tp];
    var mktRej = mktData && mktData.total >= 5 ? Math.round(mktData.afgewezen / mktData.total * 100) : null;
    if (mktRej !== null && insRej < mktRej - 5) {
      riskAreas.push({ type: tp, insRej: insRej, mktRej: mktRej, diff: mktRej - insRej, n: c.total });
    }
  });
  riskAreas.sort(function(a, b) { return b.diff - a.diff; });

  html += '<div class="analysis-card insurer-section">' +
    '<div class="analysis-card-header">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' +
      '<h3>' + t('insurer_risk_areas') + '</h3>' +
    '</div>' +
    '<p class="analysis-card-desc">' + t('insurer_risk_desc') + '</p>';

  if (riskAreas.length > 0) {
    html += '<ul class="factor-list">' +
      riskAreas.map(function(r) {
        return '<li class="insurer-risk-item">' +
          '<span>' + r.type + ' <span style="color:var(--text-dim);font-size:11px;">(n=' + r.n + ')</span></span>' +
          '<span><span class="factor-tag con">' + r.insRej + '% vs. ' + r.mktRej + '% mkt</span> <span style="font-size:11px;color:var(--red);font-weight:600;">-' + r.diff + '%</span></span>' +
        '</li>';
      }).join('') +
    '</ul>';
  } else {
    html += '<p style="font-size:14px;color:var(--green);padding:12px 0;">' + t('insurer_no_risk') + '</p>';
  }
  html += '</div>';

  contentEl.innerHTML = html;
}
