/**
 * KIFID Data Loader & Validator
 * Laadt uitspraken, valideert tegen schema, en splitst in train/test sets.
 * Kan zowel in Node.js als in de browser worden gebruikt.
 */

const VALID_TYPES = [
  "autoverzekering", "woonhuisverzekering", "inboedelverzekering",
  "reisverzekering", "aansprakelijkheidsverzekering", "rechtsbijstandverzekering",
  "levensverzekering", "arbeidsongeschiktheidsverzekering", "zorgverzekering",
  "beleggingsverzekering", "overlijdensrisicoverzekering", "opstalverzekering",
  "bromfietsverzekering", "brandverzekering", "transportverzekering", "overig"
];

const VALID_GESCHILLEN = [
  "dekkingsweigering", "uitleg_voorwaarden", "schadevaststelling",
  "premiegeschil", "mededelingsplicht", "opzegging", "zorgplicht",
  "informatievoorziening", "clausule", "vertraging", "fraude",
  "eigen_gebrek", "overig"
];

const VALID_UITKOMSTEN = ["toegewezen", "afgewezen", "deels"];

/**
 * Valideer een enkele uitspraak tegen het schema.
 * Returns { valid: boolean, errors: string[] }
 */
function validateUitspraak(u, index) {
  const errors = [];
  const prefix = `Uitspraak ${index + 1} (${u.uitspraaknr || 'onbekend'})`;

  // Verplichte velden
  if (!u.uitspraaknr || typeof u.uitspraaknr !== 'string') {
    errors.push(`${prefix}: uitspraaknr ontbreekt of is geen string`);
  } else if (!/^\d{4}-\d{3,4}$/.test(u.uitspraaknr)) {
    errors.push(`${prefix}: uitspraaknr formaat ongeldig (verwacht: YYYY-NNNN)`);
  }

  if (!u.datum) {
    errors.push(`${prefix}: datum ontbreekt`);
  }

  if (!u.type_verzekering || !VALID_TYPES.includes(u.type_verzekering)) {
    errors.push(`${prefix}: type_verzekering ongeldig: "${u.type_verzekering}"`);
  }

  if (!u.kerngeschil || !VALID_GESCHILLEN.includes(u.kerngeschil)) {
    errors.push(`${prefix}: kerngeschil ongeldig: "${u.kerngeschil}"`);
  }

  if (!u.uitkomst || !VALID_UITKOMSTEN.includes(u.uitkomst)) {
    errors.push(`${prefix}: uitkomst ongeldig: "${u.uitkomst}"`);
  }

  // Optionele numerieke velden
  if (u.bedrag_gevorderd !== undefined && (typeof u.bedrag_gevorderd !== 'number' || u.bedrag_gevorderd < 0)) {
    errors.push(`${prefix}: bedrag_gevorderd moet een positief getal zijn`);
  }

  if (u.bedrag_toegewezen !== undefined && (typeof u.bedrag_toegewezen !== 'number' || u.bedrag_toegewezen < 0)) {
    errors.push(`${prefix}: bedrag_toegewezen moet een positief getal zijn`);
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Valideer een volledige dataset.
 * Returns { valid: boolean, total: number, validCount: number, errors: string[] }
 */
function validateDataset(uitspraken) {
  const allErrors = [];
  let validCount = 0;

  uitspraken.forEach((u, i) => {
    const result = validateUitspraak(u, i);
    if (result.valid) {
      validCount++;
    } else {
      allErrors.push(...result.errors);
    }
  });

  return {
    valid: allErrors.length === 0,
    total: uitspraken.length,
    validCount,
    errors: allErrors
  };
}

/**
 * Splits dataset in train en test sets.
 * Stratified split: behoudt dezelfde uitkomst-verdeling in beide sets.
 *
 * @param {Array} uitspraken - Volledige dataset
 * @param {number} testRatio - Fractie voor test set (default 0.2 = 20%)
 * @param {number} seed - Random seed voor reproduceerbaarheid
 * @returns {{ train: Array, test: Array, stats: Object }}
 */
function splitTrainTest(uitspraken, testRatio = 0.2, seed = 42) {
  // Seeded random voor reproduceerbaarheid
  function seededRandom(s) {
    let state = s;
    return function () {
      state = (state * 1664525 + 1013904223) & 0xffffffff;
      return (state >>> 0) / 0xffffffff;
    };
  }

  const random = seededRandom(seed);

  // Groepeer per uitkomst (stratified)
  const groups = {};
  for (const u of uitspraken) {
    const key = u.uitkomst || 'onbekend';
    if (!groups[key]) groups[key] = [];
    groups[key].push(u);
  }

  const train = [];
  const test = [];

  for (const [uitkomst, items] of Object.entries(groups)) {
    // Shuffle met seeded random
    const shuffled = [...items].sort(() => random() - 0.5);
    const splitIndex = Math.max(1, Math.round(shuffled.length * testRatio));

    test.push(...shuffled.slice(0, splitIndex));
    train.push(...shuffled.slice(splitIndex));
  }

  // Stats
  const stats = {
    totaal: uitspraken.length,
    train: train.length,
    test: test.length,
    ratio: (test.length / uitspraken.length * 100).toFixed(1) + '%',
    verdeling: {
      train: countOutcomes(train),
      test: countOutcomes(test)
    }
  };

  return { train, test, stats };
}

/**
 * Tel uitkomsten in een set.
 */
function countOutcomes(uitspraken) {
  const counts = { toegewezen: 0, afgewezen: 0, deels: 0 };
  for (const u of uitspraken) {
    if (counts[u.uitkomst] !== undefined) {
      counts[u.uitkomst]++;
    }
  }
  return counts;
}

/**
 * Bereken basis-statistieken van een dataset.
 */
function datasetStats(uitspraken) {
  const byType = {};
  const byGeschil = {};
  const bedragen = [];

  for (const u of uitspraken) {
    // Per type
    if (!byType[u.type_verzekering]) byType[u.type_verzekering] = { total: 0, toegewezen: 0, afgewezen: 0, deels: 0 };
    byType[u.type_verzekering].total++;
    byType[u.type_verzekering][u.uitkomst]++;

    // Per geschil
    if (!byGeschil[u.kerngeschil]) byGeschil[u.kerngeschil] = { total: 0, toegewezen: 0, afgewezen: 0, deels: 0 };
    byGeschil[u.kerngeschil].total++;
    byGeschil[u.kerngeschil][u.uitkomst]++;

    // Bedragen
    if (u.bedrag_gevorderd > 0) bedragen.push(u.bedrag_gevorderd);
  }

  const uitkomsten = countOutcomes(uitspraken);
  const total = uitspraken.length;

  return {
    totaal: total,
    uitkomsten,
    uitkomstPercentages: {
      toegewezen: (uitkomsten.toegewezen / total * 100).toFixed(1) + '%',
      afgewezen: (uitkomsten.afgewezen / total * 100).toFixed(1) + '%',
      deels: (uitkomsten.deels / total * 100).toFixed(1) + '%'
    },
    perType: byType,
    perGeschil: byGeschil,
    bedragen: {
      gemiddeld: bedragen.length ? Math.round(bedragen.reduce((a, b) => a + b, 0) / bedragen.length) : 0,
      mediaan: bedragen.length ? bedragen.sort((a, b) => a - b)[Math.floor(bedragen.length / 2)] : 0,
      min: bedragen.length ? Math.min(...bedragen) : 0,
      max: bedragen.length ? Math.max(...bedragen) : 0
    }
  };
}

// Export voor Node.js, ook bruikbaar in browser via <script>
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { validateUitspraak, validateDataset, splitTrainTest, datasetStats, countOutcomes };
}
