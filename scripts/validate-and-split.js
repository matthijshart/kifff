#!/usr/bin/env node
/**
 * Valideer de dataset en maak een train/test split.
 * Gebruik: node scripts/validate-and-split.js
 */

const fs = require('fs');
const path = require('path');
const { validateDataset, splitTrainTest, datasetStats } = require('./data-loader');

// Laad dataset
const dataPath = path.join(__dirname, '..', 'data', 'uitspraken', 'dataset.json');
const raw = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
const uitspraken = raw.uitspraken;

console.log('=== KIFID Dataset Validatie ===\n');

// Valideer
const validation = validateDataset(uitspraken);
if (validation.valid) {
  console.log(`Alle ${validation.total} uitspraken zijn geldig.\n`);
} else {
  console.log(`${validation.validCount}/${validation.total} uitspraken geldig.`);
  console.log('Fouten:');
  validation.errors.forEach(e => console.log(`  - ${e}`));
  console.log('');
}

// Statistieken
const stats = datasetStats(uitspraken);
console.log('=== Dataset Statistieken ===\n');
console.log(`Totaal: ${stats.totaal} uitspraken`);
console.log(`Uitkomsten: ${stats.uitkomstPercentages.toegewezen} toegewezen, ${stats.uitkomstPercentages.afgewezen} afgewezen, ${stats.uitkomstPercentages.deels} deels`);
console.log(`Bedragen: gem. \u20AC${stats.bedragen.gemiddeld.toLocaleString('nl-NL')}, mediaan \u20AC${stats.bedragen.mediaan.toLocaleString('nl-NL')}, range \u20AC${stats.bedragen.min.toLocaleString('nl-NL')} - \u20AC${stats.bedragen.max.toLocaleString('nl-NL')}`);

console.log('\nPer verzekeringstype:');
for (const [type, data] of Object.entries(stats.perType)) {
  const afwijzingsPct = (data.afgewezen / data.total * 100).toFixed(0);
  console.log(`  ${type}: ${data.total}x (${afwijzingsPct}% afgewezen)`);
}

console.log('\nPer kerngeschil:');
for (const [geschil, data] of Object.entries(stats.perGeschil)) {
  const afwijzingsPct = (data.afgewezen / data.total * 100).toFixed(0);
  console.log(`  ${geschil}: ${data.total}x (${afwijzingsPct}% afgewezen)`);
}

// Train/test split
console.log('\n=== Train/Test Split ===\n');
const { train, test, stats: splitStats } = splitTrainTest(uitspraken, 0.2, 42);
console.log(`Train: ${splitStats.train} uitspraken`);
console.log(`  Verdeling: ${JSON.stringify(splitStats.verdeling.train)}`);
console.log(`Test: ${splitStats.test} uitspraken (${splitStats.ratio})`);
console.log(`  Verdeling: ${JSON.stringify(splitStats.verdeling.test)}`);

// Sla splits op
const trainPath = path.join(__dirname, '..', 'data', 'testsets', 'train.json');
const testPath = path.join(__dirname, '..', 'data', 'testsets', 'test.json');

fs.writeFileSync(trainPath, JSON.stringify({ meta: { type: 'train', aantal: train.length, datum: new Date().toISOString().split('T')[0] }, uitspraken: train }, null, 2));
fs.writeFileSync(testPath, JSON.stringify({ meta: { type: 'test', aantal: test.length, datum: new Date().toISOString().split('T')[0] }, uitspraken: test }, null, 2));

console.log(`\nOpgeslagen: ${trainPath}`);
console.log(`Opgeslagen: ${testPath}`);
