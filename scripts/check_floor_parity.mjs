// Holds site/floor.js to the golden values the Python exported.
//
// Runs on Node's standard library alone, with no package install:
//
//     node scripts/check_floor_parity.mjs [_site/example-run.json]
//
// Given a built site's example-run.json, it also derives the example report's
// ECE line from the example's rows and requires it to match the report exactly.
//
// Exits non-zero on any disagreement larger than the fixture's tolerance, on any
// verdict sentence that differs by a character, and on a PCG64 starting state
// that differs from numpy's. The site deploy depends on this passing.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const floor = createRequire(import.meta.url)(join(root, "site", "floor.js"));
const fixture = JSON.parse(readFileSync(join(root, "site", "floor-golden.json"), "utf8"));
const tolerance = fixture.tolerance;

const failures = [];

for (const [seed, expected] of Object.entries(fixture.pcg64_initial_state)) {
  const actual = floor._internal.seedStates[seed];
  if (!actual || actual.state !== expected.state || actual.inc !== expected.inc) {
    failures.push(`PCG64 state for seed ${seed} differs from numpy's`);
  }
}

let worst = 0;
for (const c of fixture.cases) {
  const label = `n=${c.n} bins=${c.n_bins} accuracy=${c.accuracy}`;
  const started = Date.now();
  const band = floor.syntheticFloor(c.n, c.n_bins, c.accuracy);
  const elapsed = Date.now() - started;

  for (const key of ["mean", "p95"]) {
    const gap = Math.abs(band[key] - c[key]);
    worst = Math.max(worst, gap);
    if (!(gap <= tolerance)) {
      failures.push(`${label}: ${key} ${band[key]} against Python ${c[key]} (off by ${gap})`);
    }
  }
  for (const [measured, expected] of Object.entries(c.statements)) {
    const actual = floor.statement(Number(measured), band);
    if (actual !== expected) {
      failures.push(`${label}: verdict for ECE ${measured} differs\n  js:     ${actual}\n  python: ${expected}`);
    }
  }
  console.log(`${label.padEnd(36)} mean ${band.mean.toFixed(6)}  p95 ${band.p95.toFixed(6)}  ${elapsed}ms`);
}

console.log(`\n${fixture.cases.length} cases, largest difference ${worst.toExponential(2)}, tolerance ${tolerance}`);

// The worked example, when a built site is given: the page derives the example
// report's ECE line from the 105 rows, and it must come out character for
// character as the report prints it.
const examplePath = process.argv[2];
if (examplePath) {
  const example = JSON.parse(readFileSync(examplePath, "utf8"));
  if (example.adapter !== "mock") failures.push(`example rows come from ${example.adapter}, not the mock`);
  const measured = floor.ece(example.probabilities, example.correct, example.n_bins);
  const band = floor.calibrationFloor(example.probabilities, example.n_bins);
  const derived = floor.statement(measured, band);
  if (derived !== example.report.ece_line) {
    failures.push(`worked example: derived line differs from docs/example-report.md\n  js:     ${derived}\n  report: ${example.report.ece_line}`);
  }
  const summary = floor.syntheticFloor(example.probabilities.length, example.n_bins, example.python.accuracy);
  for (const key of ["mean", "p95"]) {
    const gap = Math.abs(summary[key] - example.python.summary_floor[key]);
    if (!(gap <= tolerance)) failures.push(`worked example: summary floor ${key} off by ${gap}`);
  }
  console.log(`worked example: ${derived.slice(0, 110)}...`);
}
if (failures.length) {
  console.error(`\n${failures.length} disagreement(s) with the Python:\n`);
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("floor.js agrees with the Python.");
