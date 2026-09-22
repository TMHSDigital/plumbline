// Holds site/floor.js to the golden values the Python exported.
//
// Runs on Node's standard library alone, with no package install:
//
//     node scripts/check_floor_parity.mjs
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
if (failures.length) {
  console.error(`\n${failures.length} disagreement(s) with the Python:\n`);
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("floor.js agrees with the Python.");
