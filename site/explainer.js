// The explainer's calculator, worked example, and planner. The arithmetic is
// floor.js; this file only reads the forms and draws the results.
(function () {
  "use strict";
  var F = window.PlumblineFloor;
  var fmt = F.fixed4;
  function $(id) { return document.getElementById(id); }
  function text(id, value) { $(id).textContent = value; }
  function el(tag, content, className) {
    var node = document.createElement(tag);
    if (content !== undefined) node.textContent = content;
    if (className) node.className = className;
    return node;
  }
  function row(cells) {
    var tr = document.createElement("tr");
    cells.forEach(function (cell) { tr.appendChild(el("td", cell)); });
    return tr;
  }
  function binLabel(lower, upper) {
    var digits = upper - lower < 0.1 - 1e-9 ? 2 : 1;
    return lower.toFixed(digits) + "-" + upper.toFixed(digits);
  }

  /* ---- worker ---------------------------------------------------------- */

  // A worker keeps the page responsive on large n. Where the browser refuses
  // one (a page opened from disk, for example), the same code runs here.
  var worker = null;
  var pending = {};
  var nextId = 0;
  try {
    worker = new Worker("floor-worker.js");
    worker.onmessage = function (event) {
      var m = event.data;
      var job = pending[m.id];
      if (!job) return;
      if (m.progress !== undefined) job.onProgress(m.progress);
      else if (m.step !== undefined) job.onStep(m.step);
      else {
        delete pending[m.id];
        if (m.error) job.reject(new Error(m.error));
        else job.resolve(m.result);
      }
    };
    worker.onerror = function (event) {
      event.preventDefault();
      worker = null;
      Object.keys(pending).forEach(function (id) { runHere(pending[id]); delete pending[id]; });
    };
  } catch (e) {
    worker = null;
  }

  function runHere(job) {
    setTimeout(function () {
      try {
        if (job.message.kind === "synthetic") {
          job.resolve(F.syntheticFloor(job.message.n, job.message.nBins, job.message.accuracy));
        } else {
          job.resolve(F.planRows(job.message.target, job.message.nBins, job.message.accuracy, { onStep: job.onStep }));
        }
      } catch (error) { job.reject(error); }
    }, 20);
  }

  function submit(message, onProgress, onStep) {
    return new Promise(function (resolve, reject) {
      var job = { message: message, resolve: resolve, reject: reject, onProgress: onProgress || function () {}, onStep: onStep || function () {} };
      if (!worker) { runHere(job); return; }
      message.id = ++nextId;
      pending[message.id] = job;
      worker.postMessage(message);
    });
  }

  function number(input, check, message) {
    var value = Number(input.value);
    return check(value) ? { value: value } : { error: message, field: input };
  }

  function busy(prefix, on, statusText) {
    $(prefix + "-go").disabled = on;
    $(prefix + "-progress").hidden = !on;
    $(prefix + "-progress").value = 0;
    var status = $(prefix + "-status");
    status.classList.remove("error");
    status.textContent = statusText || "";
  }

  function fail(prefix, error) {
    var status = $(prefix + "-status");
    status.textContent = error.message || String(error);
    status.classList.add("error");
    if (error.field) error.field.focus();
  }

  var isWhole = function (lo, hi) { return function (v) { return Number.isInteger(v) && v >= lo && v <= hi; }; };
  var isOpenUnit = function (v) { return v > 0 && v < 1; };

  /* ---- calculator ------------------------------------------------------ */

  $("calc").addEventListener("submit", function (event) {
    event.preventDefault();
    var form = event.target;
    var fields = [
      number(form.n, isWhole(1, 100000), "Rows must be a whole number from 1 to 100,000."),
      number(form.bins, isWhole(1, 1000), "Bins must be a whole number from 1 to 1,000."),
      number(form.accuracy, isOpenUnit, "Accuracy must lie strictly between 0 and 1."),
    ];
    var measuredText = form.measured.value.trim();
    var measured = null;
    if (measuredText !== "") {
      var m = number(form.measured, function (v) { return v >= 0 && v <= 1; }, "Measured ECE must lie between 0 and 1.");
      fields.push(m);
      measured = m.value;
    }
    for (var i = 0; i < fields.length; i++) {
      if (fields[i].error) { busy("calc", false); fail("calc", fields[i]); return; }
    }
    var input = { n: fields[0].value, nBins: fields[1].value, accuracy: fields[2].value, measured: measured };
    busy("calc", true, "Running 2,000 bootstrap resamples...");
    submit({ kind: "synthetic", n: input.n, nBins: input.nBins, accuracy: input.accuracy },
      function (f) { $("calc-progress").value = f; })
      .then(function (band) { showCalc(band, input); busy("calc", false, "Done."); })
      .catch(function (error) { busy("calc", false); fail("calc", error); });
  });

  function showCalc(band, input) {
    text("calc-mean", fmt(band.mean));
    text("calc-p95", fmt(band.p95));
    text("calc-config", band.n + " rows, " + band.nBins + " bins, accuracy " + input.accuracy);
    text("calc-read", "A perfectly calibrated model scores " + fmt(band.mean) + " on average at this size, and above " +
      fmt(band.p95) + " one time in twenty. A measured ECE at or below " + fmt(band.p95) +
      " cannot be told apart from calibrated on this many rows.");
    var verdict = $("calc-verdict");
    if (input.measured === null) {
      verdict.hidden = true;
    } else {
      var clears = F.isDistinguishable(input.measured, band);
      verdict.className = "verdict " + (clears ? "distinguishable" : "inconclusive");
      verdict.replaceChildren(
        el("strong", clears ? "Distinguishable from sampling noise" : "Inconclusive, which is not a pass"),
        el("span", F.statement(input.measured, band))
      );
      verdict.hidden = false;
    }
    $("calc-result").hidden = false;
  }

  /* ---- worked example -------------------------------------------------- */

  function histogram(svg, draws, marks) {
    var left = 6, right = 394, top = 40, bottom = 172;
    var hi = Math.max.apply(null, draws.concat(marks.map(function (m) { return m.value; }))) * 1.08;
    var nb = 40;
    var counts = new Array(nb).fill(0);
    draws.forEach(function (d) { counts[Math.min(nb - 1, Math.floor(d / hi * nb))]++; });
    var peak = Math.max.apply(null, counts);
    var x = function (v) { return left + (v / hi) * (right - left); };
    var ns = "http://www.w3.org/2000/svg";
    function add(tag, attrs, content) {
      var node = document.createElementNS(ns, tag);
      Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
      if (content !== undefined) node.textContent = content;
      svg.appendChild(node);
      return node;
    }
    svg.replaceChildren();
    var bw = (right - left) / nb;
    counts.forEach(function (c, i) {
      if (!c) return;
      var h = (c / peak) * (bottom - top);
      add("rect", { class: "bar", x: left + i * bw + 0.5, y: bottom - h, width: Math.max(1, bw - 1), height: h });
    });
    add("line", { class: "axis", x1: left, x2: right, y1: bottom, y2: bottom });
    [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3].forEach(function (t) {
      if (t > hi) return;
      add("line", { class: "axis", x1: x(t), x2: x(t), y1: bottom, y2: bottom + 4 });
      add("text", { x: x(t), y: bottom + 20, "text-anchor": t === 0 ? "start" : "middle" }, t.toFixed(2));
    });
    marks.forEach(function (m, i) {
      add("line", { class: m.className, x1: x(m.value), x2: x(m.value), y1: 6 + i * 17, y2: bottom });
      // Start the label at its line when it fits to the right, else end it there.
      var anchor = x(m.value) + 4 + m.label.length * 8 < right ? "start" : "end";
      var dx = anchor === "end" ? -4 : 4;
      add("text", { class: m.className + "-label", x: x(m.value) + dx, y: 16 + i * 17, "text-anchor": anchor }, m.label);
    });
  }

  function loadExample() {
    fetch("example-run.json", { cache: "no-cache" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(showExample)
      .catch(function () {
        $("ex-report-line").replaceChildren(el("p", "The example is not available in this copy of the page.", "muted"));
        $("ex-missing").hidden = false;
      });
  }

  function showExample(ex) {
    var p = ex.probabilities, correct = ex.correct, nBins = ex.n_bins, n = p.length;
    $("ex-report-line").replaceChildren(el("p", ex.report.accuracy_line), el("p", ex.report.ece_line));

    var right = correct.reduce(function (a, b) { return a + b; }, 0);
    var accuracy = right / n;
    $("ex-rows").replaceChildren(
      document.createTextNode(n + " predictions and outcomes from the seeded mock adapter, a deterministic stand-in rather than a vendor. They were regenerated when this site was deployed, by running the command the report records: "),
      el("code", ex.command),
      document.createTextNode(". " + right + " of " + n + " were right, an accuracy of " + fmt(accuracy) + ".")
    );

    var bins = F.reliability(p, correct, nBins);
    var tbody = $("ex-bins");
    tbody.replaceChildren();
    bins.forEach(function (b) {
      tbody.appendChild(row([binLabel(b.lower, b.upper), String(b.count), fmt(b.meanProbability), fmt(b.accuracy), fmt(b.gap)]));
    });
    var measured = F.ece(p, correct, nBins);
    text("ex-n", String(n));
    text("ex-ece", fmt(measured));

    var band = F.calibrationFloor(p, nBins, { keepDraws: true });
    histogram($("ex-hist"), band.draws, [
      { value: measured, label: "measured " + fmt(measured), className: "measured" },
      { value: band.p95, label: "p95 " + fmt(band.p95), className: "p95" },
    ]);
    var beyond = band.draws.filter(function (d) { return d >= measured; }).length;
    text("ex-hist-desc", "Histogram of 2,000 resampled ECEs from a perfectly calibrated model with these 105 probabilities. Mean " +
      fmt(band.mean) + ", 95th percentile " + fmt(band.p95) + ". " + beyond + " of the 2,000 (" +
      Math.round(beyond / 20) + "%) scored at least the measured " + fmt(measured) + ".");

    var derived = F.statement(measured, band);
    var clears = F.isDistinguishable(measured, band);
    $("ex-verdict").className = "verdict " + (clears ? "distinguishable" : "inconclusive");
    $("ex-verdict").replaceChildren(el("strong", clears ? "Distinguishable from sampling noise" : "Inconclusive, which is not a pass"), el("span", derived));
    var same = derived === ex.report.ece_line;
    text("ex-match", same
      ? "Derived here, and identical to the line in the report, character for character."
      : "This derivation does not match the report line. The deploy check should have prevented this; please open an issue.");
    $("ex-steps").hidden = false;

    var summary = F.syntheticFloor(n, nBins, accuracy, { keepProbabilities: true });
    text("ex-real-floor", fmt(band.mean) + " (95th percentile " + fmt(band.p95) + ")");
    text("ex-summary-floor", fmt(summary.mean) + " (95th percentile " + fmt(summary.p95) + ")");
    var realCounts = new Array(nBins).fill(0);
    var inventedCounts = new Array(nBins).fill(0);
    bins.forEach(function (b) { realCounts[b.index] = b.count; });
    summary.probabilities.forEach(function (q) { inventedCounts[Math.min(nBins - 1, Math.floor(q * nBins))]++; });
    var spread = $("ex-spread");
    spread.replaceChildren();
    for (var k = 0; k < nBins; k++) {
      if (!realCounts[k] && !inventedCounts[k]) continue;
      spread.appendChild(row([binLabel(k / nBins, (k + 1) / nBins), String(realCounts[k]), String(inventedCounts[k])]));
    }
    $("ex-lesson").hidden = false;
  }

  loadExample();

  /* ---- planner --------------------------------------------------------- */

  $("plan").addEventListener("submit", function (event) {
    event.preventDefault();
    var form = event.target;
    var fields = [
      number(form.target, isOpenUnit, "The ECE to detect must lie strictly between 0 and 1."),
      number(form.bins, isWhole(1, 1000), "Bins must be a whole number from 1 to 1,000."),
      number(form.accuracy, isOpenUnit, "Accuracy must lie strictly between 0 and 1."),
    ];
    for (var i = 0; i < fields.length; i++) {
      if (fields[i].error) { busy("plan", false); fail("plan", fields[i]); return; }
    }
    var target = fields[0].value, nBins = fields[1].value, accuracy = fields[2].value;
    var trace = $("plan-trace");
    trace.replaceChildren();
    $("plan-result").hidden = true;
    busy("plan", true, "Searching. Large row counts take several seconds each...");
    submit({ kind: "plan", target: target, nBins: nBins, accuracy: accuracy },
      function (f) { $("plan-progress").value = f; },
      function (steps) { showTrace(steps); })
      .then(function (plan) { showPlan(plan, target, nBins, accuracy); busy("plan", false, "Done."); })
      .catch(function (error) { busy("plan", false); fail("plan", error); });
  });

  function showTrace(steps) {
    var trace = $("plan-trace");
    trace.replaceChildren();
    steps.forEach(function (s) {
      trace.appendChild(el("li", s.n.toLocaleString("en-US") + " rows: 95th percentile " + fmt(s.p95)));
    });
  }

  function showPlan(plan, target, nBins, accuracy) {
    showTrace(plan.evaluations);
    if (plan.exceeds) {
      text("plan-rows", "More than " + plan.maxRows.toLocaleString("en-US") + " rows");
      text("plan-read", "At " + plan.maxRows.toLocaleString("en-US") + " rows the summary floor's 95th percentile is " +
        fmt(plan.band.p95) + ", still above " + target + ". Extrapolating at one over the square root of n suggests roughly " +
        plan.extrapolated.toLocaleString("en-US") + " rows; this page does not compute floors that large.");
    } else {
      text("plan-rows", "About " + plan.rows.toLocaleString("en-US") + " rows");
      text("plan-read", "At " + plan.rows.toLocaleString("en-US") + " rows, " + nBins + " bins, and accuracy " + accuracy +
        ", the summary floor has mean " + fmt(plan.band.mean) + " and 95th percentile " + fmt(plan.band.p95) +
        ", so a measured ECE of " + target + " would clear it. On materially fewer rows, the same measurement would read as INCONCLUSIVE.");
    }
    $("plan-result").hidden = false;
  }
})();
