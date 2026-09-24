// Runs the floor off the main thread, so a 20,000-row bootstrap does not freeze
// the page. The arithmetic is floor.js, unchanged.
importScripts("floor.js");

self.onmessage = function (event) {
  var job = event.data;
  var F = self.PlumblineFloor;
  function progress(fraction) {
    self.postMessage({ id: job.id, progress: fraction });
  }
  try {
    var result;
    if (job.kind === "synthetic") {
      result = F.syntheticFloor(job.n, job.nBins, job.accuracy, { onProgress: progress });
    } else if (job.kind === "observed") {
      result = {
        measured: F.ece(job.probabilities, job.correct, job.nBins),
        band: F.calibrationFloor(job.probabilities, job.nBins, { onProgress: progress }),
      };
    } else if (job.kind === "plan") {
      result = F.planRows(job.target, job.nBins, job.accuracy, {
        onProgress: progress,
        onStep: function (evaluations) {
          self.postMessage({ id: job.id, step: evaluations });
        },
      });
    } else {
      throw new Error("unknown job " + job.kind);
    }
    self.postMessage({ id: job.id, result: result });
  } catch (error) {
    self.postMessage({ id: job.id, error: String(error.message || error) });
  }
};
