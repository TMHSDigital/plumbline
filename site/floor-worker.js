// Runs the floor off the main thread, so a 20,000-row bootstrap does not freeze
// the page. The arithmetic is floor.js, unchanged.
importScripts("floor.js");

self.onmessage = function (event) {
  var job = event.data;
  try {
    var band = self.PlumblineFloor.syntheticFloor(job.n, job.nBins, job.accuracy, {
      onProgress: function (fraction) {
        self.postMessage({ id: job.id, progress: fraction });
      },
    });
    self.postMessage({ id: job.id, band: band });
  } catch (error) {
    self.postMessage({ id: job.id, error: String(error.message || error) });
  }
};
