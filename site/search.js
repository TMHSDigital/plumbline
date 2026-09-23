// Search over search-index.json, which scripts/build_site.py writes from the
// explainer and the rendered docs. Pure functions, no DOM: site.js draws the
// dialog, and scripts/check_search.mjs tests the ranking under Node.
//
// Every term in the query must match, as a word or the start of one, ignoring
// case and accents. A match in a section's heading counts three times one in
// its body, and one in the page's name twice; body matches are capped per
// term so a long section cannot win on length alone. Ties keep index order,
// which is the order the pages and their sections appear on the site.
(function (root) {
  "use strict";

  function fold(text) {
    return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  }

  function tokenize(query) {
    return fold(query).split(/[^\p{L}\p{N}_]+/u).filter(Boolean);
  }

  function escape(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // Matches the term where a word starts. The flags are for counting (g) and
  // for the character classes (u).
  function termPattern(term) {
    return new RegExp("(?:^|[^\\p{L}\\p{N}_])(" + escape(term) + ")", "gu");
  }

  function count(haystack, pattern, cap) {
    pattern.lastIndex = 0;
    var n = 0;
    while (n < cap && pattern.exec(haystack)) n++;
    return n;
  }

  var BODY_CAP = 5;

  // index: [{ t, h, u, x }]. Returns up to `limit` of { entry, score }.
  function rank(index, query, limit) {
    var terms = tokenize(query);
    if (!terms.length) return [];
    var patterns = terms.map(termPattern);
    var found = [];
    for (var i = 0; i < index.length; i++) {
      var entry = index[i];
      var folded = entry._folded || (entry._folded = { t: fold(entry.t), h: fold(entry.h), x: fold(entry.x) });
      var score = 0;
      var all = true;
      for (var k = 0; k < patterns.length; k++) {
        var h = count(folded.h, patterns[k], 50);
        var t = count(folded.t, patterns[k], 50);
        var x = count(folded.x, patterns[k], BODY_CAP);
        if (!h && !t && !x) { all = false; break; }
        score += 3 * h + 2 * t + x;
      }
      if (all) found.push({ entry: entry, score: score, order: i });
    }
    found.sort(function (a, b) { return b.score - a.score || a.order - b.order; });
    return found.slice(0, limit || 20).map(function (f) { return { entry: f.entry, score: f.score }; });
  }

  // A window of `text` around the first term it contains, with the character
  // ranges of every term inside that window, for the caller to mark. Returns
  // { text, marks: [[start, end]] }; with no match, the text's opening.
  function snippet(text, query, width) {
    width = width || 160;
    var terms = tokenize(query);
    var folded = fold(text);
    // Folding can change a string's length only where it drops accents; if it
    // did, match on the lower-cased text instead so offsets stay true.
    var haystack = folded.length === text.length ? folded : text.toLowerCase();
    var first = -1;
    terms.forEach(function (term) {
      var pattern = termPattern(term);
      var m = pattern.exec(haystack);
      if (m) {
        var at = m.index + m[0].length - m[1].length;
        if (first === -1 || at < first) first = at;
      }
    });
    var start = first <= 40 ? 0 : first - 40;
    if (start > 0) {
      var space = text.lastIndexOf(" ", start);
      start = space > start - 20 ? space + 1 : start;
    }
    var end = Math.min(text.length, start + width);
    if (end < text.length) {
      var cut = text.lastIndexOf(" ", end);
      if (cut > start + width / 2) end = cut;
    }
    var window_ = text.slice(start, end);
    var windowHay = haystack.slice(start, end);
    var marks = [];
    terms.forEach(function (term) {
      var pattern = termPattern(term);
      var m;
      while ((m = pattern.exec(windowHay))) {
        var s = m.index + m[0].length - m[1].length;
        marks.push([s, s + m[1].length]);
      }
    });
    marks.sort(function (a, b) { return a[0] - b[0]; });
    var merged = [];
    marks.forEach(function (mark) {
      var last = merged[merged.length - 1];
      if (last && mark[0] <= last[1]) last[1] = Math.max(last[1], mark[1]);
      else merged.push(mark.slice());
    });
    var prefix = start > 0 ? "… " : "";
    var suffix = end < text.length ? " …" : "";
    return {
      text: prefix + window_ + suffix,
      marks: merged.map(function (m) { return [m[0] + prefix.length, m[1] + prefix.length]; }),
    };
  }

  var api = { tokenize: tokenize, rank: rank, snippet: snippet };
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.PlumblineSearch = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
