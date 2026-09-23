// Conveniences for every page. Nothing here is needed to read or navigate the
// site: the header, the docs sidebar, the contents list, and the previous and
// next links are plain HTML written at build time. This adds a theme toggle,
// copy buttons on code blocks, copy-on-click for section links, a highlight on
// the contents entry for the section in view, and folds the navigation away
// on screens too narrow to show it beside the text.
(function () {
  "use strict";

  var doc = document.documentElement;

  /* ---- a status line, shown briefly and read by screen readers ---------- */

  var toast = document.createElement("div");
  toast.className = "toast";
  toast.setAttribute("role", "status");
  document.body.appendChild(toast);
  var toastTimer = null;
  function announce(message) {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.classList.remove("show"); }, 1800);
  }

  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.className = "offscreen";
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      area.remove();
      if (ok) resolve(); else reject(new Error("copy refused"));
    });
  }

  /* ---- theme: auto, light, dark ----------------------------------------- */

  var KEY = "plumbline-theme";
  var MODES = ["auto", "light", "dark"];
  function storedMode() {
    try {
      var mode = localStorage.getItem(KEY);
      return MODES.indexOf(mode) > 0 ? mode : "auto";
    } catch (e) {
      return "auto";
    }
  }
  function applyMode(mode) {
    if (mode === "auto") doc.removeAttribute("data-theme");
    else doc.setAttribute("data-theme", mode);
    try {
      if (mode === "auto") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, mode);
    } catch (e) {
      // Storage is blocked: the choice lasts for this page only.
    }
  }
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    var mode = storedMode();
    var label = function () {
      toggle.textContent = "Theme: " + mode;
      toggle.title = "Switch between the system setting, light, and dark";
    };
    label();
    toggle.hidden = false;
    toggle.addEventListener("click", function () {
      mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
      applyMode(mode);
      label();
      announce(mode === "auto" ? "Theme follows the system setting" : "Theme: " + mode);
    });
  }

  /* ---- copy buttons on code blocks -------------------------------------- */

  Array.prototype.forEach.call(document.querySelectorAll(".prose pre"), function (pre) {
    var wrap = document.createElement("div");
    wrap.className = "code-block";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy this code");
    wrap.appendChild(button);
    button.addEventListener("click", function () {
      var code = pre.querySelector("code") || pre;
      copy(code.textContent.replace(/\n$/, "")).then(
        function () { announce("Copied"); },
        function () { announce("Could not copy; select the text instead"); }
      );
    });
  });

  /* ---- a section's link copies itself ----------------------------------- */

  document.addEventListener("click", function (event) {
    var anchor = event.target.closest && event.target.closest("a.anchor");
    if (!anchor) return;
    var url = location.href.split("#")[0] + anchor.getAttribute("href");
    copy(url).then(function () { announce("Link to this section copied"); }, function () {});
  });

  /* ---- fold the navigation away where it cannot sit beside the text ----- */

  // Matches the breakpoints in docs.css: the sidebar sits beside the text from
  // 64rem, the contents list from 80rem. Below those they are collapsible.
  [[".docs-nav", "(min-width: 64rem)"], [".toc", "(min-width: 80rem)"]].forEach(function (pair) {
    var details = document.querySelector("details" + pair[0]);
    if (!details || !window.matchMedia) return;
    var wide = window.matchMedia(pair[1]);
    var fit = function () { details.open = wide.matches; };
    fit();
    if (wide.addEventListener) wide.addEventListener("change", fit);
  });

  /* ---- highlight the section in view ------------------------------------ */

  var contents = document.querySelector('nav[aria-label="On this page"]');
  if (contents) {
    var entries = [];
    Array.prototype.forEach.call(contents.querySelectorAll('a[href^="#"]'), function (link) {
      if (link.closest(".toc-top")) return;
      var target = document.getElementById(decodeURIComponent(link.getAttribute("href").slice(1)));
      if (target) entries.push({ link: link, target: target });
    });
    var current = null;
    var header = document.querySelector(".site-header");
    var spy = function () {
      var offset = (header && getComputedStyle(header).position === "sticky" ? header.offsetHeight : 0) + 24;
      var active = null;
      for (var i = 0; i < entries.length; i++) {
        if (entries[i].target.getBoundingClientRect().top - offset <= 0) active = entries[i];
        else break;
      }
      // At the very bottom the last sections may never reach the top; the
      // reader is at the end, so the last entry is the one in view.
      var bottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
      if (bottom && entries.length) active = entries[entries.length - 1];
      if (active === current) return;
      if (current) current.link.removeAttribute("aria-current");
      if (active) active.link.setAttribute("aria-current", "location");
      current = active;
      // Keep the entry in view when the contents list scrolls on its own.
      var rail = contents.closest(".toc");
      if (active && rail && rail.scrollHeight > rail.clientHeight) {
        var link = active.link.getBoundingClientRect();
        var box = rail.getBoundingClientRect();
        if (link.top < box.top) rail.scrollTop -= box.top - link.top + 16;
        else if (link.bottom > box.bottom) rail.scrollTop += link.bottom - box.bottom + 16;
      }
    };
    var queued = false;
    window.addEventListener("scroll", function () {
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () { queued = false; spy(); });
    }, { passive: true });
    spy();
  }
})();
