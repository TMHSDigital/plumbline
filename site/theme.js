// Applies the theme a reader chose on an earlier visit before the page paints,
// so a dark choice does not flash light first. site.js owns the toggle; with
// neither script the page follows the system setting.
(function () {
  try {
    var theme = localStorage.getItem("plumbline-theme");
    if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);
  } catch (e) {
    // Storage is blocked: follow the system setting.
  }
})();
