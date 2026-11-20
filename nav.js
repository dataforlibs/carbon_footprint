/* Shared navigation bar for the Carbon Footprint Research Analysis site.
   Include on any page with:  <script src="nav.js" defer></script>   (use ../nav.js from a sub-folder).
   One place to edit the menu; links are relative, so it works on GitHub Pages and from a local server. */
(function () {
  var s = document.currentScript, base = s ? s.src.replace(/nav\.js(\?.*)?$/, '') : '';
  var PAPER = 'https://doi.org/10.1162/qss.a.508';
  var ITEMS = [
    ['Overview', 'index.html'],
    ["Countries' Lifepaths", 'lifepaths/countries_lifepaths.html'],
    ["Authors' Lifepaths", 'lifepaths/author_lifepaths.html'],
    ['Topics', 'topics/index.html'],
    ['Country sensitivity', 'dashboard_pipeline/country_sensitivity_dashboard.html'],
    ['Sensitivity analysis', 'lifepaths/sensitivity_dashboard.html'],
    ['Code &amp; data', 'https://github.com/dataforlibs/carbon_footprint/tree/main/package']
  ];
  var css = '.cfnav{position:relative;z-index:50;background:linear-gradient(to right,#3174f4,#3cb1fb);color:#fff;font:14px/1.3 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;box-shadow:0 2px 6px rgba(0,0,0,.15)}' +
    '.cfnav *{box-sizing:border-box}' +
    '.cfnav .in{max-width:1280px;margin:0 auto;padding:0 16px;display:flex;flex-wrap:wrap;align-items:center;gap:2px 2px}' +
    '.cfnav .brand{font-weight:700;font-size:14px;padding:12px 10px 12px 0;margin-right:2px;color:#fff;text-decoration:none;white-space:nowrap}' +
    '.cfnav a.l{color:#fff;text-decoration:none;padding:12px 8px;border-bottom:3px solid transparent;white-space:nowrap;opacity:.92}' +
    '.cfnav a.l:hover{opacity:1;background:rgba(255,255,255,.14)}' +
    '.cfnav a.l.on{opacity:1;font-weight:700;border-bottom-color:#fff}' +
    '.cfnav .sp{flex:1 1 auto}' +
    '.cfnav a.paper{color:#1d4fb8;background:#fff;text-decoration:none;font-weight:600;padding:6px 11px;white-space:nowrap;border-radius:999px;margin:6px 0}' +
    '.cfnav a.paper:hover{background:#eef4ff}' +
    '@media(max-width:700px){.cfnav .brand{width:100%;padding-bottom:2px}.cfnav a.l{padding:8px 9px;font-size:13px}.cfnav .sp{display:none}}';
  function href(p) { return /^https?:/.test(p) ? p : base + p; }
  function here(p) {
    if (/^https?:/.test(p)) return false;
    var path = location.pathname.replace(/\/+$/, '/');
    if (/\/$/.test(path)) path += 'index.html';
    return path.slice(-p.length) === p && (p !== 'index.html' || !/\/(lifepaths|topics|dashboard_pipeline)\//.test(path));
  }
  function build() {
    if (document.querySelector('.cfnav')) return;
    var st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
    var n = document.createElement('nav'); n.className = 'cfnav'; n.setAttribute('aria-label', 'Site');
    var h = '<div class="in"><a class="brand" href="' + base + 'index.html">Carbon Footprint Research</a>';
    ITEMS.forEach(function (it) { h += '<a class="l' + (here(it[1]) ? ' on' : '') + '" href="' + href(it[1]) + '"' + (/^https?:/.test(it[1]) ? ' target="_blank" rel="noopener"' : '') + (here(it[1]) ? ' aria-current="page"' : '') + '>' + it[0] + '</a>'; });
    h += '<span class="sp"></span><a class="paper" href="' + PAPER + '" target="_blank" rel="noopener">Read the paper &#8599;</a></div>';
    n.innerHTML = h; document.body.insertBefore(n, document.body.firstChild);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
