/* 漸進增強：沒有 JavaScript 時頁面仍然完整可讀。 */

/* 深色 / 淺色切換，選擇記在瀏覽器裡 */
(function () {
  var KEY = 'theme';
  var root = document.documentElement;

  try {
    var saved = localStorage.getItem(KEY);
    if (saved === 'dark' || saved === 'light') { root.setAttribute('data-theme', saved); }
  } catch (e) { /* 無痕模式等情況，忽略 */ }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('#themeToggle');
    if (!btn) { return; }
    var systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    var current = root.getAttribute('data-theme') || (systemDark ? 'dark' : 'light');
    var next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) { /* 忽略 */ }
  });
})();

/* 頁尾年份 */
(function () {
  var y = document.getElementById('year');
  if (y) { y.textContent = new Date().getFullYear(); }
})();
