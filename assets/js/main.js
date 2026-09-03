/* 深色 / 淺色模式切換，選擇會記在瀏覽器裡 */
(function () {
  var STORAGE_KEY = 'theme';
  var root = document.documentElement;
  var toggle = document.getElementById('themeToggle');

  function stored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  function save(value) {
    try { localStorage.setItem(STORAGE_KEY, value); } catch (e) { /* 無痕模式等情況，忽略 */ }
  }

  var saved = stored();
  if (saved === 'dark' || saved === 'light') {
    root.setAttribute('data-theme', saved);
  }

  if (toggle) {
    toggle.addEventListener('click', function () {
      var systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      var current = root.getAttribute('data-theme') || (systemDark ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      save(next);
    });
  }

  /* 頁尾年份自動更新 */
  var year = document.getElementById('year');
  if (year) { year.textContent = new Date().getFullYear(); }
})();
