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


/* 相簿：點縮圖放大檢視 */
(function () {
  var box = document.getElementById('lightbox');
  var img = document.getElementById('lightboxImg');
  if (!box || !img) { return; }

  var closeBtn = box.querySelector('.lightbox__close');
  var lastFocused = null;

  function open(src, alt) {
    lastFocused = document.activeElement;
    img.src = src;
    img.alt = alt || '';
    box.hidden = false;
    document.body.style.overflow = 'hidden';   /* 放大時鎖住背景捲動 */
    if (closeBtn) { closeBtn.focus(); }
  }

  function close() {
    box.hidden = true;
    img.src = '';
    document.body.style.overflow = '';
    if (lastFocused) { lastFocused.focus(); }   /* 焦點還給剛剛點的那張縮圖 */
  }

  document.querySelectorAll('.shot').forEach(function (shot) {
    shot.addEventListener('click', function () {
      var thumb = shot.querySelector('img');
      open(shot.dataset.full || (thumb && thumb.src), thumb && thumb.alt);
    });
  });

  /* 點背景任一處關閉；點圖片本身不關，方便看細節 */
  box.addEventListener('click', function (e) {
    if (e.target !== img) { close(); }
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !box.hidden) { close(); }
  });
})();
