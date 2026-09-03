# edward15241.github.io

我的個人網站，純靜態 HTML / CSS / JavaScript，沒有任何建置流程或相依套件。

## 檔案結構

```
index.html            # 全部頁面內容都在這裡
assets/css/style.css  # 樣式（顏色、字體集中在檔案最上方的 :root 變數）
assets/js/main.js     # 深色模式切換、頁尾年份
.nojekyll             # 告訴 GitHub Pages 不要跑 Jekyll，直接當靜態檔案送出
```

## 怎麼修改內容

打開 `index.html`，搜尋 `<!-- 改這裡` 的註解，那些就是要換成自己文字的地方。
新增專案的話，複製 `<li class="card"> ... </li>` 整個區塊再改文字即可。

想換配色，改 `assets/css/style.css` 最上面的 `--accent` 與 `--accent-hover`
兩個變數，全站顏色會一起變。深色模式的對應顏色在同一個檔案的
`@media (prefers-color-scheme: dark)` 區塊裡。

## 本機預覽

直接用瀏覽器打開 `index.html` 就看得到。想用本機伺服器的話：

```bash
python3 -m http.server 8000
# 然後開 http://localhost:8000
```

## 發布到 GitHub Pages

1. Repository 的 **Settings → General → Danger Zone → Change visibility**，
   改成 **Public**（免費方案的 private repo 無法發布 Pages）。
2. **Settings → Pages → Build and deployment**，
   Source 選 **Deploy from a branch**，分支選 `main`、資料夾選 `/ (root)`，按 Save。
3. 等一兩分鐘，網站就會出現在 <https://edward15241.github.io>。
