# edward15241.github.io

放筆記和照片的地方。純靜態 HTML / CSS / JavaScript，沒有建置流程與相依套件。

## 檔案結構

```
index.html              # 首頁：筆記列表 + 相簿
notes/
  _template.html        # 寫新筆記時複製這個檔
  2026-09-03-第一篇.html  # 範例筆記（可刪）
assets/
  css/style.css         # 樣式（顏色集中在檔案最上方的 :root 變數）
  js/main.js            # 深色模式切換、相簿放大檢視
  photos/               # 照片放這裡（目前是 sample-*.svg 佔位圖，可刪）
.nojekyll               # 告訴 GitHub Pages 不要跑 Jekyll，直接輸出靜態檔案
```

## 新增一篇筆記

1. 複製 `notes/_template.html`，另存成 `notes/YYYY-MM-DD-標題.html`
2. 改掉檔案裡的標題、日期，寫內容（模板註解裡有常用寫法的範例）
3. 跑 `python3 tools/build_index.py` 更新首頁列表

首頁列表是掃描 `notes/` 自動產生的，不要手動編輯——
下次跑 build_index.py 會被蓋掉。

## 工具

```
tools/build_index.py      掃描 notes/ 重建首頁列表，新增或刪除筆記後跑
tools/import_weekly.py    把外部來源的週報 HTML 轉成站內格式
```

`import_weekly.py` 支援兩種來源格式，會自動判斷：

- **純標籤版**：只有 `<h1>`/`<h2>`/`<ul>`，沒有自帶樣式
- **卡片版**：自帶 `<style>`，含 `.summary-box` / `.topic` / `.item` / `.ds-table`

兩種都會輸出成接上 `assets/css/style.css` 的站內頁面，
卡片版的結構包在 `<div class="wr">` 底下，樣式由 style.css 的「週報版型」段落提供。
來源檔開頭若有 Claude artifact 連結會自動移除（外部讀者打不開）。

```bash
python3 tools/import_weekly.py 週報.html
python3 tools/build_index.py
```

週報若由定期工作自動發布，作業說明與注意事項見
[`docs/自動更新.md`](docs/自動更新.md)。

## 新增照片

1. 圖片放進 `assets/photos/`
2. 打開 `index.html`，在相簿複製一個 `<li>` 區塊，改掉 `src`、`data-full`
   的檔名和 `alt` 說明文字

上傳前建議先壓縮，手機原圖動輒 3–5 MB，放多了 repo 會變很大
（GitHub 建議整個 repo 控制在 1 GB 以內，單檔上限 100 MB）。
長邊縮到 2000 px、存成 JPG 品質 80 左右，肉眼幾乎看不出差別，檔案卻小十倍。

**注意：這個 repo 是公開的，`assets/photos/` 裡的每張圖任何人都能直接開網址看，
搜尋引擎也會收錄。** 不想公開的照片不要放進來——包括不要先放上去之後再刪，
git 歷史會永久保留。

照片也記得留意 EXIF：手機拍的照片內嵌 GPS 座標與時間，
即使文字沒寫地點，照片本身可能標出精確位置。

## 修改外觀

改 `assets/css/style.css` 最上面的 `--accent` 與 `--accent-hover` 兩個變數，
全站主色會一起變。深色模式的對應顏色在同一個檔案的
`@media (prefers-color-scheme: dark)` 與 `:root[data-theme="dark"]` 區塊。

## 本機預覽

直接用瀏覽器打開 `index.html` 就看得到。或用本機伺服器：

```bash
python3 -m http.server 8000
# 然後開 http://localhost:8000
```

VS Code 的 **Live Server** 擴充套件也可以，改檔案會自動重整。

## 發布到 GitHub Pages

1. **Settings → General → Danger Zone → Change repository visibility** → Public
   （免費方案的 private repo 無法發布 Pages）
2. **Settings → Pages → Build and deployment**：
   Source 選 **Deploy from a branch**，分支 `main`、資料夾 `/ (root)`，Save
3. 等一兩分鐘，網站會出現在 <https://edward15241.github.io>

之後每次 `git push` 到 `main`，Pages 會自動重新發布。
