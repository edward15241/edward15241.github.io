# Personal Archive

## This prompt is used for Claude

個人的長期保存空間：公開技術研究、科技週報、日文學習、旅行紀錄、日誌與創作。
純 HTML / CSS，沒有建置鏈與相依套件，Python 只負責整理列表。

網站：<https://edward15241.github.io>

## 檔案結構

```
index.html              首頁
research/               研究筆記（以主題為單位，各有 landing page）
  voice-agent/
weekly/
  tech/<年>/            科技週報（由定期工作自動發布）
  japanese/<年>/        日文週報
travel/                 旅行
journal/<年>/           日誌
writing/                創作
archive/                依年份的全站索引
templates/              寫新文章用的模板
scripts/                整理用的 Python
assets/css/main.css     樣式
assets/js/main.js       深色模式切換、頁尾年份
assets/images/          圖片
```

## 新增一篇文章

1. 複製 `templates/article.html` 到對應的分類資料夾，
   檔名用 `YYYY-MM-DD-標題.html`
2. 改掉開頭的 metadata（`title` / `date` / `category` 是必要的）與內文
3. 跑 `python3 scripts/build.py`

metadata 長這樣，就寫在 `<main>` 裡面最前面：

```html
<!--
title: 文章標題
date: 2026-09-03
category: research
visibility: public
tags: voice-agent, router
summary: 一句話摘要
-->
```

`category` 用資料夾路徑（`research`、`weekly/tech`、`travel`…）。
`visibility` 不是 `public` 的文章不會出現在任何列表。

## 工具

```
scripts/build.py           重建導覽列與所有列表，並檢查站內連結
scripts/import_weekly.py   把外部來源的週報 HTML 轉成站內頁面
scripts/shell.py           文章頁面外殼（上面兩支共用）
```

`build.py` 掃描全站文章的 metadata，重新產生：

- 首頁的 Recent Updates 與 This Week
- 各分類頁的文章列表
- `archive/` 的年份索引
- 每一頁的導覽列

這些內容都在 `<!-- BUILD:... -->` 與 `<!-- /BUILD -->` 之間，
**不要手動編輯**，下次跑 build.py 會被蓋掉。要改選單改 `scripts/build.py` 的 `NAV`。

```bash
python3 scripts/build.py           # 重建
python3 scripts/build.py --check   # 只看會改什麼，不寫檔
```

週報若由定期工作自動發布，作業說明與注意事項見
[`docs/自動更新.md`](docs/自動更新.md)。

## 本機預覽

因為連結是相對路徑，用瀏覽器直接開檔案大致可行，但建議起個伺服器：

```bash
python3 -m http.server 8000
# 開 http://localhost:8000
```

VS Code 的 Live Server 擴充套件也可以。

## 設計原則

Readable、Simple、Fast、Long-lived、Low dependency。

- 本文寬度約 736px，首頁約 1088px，大量留白
- 標題用 serif，介面用 sans-serif
- 不把所有東西都做成圓角卡片
- JavaScript 只做漸進增強，關掉也能完整閱讀
- 不用 React / Next.js / npm build chain / 資料庫

想換配色，改 `assets/css/main.css` 最上面的 `--accent`。
深色模式的對應值在同一個檔案的 `@media (prefers-color-scheme: dark)`
與 `:root[data-theme="dark"]` 區塊。

## 內容邊界

這個 repo 是 **public**，網站任何人都看得到，搜尋引擎也會收錄。

只放公開論文、公開模型、公開資料與個人的通用技術理解，並且加註一些看法。
不得將任何客戶、公司資訊、公司機密、產品架構或任意個資內容洩漏至此。

git 歷史是永久的——公開 repo 一旦推上去，之後刪檔案也還原得回來，
所以「先放上去、之後再刪」在這裡沒有用。

照片同理：不想公開的不要放進 `assets/images/`。
上傳前記得壓縮（長邊 2000px、JPG 品質 80），並留意手機照片內嵌的 GPS 座標。

## 發布

推到 `main` 後 GitHub Pages 會自動建置，約 30 秒到 1 分鐘上線。

```bash
git add -A
git commit -m "說明"
git push origin main
```
