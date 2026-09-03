#!/usr/bin/env python3
"""把週報原始 HTML 轉成站內筆記頁面。

用法：
    python3 tools/import_weekly.py 來源.html
    python3 tools/import_weekly.py 來源.html --outdir notes

支援兩種來源格式：
  A. 純標籤版：<h1> + <h2>/<h3>/<ul>/<ol>，沒有自帶樣式
  B. 卡片版　：自帶 <style>，含 .summary-box / .topic / .item / .ds-table

輸出一律是站內格式：接上 assets/css/style.css 的頁面，
結構複雜的部分包在 <div class="wr"> 底下，樣式由 style.css 提供。
轉完記得跑 tools/build_index.py 更新首頁列表。
"""

import argparse
import html
import pathlib
import re
import sys

# --------------------------------------------------------------- 共用工具

def strip_tags(s):
    return re.sub(r'<[^>]+>', '', s).strip()


def find_dates(text):
    """從標題或內文找出這期涵蓋的日期範圍，回傳 (起, 迄) 的 YYYY-MM-DD。"""
    # 2026/08/17 – 08/24 或 2026 / 08 / 17 — 08 / 24
    m = re.search(r'(\d{4})\s*/\s*(\d{2})\s*/\s*(\d{2})\s*[—–\-]\s*'
                  r'(?:(\d{4})\s*/\s*)?(\d{2})\s*/\s*(\d{2})', text)
    if m:
        y1, m1, d1, y2, m2, d2 = m.groups()
        y2 = y2 or y1
        return f'{y1}-{m1}-{d1}', f'{y2}-{m2}-{d2}'
    # 檔名樣式 2026-08-17_to_2026-08-24
    m = re.search(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1), m.group(2)
    return None, None


PAGE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="../assets/css/style.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📓</text></svg>">
</head>
<body>

<header class="site-header">
  <nav class="nav" aria-label="主選單">
    <a class="nav__brand" href="../index.html">筆記與照片</a>
    <ul class="nav__links">
      <li><a href="../index.html#notes">筆記</a></li>
      <li><a href="../index.html#photos">相簿</a></li>
    </ul>
    <button class="theme-toggle" id="themeToggle" type="button" aria-label="切換深色 / 淺色模式">
      <span aria-hidden="true">◐</span>
    </button>
  </nav>
</header>

<main>
  <article class="section">
    <div class="wrap wrap--narrow">

      <time class="note__date" datetime="{date_end}">{date_label}</time>
      <h1 class="note__title">{heading}</h1>

{body}

      <p class="note__back"><a href="../index.html#notes">← 回筆記列表</a></p>

    </div>
  </article>
</main>

<footer class="site-footer">
  <div class="wrap">
    <p>&copy; <span id="year">2026</span></p>
  </div>
</footer>

<script src="../assets/js/main.js"></script>
</body>
</html>
"""

# ------------------------------------------------- 格式 B：卡片版的類別對應

CLASS_MAP = [
    ('summary-box',    'wr-summary'),
    ('topic-header',   'wr-topic-header'),
    ('topic-title-en', 'wr-topic-title-en'),
    ('topic-title',    'wr-topic-title'),
    ('topic-num',      'wr-num'),
    ('topic',          'wr-topic'),
    ('subsection',     'wr-sub'),
    ('sub-label',      'wr-label'),
    ('item-title',     'wr-item-title'),
    ('item-meta',      'wr-meta'),
    ('item-desc',      'wr-desc'),
    ('item',           'wr-item'),
    ('no-update',      'wr-none'),
    ('ds-table',       'wr-table'),
    ('date-tag',       'wr-date'),
    ('footer',         'wr-foot'),
]


def convert_card_format(src):
    """格式 B：丟掉自帶的 <style>/<link>，把類別名改成站內的 .wr-* 系列。"""
    body = src

    # 移除自帶樣式與字型連結（站內樣式已涵蓋）
    body = re.sub(r'<style>.*?</style>', '', body, flags=re.S)
    body = re.sub(r'<link[^>]*>', '', body)
    body = re.sub(r'<title>.*?</title>', '', body, flags=re.S)

    # 只取 <div class="page"> 內部
    m = re.search(r'<div class="page">(.*)</div>\s*$', body, flags=re.S)
    if m:
        body = m.group(1)

    # 原本的 <header class="header"> 已由頁面標題取代，移掉
    body = re.sub(r'<header class="header">.*?</header>', '', body, flags=re.S)

    # 類別改名：整個 class 屬性拆成 token 逐一對應，
    # 避免用子字串比對時 topic 又去吃掉已經換好的 wr-topic-header
    mapping = dict(CLASS_MAP)

    def rename(m):
        toks = [mapping.get(t, t) for t in m.group(1).split()]
        return 'class="' + ' '.join(toks) + '"'

    body = re.sub(r'class="([^"]*)"', rename, body)

    # 表格外層的行內樣式換成類別
    body = body.replace('<div style="overflow-x:auto;">', '<div class="wr-scroll">')
    body = re.sub(r'\s*style="margin-top:12px;"', '', body)

    return body


# ------------------------------------------------- 格式 A：純標籤版

def convert_plain_format(src):
    """格式 A：拿掉 <h1> 與 artifact 連結，其餘標籤原樣保留。"""
    m = re.search(r'<body>(.*)</body>', src, flags=re.S)
    body = m.group(1) if m else src

    body = re.sub(r'<h1>.*?</h1>', '', body, flags=re.S)
    # artifact 來源是內部連結，外部讀者打不開，不收錄
    body = re.sub(r'<p>\s*Artifact\s*來源.*?</p>', '', body, flags=re.S)
    body = re.sub(r'<hr\s*/?>', '', body)
    return body


# --------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='來源 HTML 檔')
    ap.add_argument('--outdir', default='notes', help='輸出目錄（預設 notes）')
    args = ap.parse_args()

    src_path = pathlib.Path(args.source)
    src = src_path.read_text(encoding='utf-8')

    is_card = 'class="topic"' in src or 'summary-box' in src

    # 標題與日期
    m = re.search(r'<title>(.*?)</title>', src, flags=re.S)
    raw_title = strip_tags(m.group(1)) if m else src_path.stem
    heading = re.split(r'\s*[—–]\s*', raw_title)[0].strip() or '週報'

    start, end = find_dates(src[:2000] + src_path.name)
    if not end:
        print(f'!! 找不到日期範圍：{src_path.name}', file=sys.stderr)
        return 1

    date_label = f'{start} – {end[5:]}' if start else end

    body = convert_card_format(src) if is_card else convert_plain_format(src)
    body = '\n'.join('      ' + ln.strip() for ln in body.splitlines() if ln.strip())
    if is_card:
        body = '      <div class="wr">\n' + body + '\n      </div>'

    page = PAGE.format(
        title=html.escape(f'{heading} {start or ""}–{end}'.strip()),
        heading=html.escape(heading),
        date_end=end,
        date_label=date_label,
        body=body,
    )

    slug = re.sub(r'\s+', '-', heading)
    out = pathlib.Path(args.outdir) / f'{end}-{slug}.html'
    out.write_text(page, encoding='utf-8')
    print(f'  {src_path.name}  →  {out}   [{"卡片版" if is_card else "純標籤版"}]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
