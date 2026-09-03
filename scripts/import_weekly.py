#!/usr/bin/env python3
"""把科技週報的原始 HTML 轉成站內頁面。

用法：
    python3 scripts/import_weekly.py 來源.html

輸出到 weekly/tech/<年>/<結束日>-<標題>.html。
轉完請跑 scripts/build.py 更新列表。

支援兩種來源格式，會自動判斷：
  A. 純標籤版：<h1> + <h2>/<h3>/<ul>/<ol>，沒有自帶樣式
  B. 卡片版　：自帶 <style>，含 .summary-box / .topic / .item / .ds-table
"""

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import shell                                              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

CLASS_MAP = {
    'summary-box':    'wr-summary',
    'topic-header':   'wr-topic-header',
    'topic-title-en': 'wr-topic-title-en',
    'topic-title':    'wr-topic-title',
    'topic-num':      'wr-num',
    'topic':          'wr-topic',
    'subsection':     'wr-sub',
    'sub-label':      'wr-label',
    'item-title':     'wr-item-title',
    'item-meta':      'wr-meta',
    'item-desc':      'wr-desc',
    'item':           'wr-item',
    'no-update':      'wr-none',
    'ds-table':       'wr-table',
    'date-tag':       'wr-date',
    'footer':         'wr-foot',
}


def find_dates(text):
    """取出這期涵蓋的日期範圍，回傳 (起, 迄)，格式 YYYY-MM-DD。"""
    m = re.search(r'(\d{4})\s*/\s*(\d{2})\s*/\s*(\d{2})\s*[—–\-]\s*'
                  r'(?:(\d{4})\s*/\s*)?(\d{2})\s*/\s*(\d{2})', text)
    if m:
        y1, m1, d1, y2, m2, d2 = m.groups()
        return f'{y1}-{m1}-{d1}', f'{y2 or y1}-{m2}-{d2}'
    m = re.search(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def convert_card(src):
    """格式 B：丟掉自帶樣式，類別改成站內的 .wr-* 系列。"""
    body = re.sub(r'<style>.*?</style>', '', src, flags=re.S)
    body = re.sub(r'<link[^>]*>|<title>.*?</title>', '', body, flags=re.S)

    m = re.search(r'<div class="page">(.*)</div>\s*\Z', body, flags=re.S)
    if m:
        body = m.group(1)
    body = re.sub(r'<header class="header">.*?</header>', '', body, flags=re.S)

    # 整個 class 屬性拆成 token 逐一對應，避免子字串互相吃掉
    body = re.sub(r'class="([^"]*)"',
                  lambda m: 'class="' + ' '.join(
                      CLASS_MAP.get(t, t) for t in m.group(1).split()) + '"',
                  body)

    body = body.replace('<div style="overflow-x:auto;">', '<div class="wr-scroll">')
    body = re.sub(r'\s*style="margin-top:12px;"', '', body)
    return body


def convert_plain(src):
    """格式 A：拿掉 <h1> 與 artifact 連結，其餘標籤原樣保留。"""
    m = re.search(r'<body>(.*)</body>', src, flags=re.S)
    body = m.group(1) if m else src
    body = re.sub(r'<h1>.*?</h1>', '', body, flags=re.S)
    # artifact 連結是內部網址，外部讀者打不開，不收錄
    body = re.sub(r'<p>\s*Artifact\s*來源.*?</p>', '', body, flags=re.S)
    return re.sub(r'<hr\s*/?>', '', body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    args = ap.parse_args()

    src_path = pathlib.Path(args.source)
    src = src_path.read_text(encoding='utf-8')
    is_card = 'class="topic"' in src or 'summary-box' in src

    m = re.search(r'<title>(.*?)</title>', src, flags=re.S)
    raw = re.sub(r'<[^>]+>', '', m.group(1)).strip() if m else src_path.stem
    title = re.split(r'\s*[—–]\s*', raw)[0].strip() or '科技週報'

    start, end = find_dates(src[:2000] + src_path.name)
    if not end:
        print(f'!! 找不到日期範圍：{src_path.name}', file=sys.stderr)
        return 1

    body = convert_card(src) if is_card else convert_plain(src)
    body = '\n'.join('      ' + ln.strip() for ln in body.splitlines() if ln.strip())
    if is_card:
        body = '      <div class="wr">\n' + body + '\n      </div>'

    page = shell.render(
        title=title,
        date=end,
        date_label=f'{start} – {end[5:]}' if start else end,
        category='weekly/tech',
        summary=f'{start} – {end} 的科技新聞、論文與 AI 發展整理。' if start else '',
        body=body,
        depth=3, up='../', cat_label='Tech Weekly',
    )

    slug = re.sub(r'\s+', '-', title)
    out = ROOT / 'weekly' / 'tech' / end[:4] / f'{end}-{slug}.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding='utf-8')
    print(f'  {src_path.name}  →  {out.relative_to(ROOT)}   '
          f'[{"卡片版" if is_card else "純標籤版"}]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
