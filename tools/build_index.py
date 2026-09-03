#!/usr/bin/env python3
"""掃描 notes/ 重建首頁的筆記列表。

用法：
    python3 tools/build_index.py

會讀取 notes/ 底下每個 .html（跳過底線開頭的模板），
從檔案裡的 <time datetime="..."> 取日期、<h1 class="note__title"> 取標題，
然後改寫 index.html 中 <ul class="entries"> 的內容，日期新的排前面。

新增筆記後跑這支就好，不用手動改 index.html。
"""

import html
import pathlib
import re
import sys

NOTES = pathlib.Path('notes')
INDEX = pathlib.Path('index.html')


def read_note(path):
    s = path.read_text(encoding='utf-8')
    date = re.search(r'<time[^>]*datetime="([^"]+)"', s)
    title = re.search(r'<h1 class="note__title">(.*?)</h1>', s, flags=re.S)
    if not date or not title:
        print(f'  略過（缺 datetime 或 note__title）：{path.name}', file=sys.stderr)
        return None
    return date.group(1), re.sub(r'<[^>]+>', '', title.group(1)).strip(), path


def main():
    notes = [n for n in (read_note(p) for p in sorted(NOTES.glob('*.html'))
                         if not p.name.startswith('_')) if n]
    notes.sort(key=lambda n: n[0], reverse=True)   # 新的在前

    if notes:
        rows = '\n'.join(
            f'        <li class="entry">\n'
            f'          <time class="entry__date" datetime="{d}">{d}</time>\n'
            f'          <a class="entry__title" href="notes/{html.escape(p.name)}">'
            f'{html.escape(t)}</a>\n'
            f'        </li>'
            for d, t, p in notes)
        block = f'      <ul class="entries">\n{rows}\n      </ul>'
    else:
        block = '      <p class="entries__empty">暫無筆記。</p>'

    src = INDEX.read_text(encoding='utf-8')
    new, n = re.subn(
        r'      (?:<ul class="entries">.*?</ul>|<p class="entries__empty">.*?</p>)',
        lambda _: block, src, count=1, flags=re.S)
    if n != 1:
        print('!! 在 index.html 找不到筆記列表區塊', file=sys.stderr)
        return 1

    INDEX.write_text(new, encoding='utf-8')
    for d, t, p in notes:
        print(f'  {d}  {t}')
    print(f'\n首頁列表已更新，共 {len(notes)} 篇。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
