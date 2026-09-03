#!/usr/bin/env python3
"""掃描內容、重建導覽列與各種列表。

用法：
    python3 scripts/build.py
    python3 scripts/build.py --check    # 只檢查不寫檔

做三件事：

1. 同步導覽列
   每個頁面的 <!-- BUILD:nav --> ... <!-- /BUILD --> 之間會被填入導覽列，
   連結會依該檔案的深度自動加上 ../ 前綴。要改選單只需改這支程式的 NAV。

2. 產生列表
   掃描所有文章開頭的 metadata 註解（見 templates/article.html），
   填入 <!-- BUILD:entries ... --> 區塊。支援三種：
     entries=<分類>   該分類的全部文章，新到舊
     recent=all       全站最近更新
     recent=<分類>    該分類的最近更新
     latest=weekly    每個週報分類各取最新一篇（首頁 This Week 用）
     entries=archive-all   全站，依年份分組

3. 檢查站內連結
   回報指向不存在檔案的 href。

只有 metadata 裡 visibility 為 public（或沒寫）的文章會被列出。
"""

import argparse
import collections
import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 導覽列：(顯示文字, 相對站根的路徑)
NAV = [
    ('Home',         'index.html'),
    ('Research',     'research/index.html'),
    ('Tech Weekly',  'weekly/tech/index.html'),
    ('Japanese',     'weekly/japanese/index.html'),
    ('Travel',       'travel/index.html'),
    ('Journal',      'journal/index.html'),
    ('Writing',      'writing/index.html'),
]

# 不當成文章掃描的目錄
SKIP_DIRS = {'assets', 'templates', 'scripts', 'docs', '.git'}

# 分類的顯示名稱
CAT_LABEL = {
    'research':         'Research',
    'weekly/tech':      'Tech Weekly',
    'weekly/japanese':  'Japanese',
    'travel':           'Travel',
    'journal':          'Journal',
    'writing':          'Writing',
}

BLOCK = re.compile(r'(<!-- BUILD:(\S+?)=(\S+?) -->)(.*?)(<!-- /BUILD -->)', re.S)
NAVBLOCK = re.compile(r'(<!-- BUILD:nav -->)(.*?)(<!-- /BUILD -->)', re.S)


# --------------------------------------------------------------- 掃描文章

Article = collections.namedtuple('Article', 'path rel date title category visibility tags')


def parse_meta(text):
    """讀取檔案開頭的 metadata 註解，回傳 dict（沒有就回 None）。"""
    m = re.search(r'<!--\s*\n\s*title:(.*?)\n\s*-->', text[:4000], re.S)
    if not m:
        return None
    meta = {}
    for line in ('title:' + m.group(1)).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip().lower()] = v.strip()
    return meta


def collect():
    arts = []
    for p in sorted(ROOT.rglob('*.html')):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in SKIP_DIRS or p.name in ('index.html', '404.html'):
            continue
        meta = parse_meta(p.read_text(encoding='utf-8'))
        if not meta:
            continue
        missing = [k for k in ('title', 'date', 'category') if not meta.get(k)]
        if missing:
            print(f'  !! {rel}：metadata 缺少 {", ".join(missing)}', file=sys.stderr)
            continue
        arts.append(Article(p, rel.as_posix(), meta['date'], meta['title'],
                            meta['category'],
                            meta.get('visibility', 'public'),
                            [t.strip() for t in meta.get('tags', '').split(',') if t.strip()]))
    arts = [a for a in arts if a.visibility == 'public']
    arts.sort(key=lambda a: (a.date, a.title), reverse=True)
    return arts


# --------------------------------------------------------------- 產生片段

def prefix_for(page_rel):
    """從某頁面回到站根要幾層 ../"""
    return '../' * (len(pathlib.PurePosixPath(page_rel).parts) - 1)


def render_nav(page_rel):
    up = prefix_for(page_rel)
    out = ['    <ul class="nav">']
    for label, target in NAV:
        cur = ' aria-current="page"' if target == page_rel else ''
        out.append(f'      <li><a href="{up}{target}"{cur}>{html.escape(label)}</a></li>')
    out.append('      <li class="nav__spacer">'
               '<button class="theme-toggle" id="themeToggle" type="button" '
               'aria-label="切換深色 / 淺色模式"><span aria-hidden="true">◐</span></button></li>')
    out.append('    </ul>')
    return '\n'.join(out)


def render_entries(items, page_rel, indent='      ', show_cat=False):
    if not items:
        return f'{indent}<p class="empty">暫無內容。</p>'
    up = prefix_for(page_rel)
    rows = []
    for a in items:
        cat = (f'<span class="entry__cat">{html.escape(CAT_LABEL.get(a.category, a.category))}</span>'
               if show_cat else '')
        rows.append(
            f'{indent}  <li class="entry">\n'
            f'{indent}    <time class="entry__date" datetime="{a.date}">{a.date}</time>\n'
            f'{indent}    <span><a class="entry__title" href="{up}{a.rel}">'
            f'{html.escape(a.title)}</a>{cat}</span>\n'
            f'{indent}  </li>')
    return f'{indent}<ul class="entries">\n' + '\n'.join(rows) + f'\n{indent}</ul>'


def render_archive(items, page_rel, indent='      '):
    if not items:
        return f'{indent}<p class="empty">暫無內容。</p>'
    by_year = collections.OrderedDict()
    for a in items:
        by_year.setdefault(a.date[:4], []).append(a)
    out = []
    for year, group in by_year.items():
        out.append(f'{indent}<h2 class="section__title">{year}</h2>')
        out.append(render_entries(group, page_rel, indent, show_cat=True))
    return '\n'.join(out)


def select(kind, arg, arts):
    if kind == 'entries' and arg == 'archive-all':
        return 'archive', arts
    if kind == 'entries':
        return 'list', [a for a in arts if a.category == arg]
    if kind == 'recent':
        pool = arts if arg == 'all' else [a for a in arts if a.category == arg]
        return 'list', pool[:8]
    if kind == 'latest':                       # 每個週報分類各取最新一篇
        seen, out = set(), []
        for a in arts:
            if a.category.startswith(arg + '/') and a.category not in seen:
                seen.add(a.category)
                out.append(a)
        return 'list', out
    print(f'  !! 不認得的區塊類型：{kind}={arg}', file=sys.stderr)
    return 'list', []


# --------------------------------------------------------------- 連結檢查

def check_links():
    bad = []
    for p in ROOT.rglob('*.html'):
        if p.relative_to(ROOT).parts[0] in {'.git'}:
            continue
        for href in re.findall(r'href="([^"]+)"', p.read_text(encoding='utf-8')):
            if href.startswith(('http://', 'https://', 'mailto:', 'data:', '#')):
                continue
            target = (ROOT if href.startswith('/') else p.parent) / href.lstrip('/').split('#')[0]
            if not target.exists():
                bad.append((p.relative_to(ROOT).as_posix(), href))
    return bad


# --------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只檢查，不寫檔')
    args = ap.parse_args()

    arts = collect()
    print(f'找到 {len(arts)} 篇文章')

    changed = 0
    for p in sorted(ROOT.rglob('*.html')):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in {'.git', 'templates'}:
            continue
        src = new = p.read_text(encoding='utf-8')
        page_rel = rel.as_posix()

        new = NAVBLOCK.sub(
            lambda m: m.group(1) + '\n' + render_nav(page_rel) + '\n    ' + m.group(3), new)

        def fill(m):
            mode, items = select(m.group(2), m.group(3), arts)
            body = (render_archive if mode == 'archive' else render_entries)(items, page_rel)
            return m.group(1) + '\n' + body + '\n      ' + m.group(5)

        new = BLOCK.sub(fill, new)

        if new != src:
            changed += 1
            if not args.check:
                p.write_text(new, encoding='utf-8')
            print(f'  {"[需更新]" if args.check else "已更新"} {page_rel}')

    bad = check_links()
    if bad:
        print(f'\n!! 斷掉的站內連結 {len(bad)} 個：', file=sys.stderr)
        for page, href in bad:
            print(f'   {page} → {href}', file=sys.stderr)
    else:
        print('\n站內連結檢查：全部正常')

    print(f'{"需更新" if args.check else "已更新"} {changed} 個檔案')
    return 1 if (bad or (args.check and changed)) else 0


if __name__ == '__main__':
    sys.exit(main())
