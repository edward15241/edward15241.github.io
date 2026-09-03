"""產生文章頁面的共用外殼。

導覽列留成 <!-- BUILD:nav --> 空區塊，交給 scripts/build.py 填，
這樣改選單只要改一個地方。
"""

import html

PAGE = '''<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_esc} · Personal Archive</title>
<link rel="stylesheet" href="{root}assets/css/main.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>\U0001f5c3️</text></svg>">
</head>
<body>

<a class="skip-link" href="#main">跳至主要內容</a>

<header class="masthead">
  <div class="masthead__inner">
    <p class="masthead__title"><a href="{root}index.html">Personal Archive</a></p>
    <!-- BUILD:nav -->
    <!-- /BUILD -->
  </div>
</header>

<main id="main">
  <!--
  title: {title}
  date: {date}
  category: {category}
  visibility: {visibility}
  tags: {tags}
  summary: {summary}
  -->

  <article class="article">
    <div class="prose">

      <time class="article__date" datetime="{date}">{date_label}</time>
      <h1 class="article__title">{title_esc}</h1>

{body}

      <nav class="article-nav">
        <a href="{up}index.html">← 回{cat_label}</a>
      </nav>

    </div>
  </article>
</main>

<footer class="site-foot">
  <div class="wrap">
    <p>Personal Archive &middot; <span id="year">2026</span></p>
  </div>
</footer>

<script src="{root}assets/js/main.js"></script>
</body>
</html>
'''


def render(*, title, date, date_label, category, body,
           visibility='public', tags='', summary='', depth, up, cat_label):
    """depth = 這個檔案在站內的目錄層數，用來算 ../ 前綴。"""
    return PAGE.format(
        title=title,
        title_esc=html.escape(title),
        date=date,
        date_label=date_label,
        category=category,
        visibility=visibility,
        tags=tags,
        summary=summary,
        body=body,
        root='../' * depth,
        up=up,
        cat_label=cat_label,
    )
