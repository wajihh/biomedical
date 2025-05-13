---
layout: default
title: Biomedical Projects Blog
---
# Welcome to My Biomedical Projects Blog

This blog showcases my work on biomedical signal processing, starting with an ECG analysis project. Check out the posts below!

{% for post in site.posts %}
  * [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %d, %Y" }}
{% endfor %}