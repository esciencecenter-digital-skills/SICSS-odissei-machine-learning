---
title: Presentations with Reveal.js
---

We use 1920x1080 images as slides background, so the presentation will show best view effect on 1920×1080 or same ratio screens.

## How to present?

- `s`: enable speaker view
- `f`: enable full screen mode
- `o`: enable overview mode

## How to edit markdown slides?
The markdown format is [Pandoc's markdown](https://pandoc.org/MANUAL.html#pandocs-markdown).

Check [Pandoc slides section](https://pandoc.org/MANUAL.html#slide-shows) for detailed syntax.

Command to generate the html:
```
pandoc -t revealjs -s 1-Introduction.md -o 1-Introduction.html --variable theme=white --variable revealjs-url=https://cdn.jsdelivr.net/npm/reveal.js@4.4.0
```