.PHONY = all presentations

all: presentations index

MD := $(wildcard docs/*.md)
HTML := $(MD:%.md=%.html)

presentations: ${HTML}

%.html: %.md
	@echo $< $@
	pandoc -s --mathjax -t revealjs --css docs/style.css -V theme=white -o $@ $<

index: presentations
	cd docs;	tree  -P "*.html" -I "image" -H '.' -L 1 --noreport --charset utf-8 -o index.html

clean:
	rm -rf ${HTML}