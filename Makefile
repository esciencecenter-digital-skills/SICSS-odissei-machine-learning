.PHONY = all presentations

all: presentations index

MD := $(wildcard docs/*.md)
HTML := $(MD:%.md=%.html)

presentations: ${HTML}

%.html: %.md
	@echo $< $@
	pandoc -s -t revealjs -o $@ $<

index: presentations
	cd docs;	tree -H '.' -L 1 --noreport --charset utf-8 -o index.html

clean:
	rm -rf ${HTML}