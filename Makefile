all: presentations index

presentations: $(wildcard docs/*.md)
	pandoc --standalone -t revealjs -o $(patsubst docs/%.md, docs/%.html,$?) docs/*.md

index: presentations
	cd docs ;	tree -H '.' -L 1 --noreport --charset utf-8 -o index.html
