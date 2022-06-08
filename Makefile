all: presentations index

presentations: $(wildcard docs/*.md)
	mkdir presentations/
	pandoc --standalone -t revealjs -o $(patsubst docs/%.md, presentations/%.html,$?) docs/*.md

index: presentations
	cd presentations ;	tree -H '.' -L 1 --noreport --charset utf-8 -o index.html

clean:
	rm -rf presentations/