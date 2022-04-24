all: Intro.md Definitions.md Proposal.md Literature.md Citations.md
	rm -f temp.md
	cat Intro.md >> temp.md
	echo "\n\n" >> temp.md
	cat Proposal.md >> temp.md
	echo "\n\n" >> temp.md
	cat Definitions.md >> temp.md
	echo "\n\n" >> temp.md
	cat Literature.md >> temp.md
	echo "\n\n" >> temp.md
	cat Update1.md >> temp.md
	echo "\n\n" >> temp.md
	cat Update2.md >> temp.md
	echo "\n\n" >> temp.md
	cat Report.md >> temp.md
	echo "\n\n" >> temp.md
	cat Citations.md >> temp.md
	echo "\n\n" >> temp.md
	pandoc temp.md -o ../propsal.pdf -s --pdf-engine=xelatex -H head.tex -V geometry:"top=1in, bottom=1in, left=1in, right=1in"
