TARGET=paper
SRCS = abstract.tex intro.tex related.tex dataset.tex analysis.tex behavior.tex primetime.tex peakratio.tex conclusion.tex
all: pdf

pdf:
	GS_OPTIONS=-dPDFSETTINGS=/prepress rubber -f --pdf -Wrefs -Wmisc $(TARGET)

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.brf *~ *.bak $(TARGET).pdf $(TARGET).ps $(TARGET).toc $(TARGET).lot $(TARGET).lof

cleanall: clean
	rm -f $(TARGET).pdf

spell:
	for i in $(SRCS); do ispell $$i; done

double:
	for i in $(SRCS); do double.pl < $$i; done

ps: pdf
	GS_OPTIONS=-dPDFSETTINGS=/prepress pdftops -level1 $(TARGET).pdf
