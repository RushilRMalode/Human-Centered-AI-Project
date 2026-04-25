# LaTeX report

Source for the final report as a LaTeX document.

## Files

- `main.tex` — the report itself (8-section article format, ~8 pages when compiled).
- `references.bib` — bibliography (BibTeX format, `plainnat` style).

## Compiling

Requires a TeX distribution (MiKTeX, TeX Live, or Overleaf). Figures live in `../figures/` — the `\includegraphics` paths are relative (`../figures/exp{1,2,3,4}_*.png`), so compile from *this* directory, not the project root.

Two-pass compile with BibTeX:

```bash
cd latex_report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or via `latexmk` (one command):

```bash
latexmk -pdf -bibtex main.tex
```

Or drop the folder into Overleaf.

## Dependencies

Standard packages only: `amsmath`, `amsthm`, `amssymb`, `graphicx`, `hyperref`, `natbib`, `booktabs`, `microtype`, `enumitem`, `xcolor`, `geometry`. All are included in any modern TeX Live / MiKTeX install.
