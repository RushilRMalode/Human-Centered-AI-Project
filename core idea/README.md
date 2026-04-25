# Core idea

A focused, self-contained LaTeX document covering the project's core idea: the model, every piece of notation, and the experimental design with specifications.

Think of this as the "theory part only" explainer --- shorter than the full report, denser on definitions, lighter on motivation and experimental results. Companion to `../latex_report/main.tex`, not a replacement.

## Files

- `main.tex` --- the document (6 pages compiled).
- `main.pdf` --- compiled output (committed here for convenience).

No separate bibliography --- the document is self-contained.

## What's in it

1. **Core idea in one paragraph** --- the central claim in the simplest form.
2. **The model** --- observations, arrival-time dynamics (Eq.~1), complete notation table, mark model (mixed-type product), how the latent $z_k$ is assigned at generation time, and the full list of assumptions.
3. **The theorem in two lines** --- informal statement + failure-mode corollary.
4. **Inference** --- the branching-EM algorithm with closed-form E- and M-steps.
5. **Experiments** --- default ground-truth parameters, common protocol, and a specification block for each of the four experiments (question, setup, axes, prediction, script).
6. **Compute and reproducibility** --- wall-clock, seeds, output artefacts.
7. **Summary**.

## Compiling

Requires a LaTeX install (MiKTeX, TeX Live, or Overleaf). Single-pass compile --- no bibliography:

```bash
cd "core idea"
pdflatex main.tex
```

## Dependencies

Standard packages only: `amsmath`, `amssymb`, `amsthm`, `booktabs`, `hyperref`, `enumitem`, `xcolor`, `tikz`, `geometry`.
