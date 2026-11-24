# Paper — Urban Accident Risk Prediction System

LaTeX source of the project paper.

| File | Content |
|------|---------|
| `main.tex` | Document setup, packages, title and team, shared TikZ styles |
| `sections/gap_analysis.tex` | Literature summary and research gaps G1–G6 |
| `sections/proposed_solution.tex` | Proposed solution (all nine required subsections) |
| `figures/*.tex` | TikZ diagrams (architecture, preprocessing, model framework, training workflow) |
| `references.bib` | Bibliography (BibTeX) |

## Compile locally

```bash
cd paper
latexmk -pdf main.tex
```

## Use with Overleaf

**Option 1 — upload:** use [`Urban_Accident_Risk_Overleaf.zip`](Urban_Accident_Risk_Overleaf.zip) (or compress
the contents of this folder yourself), then in Overleaf choose *New Project → Upload Project*. Overleaf compiles it with pdfLaTeX and BibTeX
automatically.

**Option 2 — GitHub sync (Overleaf premium):** *Menu → GitHub* links an Overleaf project to a
repository. Because this paper lives in a subfolder, upload (option 1) is the simpler route.

When text is edited in Overleaf, copy the changed `.tex` files back into this folder on the
`paper/proposed-solution` branch and open a pull request, so that the repository stays the
single source of truth.

## Writing rules

* Every figure and table is referenced in the text (`\cref{...}`).
* Every equation is numbered, and every symbol is defined below it.
* Expected improvements are written as expectations until experiments confirm them.
