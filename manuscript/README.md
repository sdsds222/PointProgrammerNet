# PointProgrammerNet manuscript draft

This draft targets **Pattern Recognition Letters (Elsevier)** and follows the journal's two-column `elsarticle` layout with the included `prletter.sty` journal style. The English main paper is seven pages. It contains the state layout, streaming workflow, both task readouts, core proofs, principal experiments, and a worked distributed-inference example. Exact normalization details, complete proofs, state-operation definitions, and controlled ablations remain in `supplement.tex`.

Author metadata:

- Dian Yu, corresponding author
- University of New South Wales, Sydney, NSW 2052, Australia
- dian.yu2@student.unsw.edu.au
- ORCID: 0009-0002-0107-1014

The manuscript uses the frozen no-global-max models and completed three-seed experiments. It separates proved properties, measured results, and the proposed future delta-rule extension.

Compile from this directory:

```powershell
tectonic --keep-logs --keep-intermediates main.tex
tectonic --keep-logs --keep-intermediates supplement.tex
```

The manuscript remains within the journal's seven-page initial-submission limit, including figures, tables, and references. Code instructions are in `../CODE_REPRODUCTION.md`. Research highlights and the journal's authorship-confirmation form remain separate submission files.

Chinese proofreading sources are in `../review/main_chinese.tex` and `../review/supplement_chinese.tex`; they follow the English main-paper and supplementary-material division without a page limit.

