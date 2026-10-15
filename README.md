# papers

Code, data cards and reproducibility materials for the papers of
[Ryan Lin](https://2004lryan.github.io) and co-authors.
Each paper has its own top-level folder, named with its three-digit project
number followed by a short name.

| Folder | Paper | Status |
|---|---|---|

## Folder contents

Each folder is self-contained. Its `README.md` explains what the code
reproduces, which data it needs and how to run it. The `LICENSE` and citation
details inside a folder apply to that folder.

## Getting the code

Whole repository:

```bash
git clone https://github.com/2004lryan/papers.git
cd papers/<folder>
```

A single folder only:

```bash
git clone --filter=blob:none --sparse https://github.com/2004lryan/papers.git
cd papers
git sparse-checkout set <folder>
cd <folder>
```

## Citing

Please cite the paper. The reference is given in the folder's README or
`CITATION.cff`.
