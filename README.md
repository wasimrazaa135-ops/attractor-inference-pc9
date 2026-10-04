# Library effects limit attractor-style inference from single-cell data on drug-tolerant lung cancer cells

Code, helper module and numerical outputs for the manuscript by Raza, Mehmood, Enderlein and Da Silva.

## What is here

| Path | Contents |
|---|---|
| `notebooks/GSE149383_GSE150949_Complete_Analysis.ipynb` | Full analysis (erlotinib dataset GSE149383, osimertinib dataset GSE150949, and the added control analyses) |
| `src/reviewer_controls.py` | Helper functions for the control analyses (Section E). The notebook also re-creates this file in cell E0. |
| `results/gse149383_sectionE/` | Output tables and a figure from the control analyses (library, depth, thinning, doublets, equal-n correlation dimension, CCM surrogates, random-gene control) |
| `results/gse150949/` | Output tables, figures and JSON summaries for the osimertinib dataset |

## Data

Raw data are **not** stored here. They are public and downloaded by the notebook from NCBI GEO:

- GSE149383 (erlotinib time course, Drop-seq; Aissa et al., Nat Commun 2021)
- GSE150949 (osimertinib time course, 10x; Oren et al., Nature 2021)

## How to run

1. Open the notebook in Google Colab (the notebook was written for Colab and uses `/content` paths).
2. Run the cells from top to bottom. The first setup cell installs packages and may ask for one runtime restart.
3. The GSE150949 section needs a high-memory runtime.

Packages are listed in `requirements.txt`. To record the exact versions you used, run `pip freeze` in the same Colab session and paste the relevant lines here.

## Notes on reproducibility

- Random seeds are set in the notebook; results from surrogate and permutation tests vary slightly between runs (see the manuscript).
- Output tables in `results/` are those used in the manuscript.

## Citation

See `CITATION.cff`. After archiving, cite the Zenodo DOI: **DOI to be added**.

## License

Code: MIT (see `LICENSE`).
