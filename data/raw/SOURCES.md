# Raw data sources

| File | Source | Licence |
|---|---|---|
| mfq2_items.csv | MFQ-2 (Atari et al., 2023), items + scoring key from the preprint appendix and analysis code, https://osf.io/srtxn/ . Item 27 uses the preprint/code wording; the OSF English .docx has "In a fair society, I want people who work harder than others to end up richer than others." | CC-BY 4.0 |
| mfq2_study2_raw.csv | Atari et al. Study 2, 19 countries, item-level, https://osf.io/download/9dwzt/ . `porient_1` 1-10 left-right. | CC-BY 4.0 |
| mfq2_study3.csv | Atari et al. Study 3 (US/India/Canada), foundation scores only (`*_tot`, 1-5), https://osf.io/download/qaxz6/ | CC-BY 4.0 |
| clifford2015_pmc.html | Clifford et al. (2015) Moral Foundations Vignettes, Table 1, https://pmc.ncbi.nlm.nih.gov/articles/PMC4780680/ ; parsed by `python -m mft.vignettes` | author manuscript |
| ../seeds/social-chem-101/ | Social Chemistry 101 (Forbes et al., 2020), real situations from Reddit etc. with rule-of-thumb moral-foundation labels (MFT-1). Used only as generation seeds, never as test data. https://github.com/mbforbes/social-chemistry-101 | **CC BY-SA 4.0** (derived data shared must keep this licence) |
| ../seeds/mfrc.csv | Moral Foundations Reddit Corpus (Trager et al., 2022), per-annotator MFQ-2 labels. Used only as generation seeds. https://huggingface.co/datasets/USC-MOLA-Lab/MFRC | CC-BY 4.0 |
