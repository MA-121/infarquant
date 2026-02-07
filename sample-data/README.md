TODO: Add sample data

# Sample Data Guide

This folder contains 2 demo section slides to try InfarQuant with.


## What Is Included

Unzip file containing raw slide input examples (for the **Preprocess** tab):

- `RK111 4x CD68 405 Y GFAP 594 R NeuN 647 B merge.tif` (all channel merge)
- `RK111 4x CD68 405 Y.tif` (CD68 (infarct) channel)

Example generated outputs (for reference):

- `preprocessed_sections/` (section-level crops)
- `preprocessed_detect_red.csv` (preprocess metadata)
- `results_detect_red_exclude_green_contour.csv` (example aggregate analysis output)

## How To Use This Data

From the repo root:

1. Start the app:
   - Launch `infarquant.exe` or `python -m infarquant`
2. Open the **Preprocess** tab.
3. Unzip the sample-data folder and select it as input (NOTE: unzip may take awhile due to large file sizes).
4. Use keywords matching these sample names:
   - Reference keyword: `merge`
   - Infarct keyword: `CD68`
5. Run preprocessing.
6. Open the **Analyze** tab and select the generated `sample-data/preprocessed_sections/` folder.
7. Run analysis interactively.
