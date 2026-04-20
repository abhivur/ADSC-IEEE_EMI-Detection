# Phase 8 — Interactive Streamlit Demo

A single-page web app that walks a viewer through the complete EMI detection
pipeline, from raw oscilloscope CSV to device identification.

## Pages

- **Overview** — problem framing, approach, headline metrics.
- **Live Demo** — pick a dataset file (or upload your own CSV) and watch every
  processing stage render in real time: raw signal, conditioned signal, FFT,
  feature fingerprint vs class medians, hierarchical prediction graph, and
  PCA location within the training distribution.
- **Dataset Explorer** — filterable table of all 783 captures, with
  device/state histograms and the sample-rate distribution that motivates
  the rate-agnostic classifier.
- **Results & Methodology** — benchmark table, trustworthiness notes, and
  all five Phase 7 result figures.

## Run

From the project root:

```bash
pip install -r Phase_8_Demo/requirements.txt
streamlit run Phase_8_Demo/app.py
```

The app expects:

- `Phase_1_Dataset_Audit/dataset_index.csv`
- `Phase_4_Feature_Engineering/features.csv`
- `Phase_6_Pipeline/models/*.joblib`
- `Phase_7_Results_Analysis/fig_p7_*.png`

All of these are produced by the earlier phases and are already in the repo.
