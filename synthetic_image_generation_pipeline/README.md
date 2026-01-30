# Synthetic Image Generation Pipeline

This repository generates synthetic error-bar chart images and matching label JSON files.

## Requirements

- Python 3
- Packages in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

python3 synth_pipeline.py \
    --count 3000 \
    --learn-from-labels \
    --use-existing-sizes \
    --engine auto \
    --engine-weights mpl:0.5,seaborn:0.2,pandas:0.1,plotly:0.1,bokeh:0.1
By default, outputs go to the `synthetic_data/` subfolder:
** If you have existing images for learning **
- images: `synthetic_data/images/`
- labels: `synthetic_data/labels/`

Generate 3000 images with learned stats and a manifest:

```bash
python3 synth_pipeline.py \
  --count 3000 \
  --learn-from-labels \
  --use-existing-sizes
```
