# Image Analysis Script Documentation (`analyze_images.py`)

The `analyze_images.py` script is a utility designed to scan a directory of plot images and extract visual statistics. These statistics describe the stylistic characteristics of the charts, such as background colors, grid presence, edge density, and color palettes.

The generated statistics (JSON) can be used as "style priors" for the synthetic image generation pipeline to produce more realistic and varied charts that mimic the input dataset.

## Requirements

This script requires **Pillow** (PIL) and **NumPy**.

```bash
pip install pillow numpy
```

## Usage

Run the script from the command line, providing the directory of images to analyze and the path for the output JSON file.

```bash
python3 analyze_images.py \
  --images /path/to/source_images \
  --output ./visual_stats.json
```

### Arguments

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--images` | Path | **Yes** | - | Directory containing the image files to analyze. |
| `--output` | Path | **Yes** | - | Path where the resulting JSON statistics will be saved. |
| `--exclude-prefix`| String | No | `None` | Skip images starting with this prefix (e.g., `synth_` to avoid analyzing previously generated images). |
| `--max-side` | Int | No | `512` | Downscale images to this maximum dimension for faster processing. |
| `--color-bins` | Int | No | `16` | Number of bins per color channel for quantization (reduces color palette complexity). |

## Output Format

The output is a JSON object containing aggregated statistics for the analyzed image set.

```json
{
  "image_count": 100,
  "sizes": {
    "640x480": 50,
    "800x600": 50
  },
  "bg_colors": {
    "255-255-255": 90,
    "240-240-240": 10
  },
  "dominant_colors": [3, 2, 4, ...],
  "fg_ratios": [0.15, 0.22, ...],
  "edge_density": [0.045, 0.082, ...],
  "grid_present": [1, 0, 1, ...],
  "log_y_scores": [0.0, 0.45, ...],
  "log_x_scores": [0.0, 0.0, ...],
  "marker_scores": [0.002, 0.05, ...],
  "cap_scores": [0.0, 0.012, ...]
}
```

### Field Descriptions

- **`image_count`**: Total number of images processed.
- **`sizes`**: Frequency distribution of image dimensions (`WidthxHeight`).
- **`bg_colors`**: Frequency of detected background colors (formatted as `R-G-B`).
- **`dominant_colors`**: List containing the count of dominant distinct colors found in the foreground of each image.
- **`fg_ratios`**: List of foreground-to-background pixel ratios for each image.
- **`edge_density`**: List of edge density scores (measure of visual complexity/clutter) for each image.
- **`grid_present`**: List of booleans (0 or 1) indicating if a grid was detected.
- **`log_[x|y]_scores`**: Heuristic scores indicating the likelihood of a logarithmic scale on the X or Y axis (based on irregular grid line spacing).
- **`marker_scores`**: Density score of small connected components, indicating presence of scatter plot markers or data points.
- **`cap_scores`**: Score indicating the presence of error bar caps (short horizontal lines at ends of vertical lines).

## How It Works

1.  **Preprocessing**: Images are downscaled to `max_side` to improve performance.
2.  **Background Detection**: Analysis of border pixels determines the background color. Diffing the image against the background creates a foreground mask.
3.  **Edge Detection**: Sobel filters compute an edge map to calculate `edge_density`.
4.  **Grid Analysis**: Projections of the edge map onto X and Y axes identify strong lines. If enough lines are found, `grid_present` is true. Spacing variance between these lines is used to calculate `log_x_score` / `log_y_score`.
5.  **Feature Detection**:
    *   **Markers**: Connected component analysis on the edge map finds small blobs typical of scatter markers.
    *   **Error Caps**: Scans for short horizontal runs in the edge map that are characteristic of error bar caps.
