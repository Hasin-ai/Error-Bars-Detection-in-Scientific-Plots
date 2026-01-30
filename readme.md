# Error Bar Detection & Synthetic Data Generation Pipeline

This project implements a complete pipeline for detecting error bars in scientific plots using computer vision. It consists of two main components: a **Synthetic Image Generation Pipeline** to create large-scale labeled datasets, and a **YOLO-based Detection System** (YOLO26L-pose) for training and inference.

## 📂 Project Structure

- **`synthetic_image_generation_pipeline/`**: Tools to generate synthetic error bar charts and analyze real plot styles.
  - `synth_pipeline.py`: Main script for generating synthetic images and labels.
  - `analyze_images.py`: Utility to extract style statistics (backgrounds, grids, colors) from real images.
  - `requirements.txt`: Python dependencies for the generation pipeline.
- **`error_bar_detection/`**: The deep learning training and inference modules.
  - `Training pipeline yolo.ipynb`: Production-ready training notebook using YOLO26L-pose.
  - `yolo26l-pose-error-bar-detection-diagnostic-tool.ipynb`: Diagnostic notebook for analyzing model performance.
  - `yolo26l-trained-70-epoch.pt`: Pre-trained model weights.
- **`Technical Report.pdf`**: Detailed report on the methodology and results.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for training/inference)

### 1. Install Dependencies
Install the required packages for both the synthetic pipeline and the deep learning modules.

```bash
# Core dependencies for generation
pip install -r synthetic_image_generation_pipeline/requirements.txt

# Additional dependencies for YOLO training/inference (Torch, Albumentations)
pip install torch torchvision albumentations tqdm
```

*Note: Ensure you install the version of PyTorch that matches your CUDA version.*

---

## 🛠️ Usage

### 1. Style Analysis (Optional But you must need to put your images folder before running)
If you have a dataset of real plot images, you can analyze them to extract "style priors" (e.g., color palettes, grid styles). This helps the synthetic generator create more realistic images.

```bash
cd synthetic_image_generation_pipeline

python3 analyze_images.py \
  --images /path/to/real_images \
  --output ./visual_stats.json
```

### 2. Generate Synthetic Data
Generate a large dataset of error bar plots with ground truth labels. This data can be used to train the YOLO model.

```bash
cd synthetic_image_generation_pipeline

# Generate 3000 images using default settings
python3 synth_pipeline.py \
    --count 3000 \
    --learn-from-labels \
    --use-existing-sizes \
    --engine auto \
    --engine-weights mpl:0.5,seaborn:0.2,pandas:0.1,plotly:0.1,bokeh:0.1

# Generate images using learned style priors from step 1
python3 synth_pipeline.py \
  --count 3000 \
  --visual-stats ./visual_stats.json
Output will be saved to `synthetic_data/` by default.

### 3. Training & Detection
The detection pipeline is implemented in Jupyter Notebooks.
** Gemini api based detection Can be run on just adding your api key in the notebook **

**Training:**
1. Open `error_bar_detection/Training pipeline yolo.ipynb`.
2. Update the dataset paths in the configuration section to point to your generated synthetic data.
3. Run the notebook to train the `YOLO26L-pose` model.
   - **Configuration**: The notebook is pre-configured for a 15GB GPU (e.g., Tesla T4) with optimized batch sizes and memory management.
   - **Output**: Trained weights (e.g., `best.pt`) and metrics.

**Diagnostics:**
Use the `yolo26l-pose-error-bar-detection-diagnostic-tool.ipynb` to visualize predictions and metrics on a test set.

## 📄 Technical Report
For a deep dive into the architecture, synthetic generation methodologies, and performance benchmarks, please refer to the `Technical Report.pdf` included in this repository.