# Running the Produce Freshness Classifier

This guide provides step-by-step instructions to set up and run the Streamlit application for fruit and vegetable freshness detection.

## Prerequisites

- Python 3.11 or higher (preferably using pyenv)
- pip or conda package manager
- Git (for cloning the repository)

## Setup Instructions

### 1. Navigate to the Project Directory

```bash
cd /Users/bala/py/FreshnessClassifier/ProduceFreshnessClassifier
```

### 2. Create a Virtual Environment (Optional but Recommended)

Using **pyenv** (if already set up):
```bash
pyenv local 3.11.0
python -m venv venv
source venv/bin/activate
```

Or using **Python venv directly**:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter numpy compatibility issues with PyTorch, ensure numpy is version <2:
```bash
pip install "numpy<2"
```

### 4. Verify Model File

Ensure the freshness classifier model exists at:
```
src/freshness_inference/fresh_rotten_resnet_tuned_conv4_conv5.keras
```

If the file is missing, download it from the Hugging Face repository:
- Repository: `pqhunter15/freshnessclassv1`
- File: `fresh_rotten_resnet_tuned_conv4_conv5.keras`

Place it in the `src/freshness_inference/` directory.

## Running the Application

### Run the Main Streamlit App

```bash
streamlit run
```

This will start the application at `http://localhost:8501`

### Run the Main Streamlit App with Custom Settings

```bash
streamlit run --logger.level=debug
```

### Alternative 1: Run app1.py (Light Model Variant)

```bash
streamlit run app1.py
```

### Alternative 2: Run app2.py (Heavy Model Variant)

```bash
streamlit run app2.py
```

## Application Features

The Streamlit app includes:

- **Model Selection:**
  - Banana Only (Light/Heavy versions)
  - Fruit & Veg (Light/Heavy versions)

- **Image Input:**
  - Upload an image or capture using camera

- **Detection Models:**
  - `best.pt`: Banana-only YOLO11x (150 epochs, 896px)
  - `banana_yolo11n_best.pt`: Banana-only YOLO11n (100 epochs, 640px)
  - `best3.pt`: Combined YOLO11m (50 epochs, 768px)
  - `best4-all-heavy.pt`: Combined heavy YOLO11x (120 epochs, 896px)

- **Output:**
  - Original image display
  - Final prediction with bounding boxes
  - Collapsible "More details" section with intermediate results

## Batch Processing (Predict Folder)

To run predictions on a folder of images:

```bash
python3.12 scripts/predict_folder.py --folder-path sample_images/
```

Output is saved to `predictions.csv`

### Additional Options:

```bash
python3.12 scripts/predict_folder.py --folder-path <path_to_folder> --output-csv <output_file.csv> --threshold 0.5
```

- `--folder-path`: Path to folder containing images (default: `sample_images/`)
- `--output-csv`: Output CSV file name (default: `predictions.csv`)
- `--threshold`: Classification threshold for rotten class (default: 0.5)

## Troubleshooting

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run app1.py --server.port 8502
```

### Module Import Errors

Ensure you're in the virtual environment:
```bash
source venv/bin/activate
```

### Numpy/TensorFlow Compatibility Issues

Downgrade numpy to <2:
```bash
pip install --upgrade "numpy<2"
```

Then restart the kernel:
```bash
streamlit cache clear
```

### YOLO Model Not Found

Verify the weight files exist:
```bash
ls -la weights/
```

Required files:
- `weights/best.pt`
- `weights/banana_yolo11n_best.pt`
- `weights/best3.pt`
- `weights/best4-all-heavy.pt`

### Freshness Classifier Model Not Found

Download from Hugging Face Hub or place the `.keras` file in:
```
src/freshness_inference/fresh_rotten_resnet_tuned_conv4_conv5.keras
```

## Project Structure

```
.
├── app1.py                          # Main Streamlit application
├── app2.py                          # Alternative Streamlit app
├── requirements.txt                 # Python dependencies
├── run.md                          # This file
├── scripts/
│   └── predict_folder.py           # Batch prediction script
├── src/
│   └── freshness_inference/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── predict.py
│       ├── preprocess.py
│       └── fresh_rotten_resnet_tuned_conv4_conv5.keras
├── weights/
│   ├── best.pt
│   ├── banana_yolo11n_best.pt
│   ├── best3.pt
│   └── best4-all-heavy.pt
└── sample_images/
    └── (sample fruit images for testing)
```

## Performance Notes

- The first run loads models (may take 30-60 seconds)
- Subsequent runs use cached models (faster)
- GPU acceleration recommended for faster inference (if CUDA available)

## Additional Resources

- **Streamlit Documentation:** https://docs.streamlit.io/
- **YOLO Documentation:** https://docs.ultralytics.com/
- **TensorFlow Documentation:** https://www.tensorflow.org/
