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

**Note:** The requirements.txt now includes proper version pinning for all dependencies including:
- tensorflow>=2.13,<2.15
- streamlit>=1.30,<1.35
- opencv-python>=4.8,<4.10
- ultralytics>=8.0,<9
- And other dependencies with version constraints

### 4. Model Setup

The freshness classifier model will be automatically downloaded from Hugging Face on first run:
- Repository: `pqhunter15/freshnessclassv1`
- File: `fresh_rotten_resnet_tuned_conv4_conv5.keras`

If you prefer to download it manually, place it in the `src/freshness_inference/` directory.

## Running the Application

### Run the Main Streamlit App

```bash
streamlit run streamlit_app.py
```

This will start the application at `http://localhost:8501`

### Run with Debug Logging

```bash
streamlit run streamlit_app.py --logger.level=debug
```

**Note:** The application has been refactored into a modular structure with:
- Enhanced error handling and logging
- Automatic model downloading
- Improved UI components
- Better code organization

The previous `app1.py` and `app2.py` files have been removed as they were duplicates.

## Application Features

The Streamlit app includes:

- **Model Selection:**
  - **Fruit & Veg** group:
    - Light: Combined dataset YOLO11m (50 epochs, 768px)
    - Heavy: Combined dataset YOLO11x (120 epochs, 896px)
  - **Banana Only** group:
    - Light: Banana-only YOLO11x (150 epochs, 896px)
    - Heavy: Banana-only YOLO11n (100 epochs, 640px)

- **Image Input:**
  - Upload an image file or capture using camera

- **Detection Models:**
  - `best3.pt`: Fruit & Veg Light (YOLO11m, 50 epochs, 768px)
  - `best4-all-heavy.pt`: Fruit & Veg Heavy (YOLO11x, 120 epochs, 896px)
  - `best.pt`: Banana Only Light (YOLO11x, 150 epochs, 896px)
  - `banana_yolo11n_best.pt`: Banana Only Heavy (YOLO11n, 100 epochs, 640px)

- **Output:**
  - Original image display
  - Final prediction with color-coded bounding boxes (green for fresh, red for rotten)
  - Collapsible "More details" section with intermediate results
  - Confidence scores and prediction probabilities

- **Enhanced Features:**
  - Comprehensive logging for debugging
  - Robust error handling
  - Automatic model downloading
  - Modular code structure for maintainability

## Code Structure

The application has been refactored for better maintainability:

```
src/
├── freshness_inference/
│   ├── constants.py          # Configuration constants
│   ├── logging_config.py     # Logging setup
│   ├── model.py             # Freshness classifier
│   ├── preprocess.py        # Image preprocessing
│   └── config.py            # Model configuration
└── streamlit_ui/
    ├── __init__.py
    └── components.py        # Reusable UI components

streamlit_app.py             # Main application (refactored)
scripts/
└── predict_folder.py        # Batch processing script
```

Key improvements:
- Centralized configuration in `constants.py`
- Structured logging with `logging_config.py`
- Modular UI components
- Enhanced error handling throughout
- Type hints and documentation

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
streamlit run streamlit_app.py --server.port 8502
```

### Module Import Errors

Ensure you're in the virtual environment:
```bash
source venv/bin/activate
```

### Model Download Issues

If the automatic model download fails:
1. Check your internet connection
2. Ensure you have the `huggingface_hub` package installed
3. Manually download from: https://huggingface.co/pqhunter15/freshnessclassv1
4. Place the file in `src/freshness_inference/`

### Logging and Debugging

The application now includes comprehensive logging. Check the `freshness_inference.log` file for detailed information about any issues.

### Numpy/TensorFlow Compatibility Issues

The requirements.txt includes proper version constraints. If you still encounter issues:
```bash
pip install --upgrade "numpy>=1.24,<2" "tensorflow>=2.13,<2.15"
```

### Performance Issues

- The app uses `@st.cache_resource` for model caching
- For large images, the app automatically resizes to 1280x720 for processing
- Detection models are cached after first load

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
