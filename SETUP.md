# Virtual Try-On Setup Guide

## Environment Setup with Python 3.9

This application requires Python 3.9 to run properly. Two options are provided for setting up your environment:

### Option 1: Using the Batch File (Recommended)

1. Simply double-click on `run_app.bat`
2. The script will automatically:
   - Check if Python 3.9 is installed
   - Create a virtual environment
   - Install all required dependencies
   - Launch the Streamlit application in your browser

### Option 2: Manual Setup

If you prefer to set up manually:

```bash
# Ensure Python 3.9 is installed
py -3.9 --version

# Create a virtual environment
py -3.9 -m venv venv

# Activate the environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run virtual_tryon_app.py
```

## Troubleshooting

### Python 3.9 Not Found

If Python 3.9 is not found:

1. Download and install Python 3.9 from https://www.python.org/downloads/release/python-3913/
2. Make sure to check "Add Python 3.9 to PATH" during installation
3. Run the setup again

### Model Download Issues

On first run, the application will download the SAM2 model (approximately 150MB). If you encounter network issues:

1. Ensure you have a stable internet connection
2. If the download fails, try running the application again

### API Issues

The application comes with demo API keys. If you encounter API rate limiting:

1. Get your own Roboflow API key from https://roboflow.com/
2. Get your own Segmind API key from https://www.segmind.com/
3. Enter these keys in the sidebar of the application
