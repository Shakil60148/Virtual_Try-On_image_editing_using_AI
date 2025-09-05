# Virtual Try-On Image Editor

A computer vision application that allows users to upload images, detect clothing items, segment them, and try on new designs using AI.

## Project Structure

- **Virtual_TryOn_Development.ipynb**: Jupyter notebook containing the development process and examples
- **virtual_tryon_app.py**: Streamlit application for the virtual try-on functionality
- **run_app.bat**: Batch file to set up and run the application using Python 3.9
- **requirements.txt**: List of required Python packages
- **SETUP.md**: Detailed setup instructions and troubleshooting
- **sam2.1_b.pt**: Pre-downloaded SAM2 model file

## Features

1. **Object Detection**: Detects clothing items in uploaded images using Roboflow's YOLO model
2. **Segmentation**: Creates precise masks of clothing items using SAM2 (Segment Anything Model 2)
3. **Inpainting**: Replaces clothing items with new designs using Segmind's API

## How to Run

### Option 1: Using the Batch File (Recommended)

1. Double-click on `run_app.bat`
2. The script will automatically:
   - Set up a Python 3.9 virtual environment
   - Install all required dependencies
   - Launch the Streamlit application in your browser

### Option 2: Manual Setup

1. Make sure Python 3.9 is installed
2. Create a virtual environment: `py -3.9 -m venv venv`
3. Activate the environment: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run virtual_tryon_app.py`

## Using the Application

1. Upload an image containing clothing items
2. Click "Detect Clothing Items" to identify all clothing in the image
3. Select a specific clothing item from the dropdown list
4. Click "Segment Selected Item" to create a precise mask
5. Enter a description for the new look (e.g., "A red polka dot shirt")
6. Click "Generate New Look" to create the edited image
7. Download the final result

## API Keys

The application uses:
- Roboflow API for object detection
- Segmind API for inpainting

Demo API keys are included, but for production use, please obtain your own.
