import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import base64
import io
from PIL import Image
from inference_sdk import InferenceHTTPClient
from ultralytics import SAM
import tempfile
import os
import supervision as sv

# Set page config
st.set_page_config(page_title="Virtual Try-On Image Editor", page_icon="👕", layout="wide")

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    roboflow_api_key = st.text_input("Roboflow API Key", value="J7ODH4RBBwzdTdQNYJkQ", type="password")
    segmind_api_key = st.text_input("Segmind API Key", value="SG_99948c485eee0992", type="password")

# Functions for image processing
@st.cache_resource
def load_detection_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=roboflow_api_key
    )

@st.cache_resource
def load_segmentation_model():
    return SAM("sam2.1_b.pt")

def extract_bboxes(detection_results):
    """
    Extracts bounding boxes from detection results in [x_min, y_min, x_max, y_max] format.
    """
    predictions = detection_results["predictions"]

    bboxes = [
        [
            int(pred["x"] - pred["width"] / 2),   # x_min
            int(pred["y"] - pred["height"] / 2),  # y_min
            int(pred["x"] + pred["width"] / 2),   # x_max
            int(pred["y"] + pred["height"] / 2)   # y_max
        ]
        for pred in predictions
    ]
    return bboxes

def image_file_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def pil_image_to_base64(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def draw_bounding_boxes(image, detection_results):
    """
    Draws polished bounding boxes with semi-transparent label backgrounds.
    Returns annotated image
    """
    # Convert to RGB if needed
    if isinstance(image, np.ndarray) and image.shape[2] == 3:
        if image.dtype == np.uint8 and image[0, 0, 0] > image[0, 0, 2]:  # Simple BGR check
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictions = detection_results["predictions"]

    # xyxy format
    xyxy = np.array([
        [
            pred["x"] - pred["width"] / 2,   # x_min
            pred["y"] - pred["height"] / 2,  # y_min
            pred["x"] + pred["width"] / 2,   # x_max
            pred["y"] + pred["height"] / 2   # y_max
        ]
        for pred in predictions
    ])

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.array([pred["confidence"] for pred in predictions]),
        class_id=np.array([pred["class_id"] for pred in predictions])
    )

    labels = [
        f"{pred['class']} {pred['confidence']*100:.1f}%"
        for pred in predictions
    ]

    # Colors (BGR int tuples)
    cmap = plt.cm.get_cmap("tab20")
    colors = [
        tuple(int(c*255) for c in cmap(i % 20)[:3][::-1])
        for i in range(len(predictions))
    ]

    # Draw boxes
    box_annotator = sv.BoxAnnotator(thickness=3)
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)

    # Draw semi-transparent text boxes
    for i, (x_min, y_min, _, _) in enumerate(xyxy):
        label = labels[i]
        color = colors[i]

        # Text size
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Rectangle coords
        x1, y1 = int(x_min), int(y_min) - th - baseline - 4
        x2, y2 = int(x_min) + tw + 6, int(y_min)

        # Ensure bounds
        x1, y1 = max(x1, 0), max(y1, 0)

        # Overlay for transparency
        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Add transparency (0.4 = 40%)
        cv2.addWeighted(overlay, 0.4, annotated_image, 0.6, 0, annotated_image)

        # Put text
        cv2.putText(
            annotated_image,
            label,
            (x1 + 3, y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return annotated_image

def create_binary_mask(segment_result):
    """
    Creates a binary mask from segmentation result
    """
    masks = segment_result[0].masks.data.cpu().numpy()  # (N, H, W)
    # Return the first mask
    if len(masks) > 0:
        binary_mask = (masks[0] > 0.5).astype(np.uint8) * 255  # convert to 0/255
        return binary_mask
    return None

def inpaint_image(image_base64, mask_base64, prompt, api_key):
    """
    Send request to Segmind API for inpainting
    """
    url = "https://api.segmind.com/v1/sdxl-inpaint"
    
    # Request payload
    data = {
      "image": image_base64,
      "mask": mask_base64,
      "prompt": prompt,
      "negative_prompt": "bad quality, painting, blur",
      "samples": 1,
      "scheduler": "DDIM",
      "num_inference_steps": 25,
      "guidance_scale": 7.5,
      "seed": 12467,
      "strength": 0.9,
      "base64": False
    }
    
    headers = {'x-api-key': api_key}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        # Use io.BytesIO to treat the byte string as a file
        image_stream = io.BytesIO(response.content)
        # Open the image using PIL
        image = Image.open(image_stream)
        return image
    else:
        st.error(f"Error from API: {response.status_code} - {response.text}")
        return None

# Main app
st.title("Virtual Try-On Image Editor")
st.markdown("Upload an image, detect clothing items, and try on new styles!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Initialize clients
    detection_client = load_detection_client()
    segmentation_model = load_segmentation_model()
    
    # Step 1: Object Detection
    st.header("Step 1: Object Detection")
    if st.button("Detect Clothing Items"):
        with st.spinner("Detecting clothing items..."):
            # Run detection
            detection_results = detection_client.infer(temp_file_path, model_id="main-fashion-wmyfk/1")
            
            # Display results
            if detection_results and "predictions" in detection_results:
                st.session_state.detection_results = detection_results
                
                # Convert image to numpy for drawing
                image_np = np.array(image)
                annotated_image = draw_bounding_boxes(image_np, detection_results)
                
                # Display annotated image
                st.image(annotated_image, caption="Detected Clothing Items", use_column_width=True)
                
                # Extract bounding boxes
                bboxes = extract_bboxes(detection_results)
                st.session_state.bboxes = bboxes
                
                # Display detected items for selection
                if len(bboxes) > 0:
                    item_options = []
                    for i, pred in enumerate(detection_results["predictions"]):
                        item_options.append(f"{i}: {pred['class']} ({pred['confidence']*100:.1f}%)")
                    
                    st.session_state.item_options = item_options
                else:
                    st.warning("No clothing items detected.")
            else:
                st.error("No objects detected or there was an API error.")
    
    # Step 2: Segmentation
    if 'detection_results' in st.session_state and 'bboxes' in st.session_state:
        st.header("Step 2: Segmentation")
        
        # Show item selection
        selected_item = st.selectbox(
            "Select a clothing item to segment and edit",
            options=st.session_state.item_options if 'item_options' in st.session_state else []
        )
        
        if selected_item and st.button("Segment Selected Item"):
            with st.spinner("Generating segmentation mask..."):
                # Extract item index
                item_idx = int(selected_item.split(":")[0])
                
                # Segment using SAM
                segment_result = segmentation_model(temp_file_path, bboxes=[st.session_state.bboxes[item_idx]])
                
                # Display segmentation
                segmented_img = segment_result[0].plot()
                st.image(segmented_img, caption="Segmented Item", use_column_width=True)
                
                # Create binary mask
                binary_mask = create_binary_mask(segment_result)
                
                if binary_mask is not None:
                    # Save mask to temp file
                    mask_path = os.path.join(temp_dir, "mask.png")
                    cv2.imwrite(mask_path, binary_mask)
                    
                    # Display mask
                    st.image(binary_mask, caption="Binary Mask", use_column_width=True)
                    
                    # Save to session state
                    st.session_state.mask_path = mask_path
                    st.session_state.item_idx = item_idx
                else:
                    st.error("Failed to generate mask.")
    
    # Step 3: Inpainting
    if 'mask_path' in st.session_state:
        st.header("Step 3: Inpainting")
        
        # Get item class
        item_class = st.session_state.detection_results["predictions"][st.session_state.item_idx]["class"]
        
        # Prompt for new style
        inpaint_prompt = st.text_input(
            "Describe the new look:",
            value=f"A new {item_class} with stylish design"
        )
        
        if st.button("Generate New Look"):
            with st.spinner("Generating new look with AI..."):
                # Convert image and mask to base64
                image_base64 = image_file_to_base64(temp_file_path)
                mask_base64 = image_file_to_base64(st.session_state.mask_path)
                
                # Call inpainting API
                result_image = inpaint_image(
                    image_base64=image_base64,
                    mask_base64=mask_base64,
                    prompt=inpaint_prompt,
                    api_key=segmind_api_key
                )
                
                if result_image:
                    # Display result
                    st.image(result_image, caption="Generated Result", use_column_width=True)
                    
                    # Add download button
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Result",
                        data=buf,
                        file_name="virtual_tryon_result.png",
                        mime="image/png"
                    )
                else:
                    st.error("Failed to generate new look.")
                    
    # Clean up temp files
    if st.button("Clear All"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Remove temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        st.rerun()

# Add instructions in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ## How to use
    1. Upload an image containing clothing items
    2. Click "Detect Clothing Items" to identify items
    3. Select a clothing item from the dropdown
    4. Click "Segment Selected Item" to create a mask
    5. Enter a description for the new look
    6. Click "Generate New Look" to create the edited image
    7. Download your result
    """)
    
    st.markdown("---")
    st.markdown("""
    ## Requirements
    - Roboflow API Key (for object detection)
    - Segmind API Key (for inpainting)
    - Internet connection for API access
    """)
