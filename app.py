## client side

import gradio as gr
import numpy as np
import cv2
import tempfile
import os
import subprocess
from PIL import Image
import time

METRICS = None

def initialize(image):
    """Initialize when an image is uploaded"""
    if image is None:
        return None, None, None, "Please upload an image.", None, None
    
    # Get the exact dimensions of the image
    height, width = image.shape[:2]
    
    # Create a blank mask with precisely the same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Store the original image for resets
    original_image = image.copy()
    
    # Log the dimensions for verification
    print(f"Image dimensions: {width}x{height}, Mask dimensions: {mask.shape[1]}x{mask.shape[0]}")
    
    return image, mask, original_image, f"Image uploaded. Size: {width}x{height}px. Click on the image to create a mask.", None, None

def update_mask(image, mask, original_image, evt: gr.SelectData, brush_size):
    if image is None or mask is None:
        return None, None, original_image, "Please upload an image first.", None, None
    
    x, y = evt.index
    updated_mask = mask.copy()
    
    half_size = brush_size // 2
    top_left = (max(0, int(x - half_size)), max(0, int(y - half_size)))
    bottom_right = (min(updated_mask.shape[1]-1, int(x + half_size)), 
                    min(updated_mask.shape[0]-1, int(y + half_size)))
    
    # Draw a filled rectangle on the mask
    cv2.rectangle(updated_mask, top_left, bottom_right, 255, -1)
    
    white_overlay = np.zeros_like(original_image)
    mask_indices = np.where(updated_mask > 0)
    white_overlay[mask_indices[0], mask_indices[1]] = [255, 255, 255]
    visual = cv2.addWeighted(original_image, 1.0, white_overlay, 0.5, 0)


    return visual, updated_mask, original_image, f"Mask updated at ({x}, {y}) with rectangular brush", None, None


def reset_mask(original_image):
    """Reset to a blank mask and restore the original image"""
    if original_image is None:
        return None, None, None, "Please upload an image first.", None, None
    
    # Create a fresh blank mask
    mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
    
    # Return the clean original image with no overlays
    # Also return None for the final_mask to clear it
    return original_image.copy(), mask, original_image, "Mask completely reset. All points cleared.", None, None

def get_white_mask(image, mask, original_image):
    """Convert the mask to a white mask for output"""
    if image is None or mask is None:
        return None, "Please create a mask first."
    
    # Get the exact dimensions of the image
    height, width = image.shape[:2]
    
    # Ensure mask has the correct dimensions
    if mask.shape[0] != height or mask.shape[1] != width:
        # Resize the mask if needed (this is a safeguard)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        print(f"Warning: Had to resize mask to match image dimensions: {width}x{height}")
    
    # Create a pure white mask (255 in all channels where mask is non-zero)
    white_mask = np.zeros_like(image)
    
    # Set all channels to 255 (pure white) where the mask is active
    for c in range(white_mask.shape[2]):  # For each channel
        white_mask[:,:,c][mask > 0] = 255
    
    # Verify final dimensions
    print(f"Final mask dimensions: {white_mask.shape[1]}x{white_mask.shape[0]}")
    
    return white_mask, f"Pure white mask created. Size: {width}x{height}px."

def convert_and_save_mask_as_npy(white_mask_np):
    """Convert the mask to binary and save as NPY file"""
    # Convert to grayscale if it's a color image
    if white_mask_np.ndim == 3 and white_mask_np.shape[2] == 3:
        white_mask_np = cv2.cvtColor(white_mask_np, cv2.COLOR_RGB2GRAY)
    
    # Binarize: Convert 255 -> 1.0, 0 -> 0.0
    binary_mask = (white_mask_np > 0).astype(np.float32)
    
    # Reshape to [N, C, H, W] = [1, 1, H, W]
    formatted_mask = binary_mask[np.newaxis, np.newaxis, :, :]  # shape [1, 1, H, W]
    
    # Save to temporary .npy file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy", mode='wb') as f:
        np.save(f, formatted_mask)
        file_path = f.name
    
    return file_path, binary_mask

def save_original_image(original_image):
    """Save the original image to a temporary file"""
    os.makedirs("tmp", exist_ok=True)
    input_path = os.path.join("tmp", "input.jpg")
    
    # Convert from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(input_path, img_bgr)
    
    return input_path


def process_image(white_mask, original_image, current_display_image):
    """Send image and mask to Flask server and return processed output"""

    global METRICS

    if white_mask is None or original_image is None:
        return None, "Please create a mask first.", None

    try:
        import requests
        from PIL import Image

        # Save the original image as JPEG
        image_path = "tmp/uploaded_image.jpg"
        Image.fromarray(original_image).save(image_path)

        # Save the white mask as JPEG
        mask_path = "tmp/uploaded_mask.jpg"
        Image.fromarray(white_mask).save(mask_path, format='PNG')

        # Send request to Flask server
        url = "http://localhost:8090/process"
        with open(image_path, 'rb') as image_file, open(mask_path, 'rb') as mask_file:
            files = {
                'image': image_file,
                'mask': mask_file
            }
            response = requests.post(url, files=files)

            if response.status_code == 200:
                from io import BytesIO
                import base64
                import json
                
                # Parse JSON response
                response_data = response.json()
                
                # Decode base64 image
                img_bytes = base64.b64decode(response_data['image'])
                processed_image = Image.open(BytesIO(img_bytes))
                processed_np = np.array(processed_image)
                
                # Get metrics
                METRICS = response_data['metrics']
                # print(METRICS)
                
                # Use the current display image (which already has the correct overlay) 
                masked_img = current_display_image if current_display_image is not None else original_image

                # Resize processed image if dimensions don't match
                if masked_img.shape != processed_np.shape:
                    processed_np = cv2.resize(processed_np, (masked_img.shape[1], masked_img.shape[0]))

                comparison = np.concatenate((masked_img, processed_np), axis=1)
                return processed_np, "Image processed successfully!", comparison
            else:
                return None, f"Error: {response.text}", None

    except Exception as e:
        return None, f"Error communicating with server: {str(e)}", None

def run_evaluation():
    time.sleep(1)  # Ensure the process has time to complete
    print("Output of the metrics:", METRICS)
    try:
        if not METRICS:
            return "No metrics available. Please process an image first."
        else:
            wanted = {"PSNR:", "SSIM:", "MAE:"}
            lines = []

            # Walk the list and whenever we see a wanted key, grab its next element
            for idx, token in enumerate(METRICS):
                t = token.strip()
                if t in wanted and idx + 1 < len(METRICS):
                    val = METRICS[idx + 1].strip()
                    lines.append(f"{t} {val}")

            # Join them with commas and line-breaks
            return ",\n".join(lines)
        
    except Exception as e:
        return f"Error running evaluation: {str(e)}"

# Function to display a file in Gradio
def display_file(file_path):
    """Load and display an image file in Gradio"""
    if os.path.exists(file_path):
        try:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, f"Successfully loaded image from {file_path}", None
            else:
                return None, f"Error: Could not load image from {file_path} (cv2.imread returned None)", None
        except Exception as e:
            return None, f"Error loading image: {str(e)}", None
    else:
        return None, f"Error: File does not exist at {file_path}", None

# Create the Gradio interface
with gr.Blocks(title="Image Inpainting with Evaluation") as iface:
    gr.Markdown("# Image Inpainting with Evaluation")
    gr.Markdown("Upload an image, click on it to create a rectangular mask, then process the image to fill the masked areas.")
    
    # Store the mask and original image as states (not visible to user)
    mask_state = gr.State(None)
    original_image_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            # The main image display (both input and visualization)
            image_display = gr.Image(type="numpy", label="Click on the image to create rectangular mask")
            
            # Controls for mask creation
            brush_size = gr.Slider(minimum=1, maximum=100, value=20, step=2, label="Rectangle Size")
            
            with gr.Row():
                reset_btn = gr.Button("Reset Mask")
                process_btn = gr.Button("Process Image with Mask", variant="primary")
                # create_mask_btn = gr.Button("Create White Mask")
            
            # Status message
        
        with gr.Column(scale=1):
            # Output for the white mask
            final_mask = gr.Image(type="numpy", label="White Mask", visible=False)
            
            # Process button
            
            # Output for the processed image
            processed_image = gr.Image(type="numpy", label="Processed Result", interactive=False)
    
    # Add a new row for side-by-side comparison
    with gr.Row():
        with gr.Column(scale=1):
            comparison_view = gr.Image(type="numpy", label="Side-by-Side Comparison (Original with Mask vs. Generated)", interactive=False)
        with gr.Column(scale=1):
            eval_results = gr.Textbox(label="Evaluation Results", lines=10)
    
             
    with gr.Row():
        status_msg = gr.Textbox(label="Status")

        
    # Set up event handlers
    image_display.upload(
        initialize,
        inputs=[image_display],
        outputs=[image_display, mask_state, original_image_state, status_msg, final_mask, comparison_view]
    )
    
    image_display.select(
        update_mask,
        inputs=[image_display, mask_state, original_image_state, brush_size],
        outputs=[image_display, mask_state, original_image_state, status_msg, final_mask, comparison_view]
    )
    
    reset_btn.click(
        reset_mask,
        inputs=[original_image_state],
        outputs=[image_display, mask_state, original_image_state, status_msg, final_mask, comparison_view]
    )
    
    ### STEP 1
    # First create the white mask
    mask_event = process_btn.click(
        get_white_mask,
        inputs=[image_display, mask_state, original_image_state],
        outputs=[final_mask, status_msg]
    )
    
    ### STEP 2
    # Then process the image with the white mask
    process_image_event = mask_event.then(
        process_image,
        inputs=[final_mask, original_image_state, image_display],  # Add image_display as third input
        outputs=[processed_image, status_msg, comparison_view]
    )

    ### STEP 3
    # After that run evaluation
    process_image_event.then(
        run_evaluation,
        outputs=[eval_results]
    )
        

# Launch the app
if __name__ == "__main__":
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)
    iface.launch()