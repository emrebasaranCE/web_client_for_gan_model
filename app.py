import gradio as gr
import numpy as np
import cv2
import tempfile
import os
import subprocess
from PIL import Image

def initialize(image):
    """Initialize when an image is uploaded"""
    if image is None:
        return None, None, None, "Please upload an image.", None
    
    # Get the exact dimensions of the image
    height, width = image.shape[:2]
    
    # Create a blank mask with precisely the same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Store the original image for resets
    original_image = image.copy()
    
    # Log the dimensions for verification
    print(f"Image dimensions: {width}x{height}, Mask dimensions: {mask.shape[1]}x{mask.shape[0]}")
    
    return image, mask, original_image, f"Image uploaded. Size: {width}x{height}px. Click on the image to create a mask.", None

def update_mask(image, mask, original_image, evt: gr.SelectData, brush_size):
    """Add to the mask where the user clicks"""
    if image is None or mask is None:
        return None, None, original_image, "Please upload an image first.", None
    
    # Get coordinates from the click event
    x, y = evt.index
    
    # Make a copy of the mask and update it
    updated_mask = mask.copy()
    cv2.circle(updated_mask, (int(x), int(y)), brush_size, 255, -1)
    
    # Create a visualization to show the user
    # Always start with the original clean image to avoid accumulating visual artifacts
    if original_image is not None:
        visual = original_image.copy()
    else:
        visual = image.copy()
    
    white_areas = (updated_mask > 0)
    
    # Add semi-transparent bright white overlay where mask exists
    if white_areas.any():
        # Using a brighter overlay (50% opacity) for better visibility
        visual[white_areas] = visual[white_areas] * 0.5 + np.array([255, 255, 255], dtype=np.uint8) * 0.5
    
    return visual, updated_mask, original_image, f"Mask updated at ({x}, {y})", None

def reset_mask(original_image):
    """Reset to a blank mask and restore the original image"""
    if original_image is None:
        return None, None, None, "Please upload an image first.", None
    
    # Create a fresh blank mask
    mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
    
    # Return the clean original image with no overlays
    # Also return None for the final_mask to clear it
    return original_image.copy(), mask, original_image, "Mask completely reset. All points cleared.", None

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

def process_image(white_mask, original_image):
    """Process the image with the mask using the external model"""
    if white_mask is None or original_image is None:
        return None, "Please create a mask first."
    
    try:
        # Save original image
        input_path = save_original_image(original_image)
        
        # Convert and save mask as NPY
        npy_file_path, binary_mask = convert_and_save_mask_as_npy(white_mask)
        
        # Prepare output path
        os.makedirs("tmp", exist_ok=True)
        output_file_path = os.path.join("tmp", "output.jpg")
        
        # Run the model command - correct paths based on error message
        cmd = [
            "python", "python_model/predict_copy.py",
            "python_model/model_cn", "python_model/config.json",
            input_path, npy_file_path, output_file_path, "--img_size", "256"
        ]
        
        # Print command for debugging
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute the command with more detailed error handling
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Command output: {process.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None, f"Command failed: {e.stderr}"
        
        # Check if output file was created
        if os.path.exists(output_file_path):
            # Load the output image
            processed_img = cv2.imread(output_file_path)
            if processed_img is None:
                print(f"Warning: cv2.imread returned None for {output_file_path}")
                # Try with PIL as an alternative
                try:
                    from PIL import Image
                    pil_img = Image.open(output_file_path)
                    processed_img = np.array(pil_img)
                    print(f"Successfully loaded image with PIL. Shape: {processed_img.shape}")
                except Exception as pil_error:
                    print(f"PIL loading also failed: {str(pil_error)}")
                    return None, f"Error: Could not load output image. Try checking the file manually at {output_file_path}"
            else:
                # Convert from BGR to RGB for Gradio
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                print(f"Successfully loaded processed image. Shape: {processed_img.shape}")
            
            return processed_img, f"Image processed successfully! Output saved to {output_file_path}"
        else:
            print(f"Output file {output_file_path} does not exist!")
            return None, f"Error: Output file not created at {output_file_path}"
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, f"Error processing image: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Mask Creator with Processing") as iface:
    gr.Markdown("# Mask Creator with Processing")
    gr.Markdown("Upload an image, click on it to create a mask, then process the image")
    
    # Store the mask and original image as states (not visible to user)
    mask_state = gr.State(None)
    original_image_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            # The main image display (both input and visualization)
            image_display = gr.Image(type="numpy", label="Click on the image to create mask")
            
            # Controls for mask creation
            brush_size = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Brush Size")
            
            with gr.Row():
                reset_btn = gr.Button("Reset Mask")
                create_mask_btn = gr.Button("Create White Mask")
            
            # Status message
            status_msg = gr.Textbox(label="Status")
        
        with gr.Column(scale=1):
            # Output for the white mask
            final_mask = gr.Image(type="numpy", label="White Mask")
            
            # Process button
            process_btn = gr.Button("Process Image with Mask", variant="primary")
            
            # Output for the processed image
            processed_image = gr.Image(type="numpy", label="Processed Result", interactive=False)
    
    # Set up event handlers
    image_display.upload(
        initialize,
        inputs=[image_display],
        outputs=[image_display, mask_state, original_image_state, status_msg, final_mask]
    )
    
    image_display.select(
        update_mask,
        inputs=[image_display, mask_state, original_image_state, brush_size],
        outputs=[image_display, mask_state, original_image_state, status_msg, final_mask]
    )
    
    reset_btn.click(
        reset_mask,
        inputs=[original_image_state],
        outputs=[image_display, mask_state, original_image_state, status_msg, final_mask]
    )
    
    create_mask_btn.click(
        get_white_mask,
        inputs=[image_display, mask_state, original_image_state],
        outputs=[final_mask, status_msg]
    )
    
    # Add a debug button to show the current output file
    debug_btn = gr.Button("Debug: Show Current Output File")
    
    process_btn.click(
        process_image,
        inputs=[final_mask, original_image_state],
        outputs=[processed_image, status_msg]
    )
    
    # Add debug button function to view output file directly
    debug_btn.click(
        lambda: display_file("tmp/output.jpg"),
        inputs=[],
        outputs=[processed_image, status_msg]
    )

# Function to display a file in Gradio
def display_file(file_path):
    """Load and display an image file in Gradio"""
    if os.path.exists(file_path):
        try:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, f"Successfully loaded image from {file_path}"
            else:
                return None, f"Error: Could not load image from {file_path} (cv2.imread returned None)"
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
    else:
        return None, f"Error: File does not exist at {file_path}"

# Launch the app
if __name__ == "__main__":
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)
    iface.launch()