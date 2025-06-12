"""
Image Quality Evaluation Script
This script calculates various image quality metrics between original and generated images.
Metrics included:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)
- IS (Inception Score)
"""

import os
import sys
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
import argparse
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_image(img_path):
    """Load an image and convert to RGB if needed"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Try using PIL first (more robust)
    try:
        img = np.array(Image.open(img_path).convert('RGB'))
        return img
    except Exception as e:
        print(f"PIL loading failed: {e}, trying OpenCV")
        
        # Fallback to OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image using OpenCV: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def calculate_psnr(original, generated):
    """Calculate Peak Signal-to-Noise Ratio"""
    # Ensure images have the same dimensions
    if original.shape != generated.shape:
        generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
    
    # Calculate PSNR (dB)
    return psnr(original, generated)

def calculate_ssim(original, generated):
    """Calculate Structural Similarity Index"""
    # Ensure images have the same dimensions
    if original.shape != generated.shape:
        generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for SSIM
    if len(original.shape) == 3:  # If RGB
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        generated_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        generated_gray = generated
    
    # Calculate SSIM
    return ssim(original_gray, generated_gray)

def preprocess_images_for_inception(images, target_size=(299, 299)):
    """Resize and preprocess images for InceptionV3"""
    processed_images = []
    for img in images:
        # Resize image to InceptionV3 input size
        resized = cv2.resize(img, target_size)
        # Preprocess for InceptionV3
        processed = preprocess_input(resized.astype('float32'))
        processed_images.append(processed)
    return np.array(processed_images)

def calculate_inception_features(images, model):
    """Extract features using InceptionV3 model"""
    # Preprocess images
    processed_images = preprocess_images_for_inception(images)
    
    # Extract features from the second-to-last layer
    features = model.predict(processed_images)
    return features

def calculate_fid(real_features, generated_features):
    """Calculate Fréchet Inception Distance"""
    # Calculate mean
    mu1, mu2 = np.mean(real_features, axis=0), np.mean(generated_features, axis=0)
    
    # Calculate covariance
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary components due to numerical error
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_inception_score(features, n_split=10, eps=1e-16):
    """Calculate Inception Score (simplified version for few images)"""
    # Get softmax predictions
    # For a proper IS calculation, we'd need more images, but this gives an approximation
    p_y = np.mean(features, axis=0)
    scores = []
    
    # Calculate KL divergence between predicted distributions
    for i in range(features.shape[0]):
        p_yx = features[i]
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        kl_d = np.sum(kl_d)
        scores.append(np.exp(kl_d))
    
    # Return mean and standard deviation
    if len(scores) > 1:
        is_score = (np.mean(scores), np.std(scores))
    else:
        is_score = (np.mean(scores), 0)
    
    return is_score

def main():
    parser = argparse.ArgumentParser(description='Calculate image quality metrics between original and generated images')
    parser.add_argument('original_path', help='Path to the original image')
    parser.add_argument('generated_path', help='Path to the generated image')
    args = parser.parse_args()
    
    try:
        # Load images
        print("Loading images...")
        original = load_image(args.original_path)
        generated = load_image(args.generated_path)
        
        # Ensure images have compatible dimensions
        if original.shape != generated.shape:
            print(f"Note: Images have different dimensions. Original: {original.shape}, Generated: {generated.shape}")
            print("Resizing generated image to match original dimensions for fair comparison.")
            generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
        
        # Calculate PSNR
        psnr_value = calculate_psnr(original, generated)
        print(f"PSNR: {psnr_value:.4f} dB")
        
        # Calculate SSIM
        ssim_value = calculate_ssim(original, generated)
        print(f"SSIM: {ssim_value:.4f}")
        
        # Calculate Inception-based metrics
        print("Loading InceptionV3 model...")
        inception_model = InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))
        
        # Extract features (need to put images in a list/batch)
        print("Calculating FID and IS...")
        images = [original, generated]
        
        # Preprocess images for Inception
        processed_images = preprocess_images_for_inception(images)
        
        # Get features for FID (using the mixed_7 layer - second to last layer before classification)
        feature_model = tf.keras.Model(inputs=inception_model.input, 
                                      outputs=inception_model.get_layer('mixed7').output)
        features = feature_model.predict(processed_images)
        
        # Calculate FID
        fid_value = calculate_fid(features[:1], features[1:])
        print(f"FID: {fid_value:.4f}")
        
        # Get predictions for IS
        predictions = inception_model.predict(processed_images)
        predictions = tf.nn.softmax(predictions).numpy()
        
        # Calculate IS (this is an approximation since we only have 1 generated image)
        is_score_mean, is_score_std = calculate_inception_score(predictions[1:])
        print(f"Inception Score: {is_score_mean:.4f} ± {is_score_std:.4f}")
        
        # Additional information
        print("\nNote: For more accurate FID and IS scores, larger datasets of images are typically used.")
        print("These metrics are approximations based on the single image pair provided.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())