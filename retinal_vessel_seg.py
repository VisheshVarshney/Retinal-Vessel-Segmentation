import cv2
import numpy as np
from skimage import exposure
from skimage.morphology import black_tophat, closing, disk
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os
import argparse
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

def segment_retinal_vessels(image_path, mask_path=None):
    # 1. Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # 2. Extract green channel
    green_channel = img[:,:,1]
    
    # 3. Apply CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green_channel)
    
    # 4. Apply mask if provided
    if mask_path and os.path.exists(mask_path):
        try:
            # Read mask using PIL for GIF support
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            mask = mask > 0  # Convert to binary
            # Erode mask with disk of radius 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            enhanced = enhanced * mask
        except Exception as e:
            print(f"Warning: Could not load mask image: {str(e)}")
            mask = None
    else:
        mask = None
    
    # 5. Bottom-hat transform with disk of radius 6
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    bottom_hat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    
    # 6. Apply mask to bottom-hat result
    if mask is not None:
        bottom_hat = bottom_hat * mask
    
    # 7. Noise removal using median filter
    denoised = cv2.medianBlur(bottom_hat.astype(np.uint8), 3)
    
    # 8. Otsu thresholding
    thresh = threshold_otsu(denoised)
    binary = denoised > thresh
    
    # 9. Apply mask to binary result
    if mask is not None:
        binary = binary & mask
    
    # 10. Remove small regions (objects with fewer than 80 pixels)
    labeled = label(binary)
    regions = regionprops(labeled)
    min_size = 80  # minimum size of vessel to keep (matching MATLAB)
    result_mask = np.zeros_like(binary)
    for region in regions:
        if region.area >= min_size:
            result_mask[labeled == region.label] = True
    
    return result_mask

def calculate_metrics(predicted, ground_truth):
    """Calculate accuracy metrics for the segmentation."""
    # Flatten the arrays for metric calculation
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat)
    recall = recall_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def visualize_results(original_img, segmented_mask, ground_truth=None):
    """Visualize the segmentation results and ground truth comparison if available."""
    if ground_truth is not None:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(segmented_mask, cmap='gray')
        plt.title('Segmented Vessels')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Calculate and display metrics
        metrics = calculate_metrics(segmented_mask, ground_truth)
        metrics_text = '\n'.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    else:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(segmented_mask, cmap='gray')
        plt.title('Segmented Vessels')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_ground_truth_path(image_path):
    """Get the corresponding ground truth image path."""
    # Convert test image path to ground truth path
    # Example: DRIVE/test/images/01_test.tif -> DRIVE/test/1st_manual/01_manual1.gif
    base_name = os.path.basename(image_path)
    number = base_name.split('_')[0]
    return os.path.join('DRIVE', 'test', '1st_manual', f'{number}_manual1.gif')

def load_ground_truth(gt_path):
    """Load ground truth image from GIF file."""
    try:
        # Use PIL to read the GIF file
        gt_img = Image.open(gt_path)
        # Convert to numpy array
        gt_array = np.array(gt_img)
        # Convert to binary (0 and 1)
        gt_binary = (gt_array > 0).astype(np.uint8)
        return gt_binary
    except Exception as e:
        print(f"Error loading ground truth image: {str(e)}")
        return None

def list_available_images(directory="DRIVE/test/images"):
    """List all available images in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return []
    
    # Get all image files (supporting common image formats)
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted(image_files)

def create_assessment_image(segmented_mask, ground_truth):
    """Create a color-coded assessment image like the MATLAB report."""
    # Ensure both masks are binary and same shape
    seg = segmented_mask.astype(bool)
    gt = ground_truth.astype(bool)
    h, w = seg.shape
    assessment_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Red: segmented only
    assessment_img[seg & ~gt, 0] = 255
    # Green: ground truth only
    assessment_img[~seg & gt, 1] = 255
    # Yellow: overlap (correct)
    assessment_img[seg & gt, 0] = 255
    assessment_img[seg & gt, 1] = 255
    return assessment_img

def main():
    parser = argparse.ArgumentParser(description='Retinal Vessel Segmentation')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--mask', type=str, help='Path to the mask image')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode to select image')
    parser.add_argument('--no-ground-truth', action='store_true', help='Skip ground truth comparison')
    args = parser.parse_args()

    try:
        if args.interactive:
            # List available images
            available_images = list_available_images()
            if not available_images:
                print("No images found in DRIVE/test/images directory!")
                return

            print("\nAvailable images:")
            for i, img_path in enumerate(available_images, 1):
                print(f"{i}. {os.path.basename(img_path)}")

            # Get user input
            while True:
                try:
                    choice = int(input("\nEnter the number of the image you want to process (or 0 to exit): "))
                    if choice == 0:
                        return
                    if 1 <= choice <= len(available_images):
                        image_path = available_images[choice - 1]
                        # Get corresponding mask path
                        base_name = os.path.basename(image_path)
                        number = base_name.split('_')[0]
                        mask_path = os.path.join('DRIVE', 'test', 'mask', f'{number}_test_mask.gif')
                        if not os.path.exists(mask_path):
                            print(f"Warning: Mask file not found at {mask_path}")
                            mask_path = None
                        break
                    else:
                        print("Invalid choice! Please try again.")
                except ValueError:
                    print("Please enter a valid number!")

        elif args.image:
            image_path = args.image
            if args.mask:
                mask_path = args.mask
            else:
                # Try to find corresponding mask
                base_name = os.path.basename(image_path)
                number = base_name.split('_')[0]
                mask_path = os.path.join('DRIVE', 'test', 'mask', f'{number}_test_mask.gif')
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found at {mask_path}")
                    mask_path = None
        else:
            print("Please either provide an image path using --image or run in interactive mode using --interactive")
            return

        # Read original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Perform vessel segmentation
        segmented_mask = segment_retinal_vessels(image_path, mask_path)
        
        # Get ground truth if available and not disabled
        ground_truth = None
        if not args.no_ground_truth:
            gt_path = get_ground_truth_path(image_path)
            if os.path.exists(gt_path):
                ground_truth = load_ground_truth(gt_path)
                if ground_truth is None:
                    print(f"Warning: Could not read ground truth image from {gt_path}")
            else:
                print(f"Warning: Ground truth image not found at {gt_path}")
        
        # Visualize results
        visualize_results(original_img, segmented_mask, ground_truth)
        
        if ground_truth is not None:
            assessment_img = create_assessment_image(segmented_mask, ground_truth)
            plt.figure(figsize=(6,6))
            plt.imshow(assessment_img)
            plt.title("Assessment Image (Red: Seg, Green: GT, Yellow: Overlap)")
            plt.axis('off')
            plt.show()
            # Optionally save:
            # plt.imsave("assessment_report.png", assessment_img)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 