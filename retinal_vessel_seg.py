import cv2
import numpy as np
from skimage import exposure
from skimage.morphology import black_tophat, closing, disk, dilation, erosion
from skimage.filters import threshold_otsu, frangi
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os
import argparse
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

def segment_retinal_vessels(image_path, mask_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Can't open image: {image_path}")
        raise ValueError(f"Can't open image: {image_path}")
    green_channel = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green_channel)
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img) > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        enhanced = enhanced * mask
    else:
        mask = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    bottom_hat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    if mask is not None:
        bottom_hat = bottom_hat * mask
    denoised = cv2.medianBlur(bottom_hat.astype(np.uint8), 3)
    thresh = threshold_otsu(denoised)
    binary = denoised > thresh
    if mask is not None:
        binary = binary & mask
    labeled = label(binary)
    regions = regionprops(labeled)
    min_size = 80
    result_mask = np.zeros_like(binary)
    for region in regions:
        if region.area >= min_size:
            result_mask[labeled == region.label] = True
    return result_mask

def improved_segment_retinal_vessels(image_path, mask_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Can't open image: {image_path}")
        raise ValueError(f"Can't open image: {image_path}")
    green_channel = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green_channel)
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img) > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        enhanced = enhanced * mask
    else:
        mask = None
    vesselness = frangi(enhanced, scale_range=(1, 6), beta=0.5, gamma=15)
    vesselness = (vesselness * 255).astype(np.uint8)
    thresh_val = 0.15 * 255
    binary = (vesselness > thresh_val).astype(np.uint8)
    binary = closing(binary, disk(2))
    if mask is not None:
        binary = binary & mask
    labeled = label(binary)
    regions = regionprops(labeled)
    min_size = 5
    result_mask = np.zeros_like(binary)
    for region in regions:
        if region.area >= min_size:
            result_mask[labeled == region.label] = True
    return result_mask

def ensemble_segment_retinal_vessels(image_path, mask_path=None):
    mask1 = segment_retinal_vessels(image_path, mask_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Can't open image: {image_path}")
        raise ValueError(f"Can't open image: {image_path}")
    green_channel = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green_channel)
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img) > 0
        enhanced = enhanced * mask
    else:
        mask = None
    vesselness = frangi(enhanced, scale_range=(1, 10), beta=0.3, gamma=50)
    vesselness = (vesselness * 255).astype(np.uint8)
    otsu_val = threshold_otsu(vesselness)
    thresh_val = max(0, otsu_val - 10)
    binary = (vesselness > thresh_val).astype(np.uint8)
    binary = closing(binary, disk(2))
    binary = dilation(binary, disk(1))
    binary = erosion(binary, disk(1))
    if mask is not None:
        binary = binary & mask
    labeled = label(binary)
    regions = regionprops(labeled)
    min_size = 3
    mask2 = np.zeros_like(binary)
    for region in regions:
        if region.area >= min_size:
            mask2[labeled == region.label] = True
    result_mask = mask1 | mask2
    return result_mask

def calculate_metrics(predicted, ground_truth):
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
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

def create_comparison_image(segmented_mask, ground_truth):
    seg = segmented_mask.astype(bool)
    gt = ground_truth.astype(bool)
    h, w = seg.shape
    comp_img = np.zeros((h, w, 3), dtype=np.uint8)
    comp_img[~seg & gt, 1] = 255
    comp_img[seg & ~gt, 0] = 255
    comp_img[seg & gt, 0] = 255
    comp_img[seg & gt, 1] = 255
    return comp_img

def visualize_results(original_img, segmented_mask, ground_truth=None):
    if ground_truth is not None:
        comp_img = create_comparison_image(segmented_mask, ground_truth)
        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(segmented_mask, cmap='gray')
        plt.title('Segmented Vessels')
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(144)
        plt.imshow(comp_img)
        plt.title('Comparison (G=Missed, R=FP, Y=Correct)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
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
    base_name = os.path.basename(image_path)
    number = base_name.split('_')[0]
    return os.path.join('DRIVE', 'test', '1st_manual', f'{number}_manual1.gif')

def load_ground_truth(gt_path):
    try:
        gt_img = Image.open(gt_path)
        gt_array = np.array(gt_img)
        gt_binary = (gt_array > 0).astype(np.uint8)
        return gt_binary
    except Exception as e:
        print("Couldn't load ground truth.")
        return None

def list_available_images(directory="DRIVE/test/images"):
    if not os.path.exists(directory):
        print(f"No directory: {directory}")
        return []
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(description='Retinal Vessel Segmentation')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--mask', type=str, help='Path to the mask image')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode to select image')
    parser.add_argument('--no-ground-truth', action='store_true', help='Skip ground truth comparison')
    args = parser.parse_args()
    try:
        if args.interactive:
            available_images = list_available_images()
            if not available_images:
                print("No images found.")
                return
            print("\nAvailable images:")
            for i, img_path in enumerate(available_images, 1):
                print(f"{i}. {os.path.basename(img_path)}")
            while True:
                try:
                    choice = int(input("\nPick an image number (or 0 to exit): "))
                    if choice == 0:
                        return
                    if 1 <= choice <= len(available_images):
                        image_path = available_images[choice - 1]
                        base_name = os.path.basename(image_path)
                        number = base_name.split('_')[0]
                        mask_path = os.path.join('DRIVE', 'test', 'mask', f'{number}_test_mask.gif')
                        if not os.path.exists(mask_path):
                            print(f"No mask at {mask_path}")
                            mask_path = None
                        break
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Enter a number.")
        elif args.image:
            image_path = args.image
            if args.mask:
                mask_path = args.mask
            else:
                base_name = os.path.basename(image_path)
                number = base_name.split('_')[0]
                mask_path = os.path.join('DRIVE', 'test', 'mask', f'{number}_test_mask.gif')
                if not os.path.exists(mask_path):
                    print(f"No mask at {mask_path}")
                    mask_path = None
        else:
            print("Give an image path or use --interactive.")
            return
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Can't open image: {image_path}")
            raise ValueError(f"Can't open image: {image_path}")
        segmented_mask = ensemble_segment_retinal_vessels(image_path, mask_path)
        ground_truth = None
        if not args.no_ground_truth:
            gt_path = get_ground_truth_path(image_path)
            if os.path.exists(gt_path):
                ground_truth = load_ground_truth(gt_path)
                if ground_truth is None:
                    print(f"Can't read ground truth at {gt_path}")
            else:
                print(f"No ground truth at {gt_path}")
        visualize_results(original_img, segmented_mask, ground_truth)
    except Exception as e:
        print("Something went wrong.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 