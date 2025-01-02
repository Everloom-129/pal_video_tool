import cv2
import numpy as np
import matplotlib.pyplot as plt
import depth_pro
import torch
import time
import pdb
import warnings
import os
import argparse
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

def check_nan(data: np.ndarray):
    if np.isnan(data).any():
        print("Warning: NaN values detected in numpy array")
        return True
    return False

def get_torch_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\033[95mUsing device: {device_name}\033[0m")
    return torch.device(device_name)

def get_model_depth(image_path, output_path):
    # Load model and preprocessing transform
    print("Loading model...")
    model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    print(f"Processing image: {image_path}")
    with torch.no_grad():
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        real_f_px = torch.tensor(607.5303)
        # Run inference
        prediction = model.infer(image, f_px=f_px)
        print(f"Shape of depth: {prediction['depth'].shape}")
        print(f"focallength_px: {prediction['focallength_px']}")
        depth = prediction["depth"]
        
        depth_cpu = depth.cpu().numpy() * 1000
        
        # Save depth image
        np.save(output_path, depth_cpu)
        
        # print(f"Depth statistics:")
        # print(f"Min depth: {depth_cpu.min():.2f}mm")
        # print(f"Max depth: {depth_cpu.max():.2f}mm") 
        # print(f"Mean depth: {depth_cpu.mean():.2f}mm")
    print(f"\033[95mProcessing completed. Depth saved to {output_path}\033[0m")
    return depth_cpu

def align_depth_to_rgb(depth_image, rgb_shape):
    """
    RealSense D435i Parameters:
    RGB FOV: 69.4° x 42.5° x 77° (H × V × D)
    Depth FOV: 87° x 58° x 95° (H × V × D)
    """
    h_ratio = 69.4 / 87.0  
    v_ratio = 42.5 / 58.0  
    
    # Calculate crop region
    h_margin = int((depth_image.shape[1] - depth_image.shape[1] * h_ratio) / 2)
    v_margin = int((depth_image.shape[0] - depth_image.shape[0] * v_ratio) / 2)
    
    # Crop depth image
    cropped_depth = depth_image[v_margin:-v_margin, h_margin:-h_margin]
    # Resize to RGB image size
    aligned_depth = cv2.resize(cropped_depth, (rgb_shape[1], rgb_shape[0]))
    
    return aligned_depth

def find_optimal_scale(pred_depth, gt_depth, valid_mask):
    # filter points
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    # Solve for scale factor that minimizes L1 error
    # scale * pred = gt
    # scale = median(gt / pred) is optimal for L1
    scale = np.median(gt_valid / pred_valid)
    
    print(f"Optimal scale factor: {scale:.3f}")
    return scale

def load_and_compare_depths(input_dir='data/disp/small-obj-2', output_dir='output', image_name='color_2.png'):
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/depth_analysis', exist_ok=True)
    
    # Load images
    rgbd_depth = np.load(f'{input_dir}/depth_{image_name[6:-4]}.npy')
    rgb_img = cv2.imread(f'{input_dir}/{image_name}')
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # Align depth map to RGB perspective
    rgbd_depth_aligned = align_depth_to_rgb(rgbd_depth, rgb_img.shape)
    
    depth_pro_depth = get_model_depth(f'{input_dir}/{image_name}', f'{output_dir}/dp_{image_name.split(".")[0]}.npy')
    
    # Find and apply optimal scale
    valid_mask = rgbd_depth_aligned > 0
    scale = find_optimal_scale(depth_pro_depth, rgbd_depth_aligned, valid_mask)
    depth_pro_depth_scaled = depth_pro_depth * scale
    
    if check_nan(depth_pro_depth_scaled):
        print("Warning: DepthPro depth contains NaN values")
    
    # Create a directory for individual plots
    plot_dir = f'{output_dir}/depth_analysis/{image_name.split(".")[0]}'
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M")

    # RGB image
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')
    plt.savefig(f'{plot_dir}/rgb_{timestamp}.png')
    plt.close()
    
    # Aligned RGBD depth map
    plt.figure(figsize=(8, 6))
    plt.imshow(rgbd_depth_aligned, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('Aligned RGBD Depth')
    plt.axis('off')
    plt.savefig(f'{plot_dir}/rgbd_depth_{timestamp}.png')
    plt.close()
    
    # Scaled DepthPro depth map
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_pro_depth_scaled, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('DepthPro Depth (L1 Optimized)')
    plt.axis('off')
    plt.savefig(f'{plot_dir}/depth_pro_{timestamp}.png')
    plt.close()
    
    # Calculate depth differences using scaled prediction
    depth_diff = np.abs(rgbd_depth_aligned - depth_pro_depth_scaled)
    depth_diff[~valid_mask] = 0
    
    # Display difference map
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_diff, cmap='hot')
    plt.colorbar(label='Depth Difference (mm)')
    plt.title('Depth Difference')
    plt.axis('off')
    plt.savefig(f'{plot_dir}/depth_diff_{timestamp}.png')
    plt.close()
    
    # Statistics (considering only valid depth values)
    valid_rgbd = rgbd_depth_aligned[valid_mask]
    valid_depth_pro = depth_pro_depth_scaled[valid_mask]
    
    rmse, rel_error = calculate_metrics(valid_rgbd, valid_depth_pro, valid_mask)

    print(f"- RGBD Depth range: {valid_rgbd.min():.3f} to {valid_rgbd.max():.3f}")
    print(f"- DepthPro range: {valid_depth_pro.min():.3f} to {valid_depth_pro.max():.3f}")
    print(f"- Mean absolute difference: {np.mean(depth_diff[valid_mask]):.3f}")
    print(f"- Max absolute difference: {np.max(depth_diff[valid_mask]):.3f}")
    print(f"- RMSE: {rmse:.3f}")
    print(f"- Relative Error: {rel_error:.3f}")
    
    # Depth distribution histogram
    plt.figure(figsize=(8, 6))
    plt.hist(valid_rgbd.flatten(), bins=50, alpha=0.5, label='RGBD')
    plt.hist(valid_depth_pro.flatten(), bins=50, alpha=0.5, label='DepthPro')
    plt.xlabel('Depth (mm)')
    plt.ylabel('Frequency')
    plt.title('Depth Distribution')
    plt.legend()
    plt.savefig(f'{plot_dir}/depth_dist_{timestamp}.png')
    plt.close()

    # Create combined visualization
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Depth Analysis on {input_dir}/{image_name}')
    
    # Load and display all saved plots
    for i, name in enumerate(['rgb', 'rgbd_depth', 'depth_pro', 'depth_diff', 'depth_dist']):
        img = plt.imread(f'{plot_dir}/{name}_{timestamp}.png')
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/depth_analysis/combined_graph_{timestamp}.png')
    print(f"\033[95mProcessing completed. Individual plots saved to {plot_dir}/\033[0m")
    print(f"\033[95mCombined graph saved to {output_dir}/depth_analysis/combined_graph_{timestamp}.png\033[0m")
    plt.close()

def calculate_metrics(rgbd, depth_pro, valid_mask):
    # Since rgbd and depth_pro are already masked arrays when passed in
    # we don't need to apply the mask again！
    rmse = np.sqrt(np.mean((rgbd - depth_pro)**2))
    rel_error = np.mean(np.abs(rgbd - depth_pro) / rgbd)
    
    # Add error checking
    if np.isnan(rmse) or np.isnan(rel_error):
        print("Warning: NaN values detected in metrics calculation")
        print(f"RGBD range: {rgbd.min():.3f} to {rgbd.max():.3f}")
        print(f"DepthPro range: {depth_pro.min():.3f} to {depth_pro.max():.3f}")
        
    return rmse, rel_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare depth maps from RGBD and DepthPro')
    parser.add_argument('--input_dir', type=str, required=False, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save output files')
    parser.add_argument('--image_name', type=str, required=False, help='Name of the image file to process')
    args = parser.parse_args()
    # Quick test
    # args.input_dir = 'data/disp/small-obj-3'
    # args.output_dir = 'output'
    # args.image_name = 'color_3.png'
    load_and_compare_depths(input_dir=args.input_dir, output_dir=args.output_dir, image_name=args.image_name)
