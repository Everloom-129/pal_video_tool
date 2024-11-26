import cv2
import numpy as np
import matplotlib.pyplot as plt
import depth_pro
import torch
import time
import pdb
import warnings
import os

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
        
        print(f"Depth statistics:")
        print(f"Min depth: {depth_cpu.min():.2f}mm")
        print(f"Max depth: {depth_cpu.max():.2f}mm") 
        print(f"Mean depth: {depth_cpu.mean():.2f}mm")
    print(f"\033[95mProcessing completed. Depth saved to {output_path}\033[0m")
    return depth_cpu

def align_depth_to_rgb(depth_image, rgb_shape):
    """
    RealSense D435i Parameters:
    RGB FOV: 69.4° x 42.5° x 77° (H × V × D)
    Depth FOV: 87° x 58° x 95° (H × V × D)
    """
    h_ratio = 69.4 / 87.0  # 更精确的水平比例
    v_ratio = 42.5 / 58.0  # 更精确的垂直比例
    
    # Calculate crop region
    h_margin = int((depth_image.shape[1] - depth_image.shape[1] * h_ratio) / 2)
    v_margin = int((depth_image.shape[0] - depth_image.shape[0] * v_ratio) / 2)
    
    # Crop depth image
    cropped_depth = depth_image[v_margin:-v_margin, h_margin:-h_margin]
    
    # Resize to RGB image size
    aligned_depth = cv2.resize(cropped_depth, (rgb_shape[1], rgb_shape[0]))
    
    return aligned_depth

def find_optimal_scale(pred_depth, gt_depth, valid_mask):
    """
    Find optimal scale factor to minimize L1 error between prediction and ground truth
    Args:
        pred_depth: predicted depth map
        gt_depth: ground truth depth map
        valid_mask: boolean mask for valid depth values
    Returns:
        optimal scale factor
    """
    # Only consider valid depth values
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    # Solve for scale factor that minimizes L1 error
    # scale * pred = gt
    # scale = median(gt / pred) is optimal for L1
    scale = np.median(gt_valid / pred_valid)
    
    print(f"Optimal scale factor: {scale:.3f}")
    return scale

def load_and_compare_depths(input_dir='data/disp/small-obj-3', output_dir='output', image_name='color_3.png'):
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
    
    print(f"DepthPro depth range: {depth_pro_depth_scaled.min():.2f} - {depth_pro_depth_scaled.max():.2f}")
    if check_nan(depth_pro_depth_scaled):
        print("Warning: DepthPro depth contains NaN values")
    

    plt.figure(figsize=(15, 10))
    
    # RGB image
    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Aligned RGBD depth map
    plt.subplot(232)
    plt.imshow(rgbd_depth_aligned, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('Aligned RGBD Depth')
    plt.axis('off')
    
    # Scaled DepthPro depth map
    plt.subplot(233)
    plt.imshow(depth_pro_depth_scaled, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('DepthPro Depth (L1 Optimized)')
    plt.axis('off')
    
    # Calculate depth differences using scaled prediction
    depth_diff = np.abs(rgbd_depth_aligned - depth_pro_depth_scaled)
    depth_diff[~valid_mask] = 0
    
    # Display difference map
    plt.subplot(234)
    plt.imshow(depth_diff, cmap='hot')
    plt.colorbar(label='Depth Difference (mm)')
    plt.title('Depth Difference')
    plt.axis('off')
    
    # Statistics (considering only valid depth values)
    valid_rgbd = rgbd_depth_aligned[valid_mask]
    valid_depth_pro = depth_pro_depth_scaled[valid_mask]
    
    rmse, rel_error = calculate_metrics(valid_rgbd, valid_depth_pro, valid_mask)

    print(f"RGBD Depth range: {valid_rgbd.min():.3f} to {valid_rgbd.max():.3f}")
    print(f" DepthPro range: {valid_depth_pro.min():.3f} to {valid_depth_pro.max():.3f}")
    print(f"Mean absolute difference: {np.mean(depth_diff[valid_mask]):.3f}")
    print(f"Max absolute difference: {np.max(depth_diff[valid_mask]):.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Relative Error: {rel_error:.3f}")
    
    # Depth distribution histogram
    plt.subplot(235)
    plt.hist(valid_rgbd.flatten(), bins=50, alpha=0.5, label='RGBD')
    plt.hist(valid_depth_pro.flatten(), bins=50, alpha=0.5, label='DepthPro')
    plt.xlabel('Depth (mm)')
    plt.ylabel('Frequency')
    plt.title('Depth Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.title(f'Depth Analysis on {input_dir}/{image_name} ')
    # Save figure instead of showing it
    plt.savefig(f'{output_dir}/depth_analysis/graph_{time.strftime("%m%d_%H%M")}.png')
    print(f"\033[95mProcessing completed. Graph saved to {output_dir}/depth_analysis/graph_{time.strftime('%m%d_%H%M')}.png\033[0m")

    plt.close()

def calculate_metrics(rgbd, depth_pro, valid_mask):
    # Since rgbd and depth_pro are already masked arrays when passed in
    # we don't need to apply the mask again
    rmse = np.sqrt(np.mean((rgbd - depth_pro)**2))
    rel_error = np.mean(np.abs(rgbd - depth_pro) / rgbd)
    
    # Add error checking
    if np.isnan(rmse) or np.isnan(rel_error):
        print("Warning: NaN values detected in metrics calculation")
        print(f"RGBD range: {rgbd.min():.3f} to {rgbd.max():.3f}")
        print(f"DepthPro range: {depth_pro.min():.3f} to {depth_pro.max():.3f}")
        
    return rmse, rel_error

if __name__ == "__main__":
    load_and_compare_depths()
