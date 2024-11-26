import cv2
import numpy as np
import matplotlib.pyplot as plt
import depth_pro
import torch
import time
import pdb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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
    model, transform = depth_pro.@(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    print(f"Processing image: {image_path}")
    with torch.no_grad():
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        
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
    # FOV ratio between depth camera and RGB camera
    h_ratio = 69 / 87.0
    v_ratio = 42 / 58.0
    
    # Calculate crop region
    h_margin = int((depth_image.shape[1] - depth_image.shape[1] * h_ratio) / 2)
    v_margin = int((depth_image.shape[0] - depth_image.shape[0] * v_ratio) / 2)
    
    # Crop depth image
    cropped_depth = depth_image[v_margin:-v_margin, h_margin:-h_margin]
    
    # Resize to RGB image size
    aligned_depth = cv2.resize(cropped_depth, (rgb_shape[1], rgb_shape[0]))
    
    return aligned_depth

def load_and_compare_depths():
    # Load images
    image_name = 'color_water-1-3.png'
    rgbd_depth = np.load(f'input/depth_{image_name[6:-4]}.npy')
    rgb_img = cv2.imread(f'input/{image_name}')
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # Align depth map to RGB perspective
    rgbd_depth_aligned = align_depth_to_rgb(rgbd_depth, rgb_img.shape)
    
    depth_pro_depth = get_model_depth(f'input/{image_name}', f'output/dp_{image_name.split(".")[0]}.npy') *1000
    # import pdb; pdb.set_trace()
    
    print(f"DepthPro depth range: {depth_pro_depth.min():.2f} - {depth_pro_depth.max():.2f}")
    if check_nan(depth_pro_depth):
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
    plt.imshow(depth_pro_depth, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.title('DepthPro Depth')
    plt.axis('off')
    
    # Calculate depth differences after alignment
    valid_mask = rgbd_depth_aligned > 0
    depth_diff = np.abs(rgbd_depth_aligned - depth_pro_depth)
    depth_diff[~valid_mask] = 0
    
    # Display difference map
    plt.subplot(234)
    plt.imshow(depth_diff, cmap='hot')
    plt.colorbar(label='Depth Difference (mm)')
    plt.title('Depth Difference')
    plt.axis('off')
    
    # Statistics (considering only valid depth values)
    valid_rgbd = rgbd_depth_aligned[valid_mask]
    valid_depth_pro = depth_pro_depth[valid_mask]
    
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
    # Save figure instead of showing it
    plt.savefig(f'./depth_analysis/depth_comparison_{time.strftime("%Y%m%d_%H%M%S")}.png')
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
