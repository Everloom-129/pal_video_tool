import os
import time
import numpy as np
import depth_pro
import torch
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)   

def get_torch_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\033[95mUsing device: {device_name}\033[0m")
    return torch.device(device_name)

def save_visualization(depth_resized, original_img, output_dir, image_name, date):
    """Save visualization of depth map results"""
    # Create and save comparison plot (original vs depth map)
    plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = plt.GridSpec(1, 2, figure=plt.gcf(), width_ratios=[1, 1])
    
    # Original image on left
    ax1 = plt.subplot(gs[0])
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Depth map on right
    ax2 = plt.subplot(gs[1])
    im = plt.imshow(depth_resized, cmap='turbo')
    plt.title('Depth Map')
    plt.axis('off')
    
    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Depth (mm)')
    
    # Set same aspect ratio
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Depth-Pro Result for {image_name}')
    colored_depth_path = os.path.join(output_dir, f'colored_{date}_{image_name}.png')
    plt.savefig(colored_depth_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save depth map only
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_resized, cmap='turbo')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    depth_only_path = os.path.join(output_dir, f'depth_{date}_{image_name}.png')
    plt.savefig(depth_only_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return colored_depth_path, depth_only_path

def save_depth_stats(depth_data, image_path, output_dir):
    """Save depth data statistics"""
    log_path = os.path.join(output_dir, 'depth_stats.log')
    with open(log_path, 'a') as f:
        f.write(f"Depth statistics for {image_path}:\n")
        f.write(f"Min depth: {depth_data.min():.2f}m\n")
        f.write(f"Max depth: {depth_data.max():.2f}m\n")
        f.write(f"Mean depth: {depth_data.mean():.2f}m\n")

def process_single_image(image_path, model, transform, output_dir):
    """Process a single image"""
    with torch.no_grad():
        # Predict depth
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        depth_cpu = depth.cpu().numpy() * 1000  # convert to mm
        
        # Save depth data
        image_name = os.path.basename(image_path).replace('.jpg', '')
        date = time.strftime("%m-%d")
        depth_path = os.path.join(output_dir, f'depth_{date}_{image_name}.npy')
        np.save(depth_path, depth_cpu)
        
        # Resize depth map
        original_img = plt.imread(image_path)
        original_height, original_width = original_img.shape[:2]
        depth_resized = cv2.resize(depth_cpu, (original_width, original_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Save visualization results
        colored_path, depth_path = save_visualization(
            depth_resized, original_img, output_dir, image_name, date)
        
        # Save statistics
        save_depth_stats(depth_cpu, image_path, output_dir)
        
        return depth_path, colored_path

def process_image_dir(input_dir, output_dir):
    """Process entire image directory"""
    print(f"Processing image: {input_dir}")
    print("Loading Depth-Pro model...")
    
    # Create model
    model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    for image_file in tqdm(os.listdir(input_dir), desc="Generating depth maps"):
        image_path = os.path.join(input_dir, image_file)
        depth_path, colored_path = process_single_image(
            image_path, model, transform, output_dir)
        
        print(f"\033[95mProcessing completed. Results saved to:")
        print(f"Depth array: {depth_path}")
        print(f"Colored depth: {colored_path}\033[0m")
    
    plt.close('all')

if __name__ == "__main__":
    # input_path = "/home/tonyw/VLM/pal_video_tool/ml-depth-pro/data/disp/water-2/rgb/image.png"
    # output_dir = "output"
    # process_image(input_path, output_dir)
    import sys
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
        process_image_dir(input_dir, output_dir)
    else:
        print("Usage: python generate_depth.py <input_dir> <output_dir>")
    # input_dir = "/home/franka/R2D2_llm/processed_data/2024-12-09/12"
    # output_dir = f"output/Franka/1209-12/"
    # process_image_dir(input_dir, output_dir)


