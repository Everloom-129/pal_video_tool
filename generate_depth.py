import os
import time
import numpy as np
import depth_pro
import torch
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_torch_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\033[95mUsing device: {device_name}\033[0m")
    return torch.device(device_name)

def process_image(image_path, output_dir):
    print(f"Processing image: {image_path}")
    # Set device, default is cpu so very slow
    print("Loading Depth-Pro model...")
    model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # use prediction fpx, this could be replace with actual data
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        
        depth_cpu = depth.cpu().numpy() * 1000 # convert from m to mm
        
        # Save depth image
        image_name = os.path.basename(image_path).replace('.jpg', '')
        date = time.strftime("%m-%d")
        depth_path = os.path.join(output_dir, f'depth_{date}_{image_name}.npy')
        np.save(depth_path, depth_cpu)
        
        
        # Create subplot with original image and depth visualization
        plt.figure(figsize=(16, 6), constrained_layout=True)
        
        # Create more precise grid
        gs = plt.GridSpec(1, 2, figure=plt.gcf(), width_ratios=[1, 1])
        
        # Left original image
        ax1 = plt.subplot(gs[0])
        original_img = plt.imread(image_path)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Get original image dimensions
        original_height, original_width = original_img.shape[:2]
        
        # Resize depth image to match original image dimensions
        depth_resized = cv2.resize(depth_cpu, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        # Plot depth visualization
        ax2 = plt.subplot(gs[1])
        im = plt.imshow(depth_resized, cmap='turbo')
        plt.title('Depth Map')
        plt.axis('off')
        
        # More precise control of colorbar
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="3%", pad=0.05)  # Reduce colorbar width
        plt.colorbar(im, cax=cax, label='Depth (mm)')
        
        # Ensure both subplots have the same aspect ratio
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        
        plt.suptitle(f'Depth-Pro Result for {os.path.basename(image_path)}')
        colored_depth_path = os.path.join(output_dir, f'colored_{date}_{image_name}.png')
        plt.savefig(colored_depth_path, bbox_inches='tight', pad_inches=0)
        
        # Save depth visualization as a single clean image
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_resized, cmap='turbo')
        plt.axis('off')  # 完全关闭坐标轴
        plt.gca().set_position([0, 0, 1, 1])  # 使图像填充整个画布
        depth_only_path = os.path.join(output_dir, f'depth_{date}_{image_name}.png')
        plt.savefig(depth_only_path, bbox_inches='tight', pad_inches=0)  # 移除所有边距
        plt.close()


        # Create log file in output directory
        log_path = os.path.join(output_dir, 'depth_stats.log')
        with open(log_path, 'a') as f:
            f.write(f"Depth statistics for {image_path}:\n")
            f.write(f"Min depth: {depth_cpu.min():.2f}m\n")
            f.write(f"Max depth: {depth_cpu.max():.2f}m\n")
            f.write(f"Mean depth: {depth_cpu.mean():.2f}m\n")
    print(f"\033[95mProcessing completed. Results saved to:")
    print(f"Depth array: {depth_path}")
    print(f"Colored depth: {colored_depth_path}\033[0m")
    print(f"Depth only: {depth_only_path}\033[0m")

if __name__ == "__main__":
    # input_path = "/home/tonyw/VLM/pal_video_tool/ml-depth-pro/data/disp/water-2/rgb/image.png"
    # output_dir = "output"
    # process_image(input_path, output_dir)
    # input_dir = "data/Koch_images/Koch_1210"
    input_dir = '/home/franka/R2D2_llm/processed_data/2024-12-09/12'
    output_dir = f"output/Franka_12-09/"
    i = 0
    for image_path in os.listdir(input_dir):
        if i % 30 == 0:
            process_image(os.path.join(input_dir, image_path), output_dir)
        i += 1


