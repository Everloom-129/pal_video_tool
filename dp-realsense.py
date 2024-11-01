import os
import time
import numpy as np
from PIL import Image
import depth_pro
import torch

def get_torch_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\033[95mUsing device: {device_name}\033[0m")
    return torch.device(device_name)

def process_folder(input_folder, output_folder):
    # Load model and preprocessing transform
    print("Loading model...")
    model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Statistics initialization
    total_time = 0
    depths_stats = []
    image_count = 0
    
    # Process each image
    print(f"Processing images from {input_folder}")
    with torch.no_grad():
        for img_name in os.listdir(input_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
                img_path = os.path.join(input_folder, img_name)
                
                # Time the inference
                start_time = time.time()
                
                # Load and preprocess image
                image, _, f_px = depth_pro.load_rgb(img_path)
                image = transform(image)
                
                # Run inference
                prediction = model.infer(image, f_px=f_px)
                depth = prediction["depth"]
                
                # 将depth转移到CPU并转换为numpy
                depth_cpu = depth.cpu().numpy()
                
                # Save depth image
                depth_filename = os.path.splitext(img_name)[0] + '_depth.npy'
                np.save(os.path.join(output_folder, depth_filename), depth_cpu)
                
                # Calculate processing time
                process_time = time.time() - start_time
                total_time += process_time
                
                # Collect statistics (使用CPU上的depth)
                depths_stats.append({
                    'min_depth': float(depth_cpu.min()),
                    'max_depth': float(depth_cpu.max()),
                    'mean_depth': float(depth_cpu.mean()),
                    'process_time': process_time
                })
                
                print(f"Processed {img_name}: {process_time:.2f}s")
    
    # Write statistics to log file
    with open(os.path.join(output_folder, 'log.md'), 'w') as f:
        f.write(f"Processing Summary\n")
        f.write(f"=================\n")
        f.write(f"Total images processed: {image_count}\n")
        f.write(f"Total processing time: {total_time:.2f}s\n")
        f.write(f"Average processing time per image: {total_time/image_count:.2f}s\n\n")
        
        f.write("Depth Statistics:\n")
        min_depths = [s['min_depth'] for s in depths_stats]
        max_depths = [s['max_depth'] for s in depths_stats]
        mean_depths = [s['mean_depth'] for s in depths_stats]
        
        f.write(f"Overall min depth: {min(min_depths):.2f}m\n")
        f.write(f"Overall max depth: {max(max_depths):.2f}m\n")
        f.write(f"Overall mean depth: {np.mean(mean_depths):.2f}m\n")
    print(f"\033[95mProcessing {output_folder} completed.\033[0m")

if __name__ == "__main__":
    input_folder = "/home/tonyw/VLM/pal_video_tool/ml-depth-pro/data/disp/water-2/rgb/"  
    output_folder = "output/rgb/water-2"      
    process_folder(input_folder, output_folder)
