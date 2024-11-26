import os
import numpy as np
import depth_pro
import torch

def get_torch_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\033[95mUsing device: {device_name}\033[0m")
    return torch.device(device_name)

def process_image(image_path, output_path):
    # Load model and preprocessing transform
    print("Loading model...")
    model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Processing image: {image_path}")
    with torch.no_grad():
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        
        # Run inference
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        
        # Convert to numpy array
        depth_cpu = depth.cpu().numpy()
        
        # Save depth image
        np.save(output_path, depth_cpu)
        
        print(f"Depth statistics:")
        print(f"Min depth: {depth_cpu.min():.2f}m")
        print(f"Max depth: {depth_cpu.max():.2f}m") 
        print(f"Mean depth: {depth_cpu.mean():.2f}m")
    
    print(f"\033[95mProcessing completed. Depth saved to {output_path}\033[0m")

if __name__ == "__main__":
    input_path = "/home/tonyw/VLM/pal_video_tool/ml-depth-pro/data/disp/water-2/rgb/image.png"
    output_path = "output/depth.npy"
    process_image(input_path, output_path)
