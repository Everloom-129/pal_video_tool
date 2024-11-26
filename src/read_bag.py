# read from realsense bag file
import os
import pyrealsense2 as rs
import numpy as np
import cv2

def read_bag_file(bag_path):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Tell config that we will use a recorded device from file
    rs.config.enable_device_from_file(config, bag_path)

    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # Start streaming from file
    pipeline.start(config)

    try:
        frame_count = 0
        while frame_count < 10:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # Save depth and color images
            bag_name = os.path.basename(bag_path).split('.')[0]
            save_dir = os.path.join("data", "disp", bag_name)
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(os.path.join(save_dir, f"depth_{frame_count}.npy"), depth_image)
            cv2.imwrite(os.path.join(save_dir, f"color_{frame_count}.png"), cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            frame_count += 1
    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    bag_dir = "./data/d435i/"
    for bag_file in os.listdir(bag_dir):
        if bag_file.endswith(".bag"):
            bag_path = os.path.join(bag_dir, bag_file)
            print(f"Processing {bag_path}")
            read_bag_file(bag_path)
