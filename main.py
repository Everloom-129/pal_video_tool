import sys
import time
import argparse
from src.video_processor import VideoProcessor

def parse_arguments():
    parser = argparse.ArgumentParser(description="PAL Video Processing Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video")
    parser.add_argument("--prompts", nargs='+', default=['cup', 'robot'], help="Detection prompts")
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = VideoProcessor()
    
    output_path = args.output if args.output else args.input.rsplit('.', 1)[0] + '_gd16.mp4'
    
    args.prompts = ['robot', 'plate', 'drawer','drawer handle', 'mug']

    start_time = time.time()
    processor.process_video(args.input, output_path, args.prompts)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    print(f"Processed video saved to {output_path}")
    

if __name__ == "__main__":
    main()