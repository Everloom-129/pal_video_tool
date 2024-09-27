import argparse
import sys
from src.video_processor import VideoProcessor

def parse_arguments():
    parser = argparse.ArgumentParser(description="PAL Video Processing Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video")
    parser.add_argument("--prompts", nargs='+', default=['person', 'car'], help="Detection prompts")
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = VideoProcessor()
    
    output_path = args.output if args.output else args.input.rsplit('.', 1)[0] + '_processed.mp4'
    
    try:
        processor.process_video(args.input, output_path, args.prompts)
        print(f"Processed video saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()