import cv2
import numpy as np
from .api_client import APIClient
from .visualization import Visualizer

class VideoProcessor:
    def __init__(self):
        self.api_client = APIClient()
        self.visualizer = Visualizer()

    def process_video(self, input_path, output_path, prompts):
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % fps == 0:  # Process every second
                cv2.imwrite('temp_frame.jpg', frame)
                result = self.api_client.detect_objects('temp_frame.jpg', prompts)
                detections = self.convert_result_to_detections(result)
                
                frame = self.visualizer.annotate_frame(frame, detections)
                frame = self.visualizer.add_trace(frame, detections, frame_count)
                frame = self.visualizer.add_text_overlay(frame, f"Frame: {frame_count}")
                frame = self.visualizer.create_heatmap(frame, detections)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def convert_result_to_detections(self, result):
        detections = []
        for obj in result.objects:
            mask = self.api_client.rle2rgba(obj.mask)
            mask_np = np.array(mask)
            detections.append({
                'bbox': obj.bbox,
                'mask': mask_np,
                'category': obj.category,
                'score': obj.score
            })
        return detections