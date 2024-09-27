import cv2
import numpy as np
import supervision as sv
from typing import List

class Visualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
        )
        self.mask_annotator = sv.MaskAnnotator(
            opacity=0.5
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30
        )

    def annotate_frame(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Annotate the frame with bounding boxes, masks, and labels.
        
        :param frame: The input frame to annotate
        :param detections: List of detection results, each containing 'bbox', 'mask', 'category', and 'score'
        :return: Annotated frame
        """
        # Convert detections to supervision Detections format
        boxes = [detection['bbox'] for detection in detections]
        masks = [detection['mask'] for detection in detections]
        labels = [f"{detection['category']} {detection['score']:.2f}" for detection in detections]

        sv_detections = sv.Detections(
            xyxy=np.array(boxes),
            mask=np.array(masks),
            class_id=np.arange(len(detections))
        )

        # Annotate bounding boxes and labels
        frame = self.box_annotator.annotate(scene=frame, detections=sv_detections)
        
        # Add labels manually
        for label, box in zip(labels, boxes):
            x1, y1, _, _ = box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Annotate masks
        frame = self.mask_annotator.annotate(scene=frame, detections=sv_detections)

        return frame

    def add_trace(self, frame: np.ndarray, detections: List[dict], frame_number: int) -> np.ndarray:
        """
        Add motion traces to the frame.
        
        :param frame: The input frame to annotate
        :param detections: List of detection results
        :param frame_number: Current frame number
        :return: Frame with motion traces
        """
        if detections:
            boxes = [detection['bbox'] for detection in detections]
            sv_detections = sv.Detections(xyxy=np.array(boxes))
            if sv_detections.xyxy.any():
                frame = self.trace_annotator.annotate(scene=frame, detections=sv_detections)
            else:
                print("No detections to annotate")
        return frame

    def add_text_overlay(self, frame: np.ndarray, text: str, position: tuple = (10, 30)) -> np.ndarray:
        """
        Add a text overlay to the frame.
        
        :param frame: The input frame
        :param text: Text to overlay
        :param position: Position of the text (default: top-left corner)
        :return: Frame with text overlay
        """
        return sv.draw_text(scene=frame, text=text, position=position, color=sv.Color.red())

    def highlight_region(self, frame: np.ndarray, region: tuple) -> np.ndarray:
        """
        Highlight a specific region in the frame.
        
        :param frame: The input frame
        :param region: Tuple of (x1, y1, x2, y2) coordinates
        :return: Frame with highlighted region
        """
        return sv.draw_rectangle(scene=frame, rectangle=region, color=sv.Color.yellow())

    def create_heatmap(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Create a heatmap based on detection density.
        
        :param frame: The input frame
        :param detections: List of detection results
        :return: Heatmap overlay
        """
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            heatmap[y1:y2, x1:x2] += 1
        
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)