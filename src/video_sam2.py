# class to process video with sam2

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
import cv2

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


class VideoSAM2:
    def __init__(self, model_path="facebook/sam2-hiera-large"):
        self.model = SAM2VideoPredictor.from_pretrained(model_path)
        self.inference_state = None
        self.video_segments = {}

    def load_video(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def add_points(self, frame_idx, points, labels):
        
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        
        self.inference_state = self.model.add_points(
            self.inference_state,
            image,
            np.array(points),
            np.array(labels),
            frame_idx
        )

    def add_box(self, frame_idx, box):
        if self.inference_state is None:
            self.inference_state = self.model.get_inference_state()
        
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        
        self.inference_state = self.model.add_box(
            self.inference_state,
            image,
            np.array(box),
            frame_idx
        )

    def propagate(self):
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def visualize(self, stride=30):
        plt.close("all")
        for frame_idx in range(0, self.frame_count, stride):
            if frame_idx not in self.video_segments:
                continue
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {frame_idx}")
            plt.imshow(frame)
            
            for obj_id, mask in self.video_segments[frame_idx].items():
                show_mask(mask, plt.gca(), obj_id=obj_id)
            
            plt.show()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    video_sam2 = VideoSAM2()
    video_sam2.load_video("demo/HODOR_23.mp4")
    video_sam2.add_points(10, np.array([[100, 100]]), np.array([1]))
    video_sam2.propagate()
    video_sam2.visualize()