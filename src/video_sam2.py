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

    def load_video(self, video_dir):
        self.inference_state = self.predictor.init_state(video_path=video_dir)

    def add_object(self, frame_idx, obj_id, points=None, labels=None, box=None):
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box
        )
        return out_obj_ids, out_mask_logits

    def propagate(self):
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video():
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def get_segments(self):
        return self.video_segments

    def reset(self):
        self.predictor.reset_state()
        self.video_segments = {}

# segments = predictor.get_segments()
if __name__ == "__main__":
    video_sam2 = VideoSAM2()
    video_sam2.load_video("demo/HODOR_23.mp4")
    video_sam2.add_points(10, np.array([[100, 100]]), np.array([1]))
    video_sam2.propagate()
    video_sam2.visualize()