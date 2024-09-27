# PAL Video Processing Tool
- Tony Wang
- 09/25/2024
---


This tool is designed for use in the PAL lab to annotate video, particularly for robotics demos.


## Features

- [x] **Object Detection with Text**: Collabrating with `IDEA-Research`, this library use the SOTA Openset Detection Model: `Grounding DINO Series` to detect objects in the video and overlays text annotations. With API access, this tool is light weight, easy to use and solely built for PAL Lab members. 

- [ ] **Masking**: With `Segment Anything 2`, the tool can render masks to specific objects or regions within the video.

- [ ] **Fancy Visualization**: With `Supervision`, more advanced visualization techniques to enhance the presentation of video content, to be developed...

## Installation

To install the PAL Video Processing Tool, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Everloom-129/pal_video_tool.git
    ```
2. Navigate to the project directory:
    ```sh
    cd pal_video_tool
    ```
3. Install the required dependencies in conda environment:
    ```sh
    conda create -n gd16 python=3.8
    conda activate gd16
    pip install -r requirements.txt
    cd idea-research-api # IDEA-Research's SDK repo
    pip install -e .
    ```

## Usage

To use the tool, run the following command:

```sh
python main.py --input <input_video> --output <output_video> --prompts <detection_prompts> #Optional, default output video will be <input_video_name>_pal.mp4
```


## Contributing

Welcome suggestions! Please raise any issue / PR if you are interested in it.
- [ ] Add more scripts for detection
- [ ] Add more scripts for segmentation
- [ ] Add more scripts for visualization

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact
For any questions or feedback, please contact the PAL lab team at `tonyw3@seas.upenn.edu`