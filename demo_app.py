import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import sys
sys.path.append('./lstn')

import torch
import torchvision
import numpy as np

import cv2
from skimage import measure
import gradio as gr

from sam import sam_model_registry
from sam.utils.transforms import ResizeLongestSide

from tool.mask_tool import mask_painter
from lstn.tools.demo_eval import get_parser
from lstn.networks.managers.eval_demo import Evaluator


# initial sam
sam_checkpoint = "./checkpoints/chk_sam/finetune.pth"  # 41
model_type = "vit_b"
# path = "/data/wangyh/data4/Datasets/shadow/video_new/visha4/test"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)

device = "cuda"
sam_model.to(device=device)
sam_model.eval()


def run_example(video_input):
    return video_input


def get_frames_from_video(video_input, video_state):
    """video -> frame
    Get video meta info. 

    Args:
        video_path: input video path, e.g., 'bike1.mp4'
        video_state: all video info to be logged
    Return 
        video_state
        video_info
        first_video_frame
    """
    video_path = video_input
    frames = []
    cap = cv2.VideoCapture(video_path)  # load video
    fps = cap.get(cv2.CAP_PROP_FPS)  # get frame number
    while True:
        ret, frame = cap.read()
        if ret is True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    frame_num = len(frames)
    image_size = (frames[0].shape[0], frames[0].shape[1])  # (H, W,)

    video_state = {
        "video_name": os.path.split(video_path)[-1],  # video.mp4
        "image_size": image_size,
        "origin_images": frames,  # [frame1, frame2, ...]
        "bboxes": [],
        "masks": [],
        "frame_num": frame_num,
        "fps": fps
    }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(\
        video_state["video_name"], video_state["fps"], len(frames), image_size)

    return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True)



# get frist frame mask from finetuned sam
def generate_mask_sam(first_frame, video_state):
    """
    Args:
        first frame: first video frame
        video_state: video info
    Return 
        video_state
        mask_paint: image with its corresponding mask
        update:
    """
    # transform
    first_frame_image = np.array(first_frame['image'])[:, :, :3]
    first_frame_mask = np.array(first_frame["mask"].convert('L'))[:, :]  # user prompt

    # extract image feature
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)  # 1024
    resize_image = sam_trans.apply_image(first_frame_image)  # 等比例填充到一边为1024
    image_tensor = torch.as_tensor(resize_image, device=device)
    input_image_torch = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_image = sam_model.preprocess(input_image_torch)
    original_image_size = first_frame_image.shape[:2]
    input_size = tuple(input_image_torch.shape[-2:])

    # get bbox
    bboxes = get_bbox_from_mask(first_frame_mask)

    # sam predict
    with torch.no_grad():
        box = sam_trans.apply_boxes(bboxes, (original_image_size))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

        image_embedding = sam_model.image_encoder(input_image)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    mask_save = (upscaled_masks>0.5)[0].detach().squeeze(0).cpu().numpy()
    mask_save = np.array(mask_save * 255).astype(np.uint8)

    video_state['masks'].append(mask_save)
    mask_paint = mask_painter(first_frame_image, mask_save, mask_color=np.random.randint(0, 81))

    return video_state, mask_paint, gr.update(visible=True)


# vsd
def generate_mask_vsd(video_state):
    # initial vsd
    cfg = get_parser()
    evaluator = Evaluator(cfg, video_state["origin_images"], video_state["masks"])
    final_mask_list, log, log_time = evaluator.evaluating()
    video_state["masks"] += final_mask_list


    # 将image与相应的mask合成一张图
    video_paint = []
    color = np.random.randint(0, 81)
    for i in range(len(video_state["masks"])):
        _image = video_state["origin_images"][i]
        _mask = video_state["masks"][i]
        mask_paint = mask_painter(_image, _mask, mask_color=color)
        video_paint.append(mask_paint)

    video_output = generate_video_from_frames(video_paint, output_path="./assets/"+video_state["video_name"].split('.')[0]+"_mask.mp4")

    return video_output, log, log_time




# get bbox according to user's paint
def get_bbox_from_mask(user_mask):
    final_bbox = []
    # get one hot mask
    labels, num = measure.label(user_mask, connectivity=2, return_num=True)
    properties = measure.regionprops(labels)
    valid_label = set()
    for prop in properties:
        if prop.area > 50:
            valid_label.add(prop.label)
    valid_label = np.array(list(valid_label))
    one_hot_mask = (labels[None,:,:]==valid_label[:,None,None])

    # extract bbox
    if len(valid_label) >= 8 or len(valid_label)==0:
        y_indices, x_indices = np.where(user_mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = user_mask.shape
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)
        bboxes = np.array([x_min, y_min, x_max, y_max])
        final_bbox.append(bboxes.tolist())
    else:
        region_num = one_hot_mask.shape[0]
        for i in range(region_num):
            _mask = one_hot_mask[i]
            # get bbox
            y_indices, x_indices = np.where(_mask > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = _mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bboxes = np.array([x_min, y_min, x_max, y_max])
            final_bbox.append(bboxes.tolist())

    return np.array(final_bbox)



# generate video after vsd inference
def generate_video_from_frames(frames, output_path, fps=30):
    """frame -> video
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """

    frames = torch.from_numpy(np.asarray(frames))  # [frame, h, w, c]
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path





title = """<p><h1 align="center">Detect-AnyShadow</h1></p>"""
description = """<p>Gradio demo for Video Shadow Detection<p>"""

with gr.Blocks() as vsd:
    # initial state
    video_state = gr.State(
        {
            "video_name": "",
            "image_size": None,
            "origin_images": [],
            "bboxes": None,
            "masks": [],
            "frame_num": 0,
            "fps": 30
        }
    )

    # title and description
    gr.Markdown(title)
    gr.Markdown(description)

    # start content
    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(autosize=True)
                    extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary")
                with gr.Column():
                    video_info = gr.Textbox(value="video info here", label="Video Info")
                    gr.Examples(examples=["./assets/bike1.mp4"], 
                                fn=run_example, 
                                inputs=[video_input], 
                                outputs=[video_input],)

            with gr.Row():
                with gr.Column():
                    first_frame = gr.Image(type="pil", interactive=True, tool='sketch', elem_id="first_frame", visible=True)
                    first_mask_buttion = gr.Button(value="Get first mask", interactive=True, variant="primary")
                with gr.Column():
                    first_mask = gr.Image(type="pil", interactive=True, elem_id="first_mask", visible=True)
                    # first_mask = gr.Image(type="pil", interactive=True, elem_id="first_mask", visible=True).style(height=360)

            with gr.Row():
                with gr.Column():
                    vsd_final_video = gr.Video(autosize=True)
                    final_video_buttion = gr.Button(value="Get final video", interactive=True, variant="primary")
                with gr.Column():
                    process_info = gr.Textbox(value="infer info here", label="Process Info")
                    time_info = gr.Textbox(value="time consumption here", label="Time Info")


    # 1. first, get frames from video
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[video_input, video_state],
        outputs=[video_state, video_info, first_frame, first_frame]
    )

    # 2. get bbox and use sam to generate first mask
    first_mask_buttion.click(
        fn=generate_mask_sam,
        inputs=[first_frame, video_state],
        outputs=[video_state, first_mask, first_mask]
    )

    # 3. generate masks using lstn
    final_video_buttion.click(
        fn=generate_mask_vsd,
        inputs=[video_state, ],
        outputs=[vsd_final_video, process_info, time_info]
    )


vsd.launch(server_name="0.0.0.0")
