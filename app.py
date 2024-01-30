import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
import requests
import json
import torchvision
import torch
import psutil
import time
import shutil
from glob import glob
import zipfile
import PIL
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
import time
sys.path.append(sys.path[0] + "/tracker")
sys.path.append(sys.path[0] + "/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
from tools.painter import mask_painter

try:
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")

RESIZE_TO_PREV_FACTOR = 2


def round2(num):
    num = round(num)
    if num % 2 != 0:
        num += 1
    return num


def resize_by(image, resize_factor=0.5):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.resize((round(image.size[0] * resize_factor), round(image.size[1] * resize_factor)), Image.ANTIALIAS)


def resize_to_preview(image):
    return resize_by(image, resize_factor=1./RESIZE_TO_PREV_FACTOR)


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath


# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, interactive_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    print(video_path)
    user_name = time.time()
    operation_log = [("", ""), ("Upload video already. Try click the image for adding targets to track.", "Normal")]

    with zipfile.ZipFile(video_path.name) as zip_ref:
        img_file_names = [_.filename for _ in zip_ref.infolist() if not _.is_dir()]
        img_file_names = sorted(img_file_names)

        frames = [None] * len(img_file_names)
        exifs = [None] * len(img_file_names)

        def extract_frame(img_file_name_i):
            img_file_name, i = img_file_name_i
            with zip_ref.open(img_file_name) as file:
                image = Image.open(BytesIO(file.read()))
                image = PIL.ImageOps.exif_transpose(image)
                max_length = max(image.size)
                resize_ratio = interactive_state["resize_ratio"] * 1600. / max_length
                image = image.resize((round2(image.size[0] * resize_ratio), round2(image.size[1] * resize_ratio)), Image.ANTIALIAS)
                frames[i] = np.array(image)
                exifs[i] = image.info.get('exif', None)
        with Pool() as pool:
            list(tqdm(pool.imap_unordered(extract_frame, zip(img_file_names, range(len(img_file_names)))), total=len(img_file_names)))

    image_size = (frames[0].shape[0], frames[0].shape[1])
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path.name)[-1] + '_' + time.strftime("%Y%m%d-%H%M%S"),
        "exifs": exifs,
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)] * len(frames),
        "select_frame_number": 0,
        "fps": 3
    }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)
    interactive_state['track_end_number'] = len(frames)
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return (video_state, video_info, interactive_state, resize_to_preview(video_state["origin_images"][0]),
            gr.update(visible=True, maximum=len(frames), value=1),
            gr.update(visible=True, maximum=len(frames), value=len(frames)),
            gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True, value=operation_log))


def run_example(example):
    return video_input


# get the select frame from gradio slider


def select_template(image_selection_slider, video_state, mask_dropdown):
    image_selection_slider = image_selection_slider - 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # update the masks when select a new template frame
    if mask_dropdown:
        print("ok")
    operation_log = [("", ""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider), "Normal")]

    return resize_to_preview(video_state["painted_images"][image_selection_slider]), video_state, operation_log


# set the tracking end frame
def get_end_number(track_pause_number_slider, interactive_state):
    track_pause_number_slider = track_pause_number_slider - 1
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("", ""), ("Set the tracking finish at frame {}".format(track_pause_number_slider), "Normal")]
    return interactive_state, operation_log


def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state


# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt: gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    print('Enter sam_refine')

    coordinate = "[[{},{},{}]]".format(evt.index[0] * RESIZE_TO_PREV_FACTOR,
                                       evt.index[1] * RESIZE_TO_PREV_FACTOR,
                                       int(point_prompt == "Positive"))
    if point_prompt == "Positive":
        interactive_state["positive_click_times"] += 1
    else:
        interactive_state["negative_click_times"] += 1

    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    print('Done model.samcontroler.sam_controler.reset_image()')
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    print('Done model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])')
    prompt = get_prompt(click_state=click_state, click_input=coordinate)
    print('Done prompt = get_prompt(click_state=click_state, click_input=coordinate)')

    mask, logit, painted_image = model.first_frame_click(
        image=video_state["origin_images"][video_state["select_frame_number"]],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )
    print('Done model.first_frame_click(')

    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("", ""), ("Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment", "Normal")]
    print('Done sam_refine...')
    return resize_to_preview(painted_image), video_state, interactive_state, operation_log


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("", ""), ("Added a mask, use the mask select for target tracking.", "Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("", "")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[], []], operation_log


def clear_click(video_state, interactive_state):
    click_state = [[], []]
    interactive_state["negative_click_times"] = 0
    interactive_state["positive_click_times"] = 0
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("", ""), ("Clear points history and refresh the image.", "Normal")]
    return resize_to_preview(template_frame), interactive_state, click_state, operation_log


def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number + 2)

    operation_log = [("", ""), ("Select {} for tracking".format(mask_dropdown), "Normal")]
    return resize_to_preview(select_frame), operation_log


def generate_zip(video_state):
    save_masks(video_state)
    zips_folder = './result/zips/'
    os.makedirs(zips_folder, exist_ok=True)
    zip_file_path = '{}/{}'.format(zips_folder, video_state["video_name"].split('.')[0])
    if os.path.exists(zip_file_path+'.zip'):
        os.remove(zip_file_path+'.zip')
    shutil.make_archive(zip_file_path, 'zip', './result/mask/{}'.format(video_state["video_name"].split('.')[0]))
    return zip_file_path + '.zip'


# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("", ""), ("Track the selected masks, and then you can select the masks.", "Normal")]
    model.xmem.clear_memory()
    following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]+1]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1, len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(template_mask + interactive_state["multi_mask"]["masks"][mask_number] * (mask_number + 1),
                                    0,
                                    mask_number + 1)
        video_state["masks"][video_state["select_frame_number"]] = template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.", "Error"), ("", "")]
        # return video_output, video_state, interactive_state, operation_error
    masks, _, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.xmem.clear_memory()

    video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]+1] = masks
    video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]+1] = painted_images

    video_output = generate_video_from_frames(
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]+1],
        output_path="./result/track/{}.mp4".format(video_state["video_name"]),
        fps=fps,
        start_id=video_state["select_frame_number"])  # import video_input to name the output video
    interactive_state["inference_times"] += 1

    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(
        interactive_state["inference_times"],
        interactive_state["positive_click_times"] + interactive_state["negative_click_times"],
        interactive_state["positive_click_times"],
        interactive_state["negative_click_times"]))

    if interactive_state["mask_save"]:
        save_masks(video_state)
    return video_output, video_state, interactive_state, operation_log


def save_masks(video_state):
    os.makedirs('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), exist_ok=True)
    os.makedirs('./result/mask/{}/masks'.format(video_state["video_name"].split('.')[0]), exist_ok=True)
    i = 0
    print("save mask")
    for mask, img, exif in zip(video_state["masks"], video_state["origin_images"], video_state["exifs"]):
        # np.save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.npy'.format(i)), mask)
        Image.fromarray(((mask > 0) * 255).astype(np.uint8)).save(os.path.join('./result/mask/{}/masks'.format(video_state["video_name"].split('.')[0]), '{:05d}.png'.format(i)))
        Image.fromarray(img).save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.jpg'.format(i)),exif=exif)
        i += 1

def add_text_to_image(img, text, font_size=240):
    img = Image.fromarray(img)
    # Open the image
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Load a font (change the font file path as needed)
    font = ImageFont.truetype("/home/gdudovitch/Track-Anything/FreeMono.ttf", 80)

    # Specify the position, font size, and color of the text
    x, y = img.size[1] // 3, img.size[0] // 5
    font_size = font_size

    # Add text to the image
    draw.text((x, y), text, font=font, fill=128, font_size=font_size)

    # Save the modified image
    return np.array(img)


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30, start_id=0):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    print('generate_video_from_frames')
    for i in range(len(frames)):
        frames[i] = add_text_to_image(frames[i], str(i + start_id+1))
    frames = torch.from_numpy(np.asarray(frames))

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


# args, defined in track_anything.py
args = parse_augment()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    'vit_h': "sam_vit_h_4b8939.pth",
    'vit_l': "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}
SAM_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
# xmem_checkpoint = "XMem-s012.pth"
# xmem_checkpoint = "XMem.pth"
xmem_checkpoint = "XMem-no-sensory.pth"
xmem_checkpoint_url = f"https://github.com/hkchengrex/XMem/releases/download/v1.0/{xmem_checkpoint}"

folder = "./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
args.port = 7860
args.device = "cuda:0"
args.mask_save = False

# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, args)

title = """<p><h1 align="center">Track-Anything</h1></p>
    """
description = """<p>Gradio demo for Track Anything, a flexible and interactive tool for video object tracking, segmentation. I To use it, simply upload your video, or click one of the examples to load them. Code: <a href="https://github.com/gaomingqi/Track-Anything">https://github.com/gaomingqi/Track-Anything</a> <a href="https://huggingface.co/spaces/watchtowerss/Track-Anything?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>"""

# +
with gr.Blocks() as demo:
    iface = demo
    """
        state for 
    """
    click_state = gr.State([[], []])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times": 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1
    })

    video_state = gr.State(
        {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "select_frame_number": 0,
            "fps": 30
        }
    )
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        # for user video input
        with gr.Column():
            with gr.Row():
                video_input = gr.File(label='Input Image-Seq', file_count='single')
                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
                    resize_ratio_slider = gr.Slider(minimum=0.02, maximum=1, step=0.02, value=1, label="Resize ratio", visible=True, interactive=True)

            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    # click points settings, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point prompt",
                                interactive=True,
                                visible=False)
                            remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False)
                            clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)
                            Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False)
                    template_frame = gr.Image(type="pil", interactive=True, elem_id="template_frame", visible=False)
                    image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                    track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(value="Tracking", visible=False)

                with gr.Column():
                    run_status = gr.HighlightedText(value=[("Text", "Error"), ("to be", "Label 2"), ("highlighted", "Label 3")],
                                                    visible=False)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                    video_output = gr.Video(visible=False)
                    generate_zip_btn = gr.Button(value="Generate Zip", interactive=True, visible=True)
                    download_file_zip = gr.File(label="Zipped results")

    video_input.change(fn=get_frames_from_video,
                       inputs=[video_input, interactive_state],
                       outputs=[video_state, video_info, interactive_state, template_frame,
                                image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, Add_mask_button, template_frame,
                                tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status])

    # second step: select images from slider
    image_selection_slider.release(fn=select_template,
                                   inputs=[image_selection_slider, video_state],
                                   outputs=[template_frame, video_state, run_status], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number,
                                      inputs=[track_pause_number_slider, interactive_state],
                                      outputs=[interactive_state, run_status], api_name="end_image")
    resize_ratio_slider.release(fn=get_resize_ratio,
                                inputs=[resize_ratio_slider, interactive_state],
                                outputs=[interactive_state], api_name="resize_ratio")

    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    generate_zip_btn.click(
        fn=generate_zip,
        inputs=[video_state],
        outputs=[download_file_zip]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status]
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status]
    )

    # clear input
    video_input.clear(
        lambda: (
            {
                "user_name": "",
                "video_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "select_frame_number": 0,
                "fps": 30
            },
            {
                "inference_times": 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {
                    "mask_names": [],
                    "masks": []
                },
                "track_end_number": 0,
                "resize_ratio": 1
            },
            [[], []],
            None,
            None,
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=[]), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False)

        ),
        [],
        [
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click,
            Add_mask_button, template_frame, tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn=clear_click,
        inputs=[video_state, interactive_state],
        outputs=[template_frame, interactive_state, click_state, run_status],
    )
    # set example
    gr.Markdown("##  Examples")
    gr.Examples(
        examples=list(glob(os.path.join(os.path.dirname(__file__), "./test_sample/*.zip"))),
        fn=run_example,
        inputs=[
            video_input
        ],
        outputs=[video_input],
        # cache_examples=True,
    )
    iface.queue()

if __name__ == "__main__":
    demo.launch(debug=False, server_port=args.port, server_name="0.0.0.0", share=True)
# iface.launch(debug=True, enable_queue=True)
