import base64
import glob
import json
import os
import shutil
import time
from io import BytesIO
from typing import List

import gradio as gr
import moviepy.editor as mp
import pandas as pd
import requests
import yaml
#from clip_interrogator import Config, Interrogator
from PIL import Image

from src.animatediff.cli import generate
from webui_demo.test_workflow import call_comfy


def gif_to_mp4(gif_filename, mp4_filename):
    """
    Convert a GIF file to an MP4 file.

    :param gif_filename: The filename of the GIF file to convert.
    :param mp4_filename: The filename for the output MP4 file.
    """
    # Load the GIF file
    clip = mp.VideoFileClip(gif_filename)

    # Write the clip as an MP4 file
    clip.write_videofile(mp4_filename)

def resize_divisible(image, max_size=1024, divisible=16):
    W, H = image.size
    if W > H:
        W, H = max_size, int(max_size * H / W)
    else:
        W, H = int(max_size * W / H), max_size
    W = W - W % divisible
    H = H - H % divisible
    image = image.resize((W, H))
    return image


def pil_image_to_base64(pil_image):
    # Create a BytesIO object to hold the image data
    image_buffer = BytesIO()

    # Save the PIL Image to the BytesIO object in JPEG format
    pil_image.save(image_buffer, format='JPEG')

    # Get the bytes from the BytesIO object
    image_bytes = image_buffer.getvalue()

    # Encode the bytes as base64
    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

    return base64_encoded

ckpt_config = yaml.load(open("webui_demo/model_maps.yaml"), Loader=yaml.FullLoader)
SAVE_DIR = "gradio_output"
CONTROL_IMAGE_DIR = "data/temp/control_image"
CONFIG_CONTROL_IMAGE_DIR = "temp/control_image"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CONTROL_IMAGE_DIR, exist_ok=True)
# ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

# def generate_prompt(image: Image.Image):
#     prompt = ci.interrogate_fast(image)
#     return prompt
# Define a function to process the input configuration
from PIL import Image

def center_crop_image(image, ratio):
    # Load the image
    width, height = image.size

    # Determine new dimensions based on the ratio
    if ratio == "square":
        new_size = min(width, height)
        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2
    elif ratio == "wide":
        # Assuming a 16:9 ratio for "wide"
        if width / height > 16/9:
            new_width = height * 16/9
            new_height = height
        else:
            new_width = width
            new_height = width * 9/16
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
    elif ratio == "tall":
        # Assuming a 9:16 ratio for "tall"
        if height / width > 16/9:
            new_height = width * 16/9
            new_width = width
        else:
            new_height = height
            new_width = height * 9/16
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
    else:
        raise ValueError("Unsupported ratio. Choose 'square', 'wide', or 'tall'.")

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def get_prompt(image: Image.Image):
    url = "https://ai-api.sankakucomplex.com/sdapi/v1/tagging"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "input_image": pil_image_to_base64(image)
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    tokens = response.json()
    sorted_tokens = dict(sorted(tokens.items(), key=lambda item: item[1], reverse=True))
    top_k_tokens = list(sorted_tokens.keys())[:10]
    return ','.join(top_k_tokens)

def process_config(
    conditional_image: Image.Image,
    head_prompt: str,
    prompt_map: pd.DataFrame,
    n_prompt: str,
    length: int,
    use_v2: bool,
    denoise_strength: float,
    v1_duration: float,
    ratio: str = "square",
):
    # conditional_image = resize_divisible(conditional_image, 1024)
    # conditional_image = center_crop_image(conditional_image, ratio)
    # conditional_image.save("temppp.webp")

    if not use_v2:
        prompt = head_prompt
        save_image_path = "/home/ai/animatediff-cli-prompt-travel/conditional_image_comfyui.png"
        conditional_image.save(save_image_path)
        output_fn = call_comfy(save_image_path, prompt, denoise_strength, v1_duration, length)
        print("output fn", output_fn)
        start_waiting = time.time()

        output_files = []
        while time.time() - start_waiting < 720 and not output_files:
            output_files = glob.glob(f"{output_fn}*.gif")
            time.sleep(1)
        print(output_files)
        output_mp4_file = "temp/output.mp4"
        time.sleep(10)
        gif_to_mp4(output_files[0], output_mp4_file)
        shutil.rmtree(SAVE_DIR, ignore_errors=True)
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.system(
            f"cd Real-ESRGAN && python inference_realesrgan_video.py -i ../{output_mp4_file} -n realesr-animevideov3 -s 2 -o ../{SAVE_DIR}/upscaled"
        )
        output_file = glob.glob(f"{SAVE_DIR}/upscaled/**")[0]
        return output_file
    prompt_map = prompt_map.to_dict()
    prompt_map_dict = {
        str(k): v
        for k, v in zip(prompt_map["start_frame"].values(), prompt_map["prompt"].values())
        if str(k) != ""
    }
    image_path = f"{CONTROL_IMAGE_DIR}/000.png"
    config_image_path = f"{CONFIG_CONTROL_IMAGE_DIR}/000.png"
    config = {
        "name": "demo",
        # "vae_path": "anything-v4.5-pruned-fp16.ckpt",
        "path": "anything_v5_ink.safetensors",
        "motion_module": ckpt_config["motion_module"]["mm_sd_v15_v2"],
        "motion_lora_map": {
            # ckpt_config["motion_lora_module"][motion_lora_name]: motion_lora_scale,
        },
        "lora_map": {
        },
        "scheduler": "ddim",      # "ddim","euler","euler_a","k_dpmpp_2m", etc...
        "seed": [42],
        "steps": 40,
        "guidance_scale": 7,
        "clip_skip": 1,
        "head_prompt": head_prompt,
        "prompt_map": prompt_map_dict,
        "tail_prompt": "",
        "n_prompt": [n_prompt],
        "context_schedule": "uniform",
        "ip_adapter_map": {
            "enable": True,
            "input_image_dir": CONFIG_CONTROL_IMAGE_DIR,
            "prompt_fixed_ratio": 0.5,
            "save_input_image": True,
            "resized_to_square": False,
            "scale": 0.5,
            "is_plus_face": False,
            "is_plus": True,
            "is_light": False,
        },
        "upscale_config": {
            "scheduler": "k_dpmpp_sde",
            "steps": 25,
            "strength": 0.5,
            "guidance_scale": 7,
            # "controlnet_tile": {
            #     "enable": True,
            #     "controlnet_conditioning_scale": 1.0,
            #     "guess_mode": False,
            #     "control_guidance_start": 0.0,
            #     "control_guidance_end": 1.0,
            # },
            # "controlnet_line_anime": {
            #     "enable": False,
            #     "controlnet_conditioning_scale": 1.0,
            #     "guess_mode": False,
            #     "control_guidance_start": 0.0,
            #     "control_guidance_end": 1.0,
            # },
            # "controlnet_ip2p": {
            #     "enable": False,
            #     "controlnet_conditioning_scale": 0.5,
            #     "guess_mode": False,
            #     "control_guidance_start": 0.0,
            #     "control_guidance_end": 1.0,
            # },
            "controlnet_ref": {
                "enable": True,
                "use_frame_as_ref_image": False,
                "use_1st_frame_as_ref_image": True,
                "ref_image": config_image_path,
                "attention_auto_machine_weight": 1.0,
                "gn_auto_machine_weight": 1.0,
                "style_fidelity": 0.75,
                "reference_attn": True,
                "reference_adain": False,
                "scale_pattern": [1.0],
            },
        },
    }
    config["output"] = {"format": "mp4", "fps": 8, "encode_param": {"crf": 10}}
    with open("config.json", "w") as f:
        json.dump(config, f)
    command = f"animatediff generate -c config.json -W {int(512)} -H {int(512)} -L {int(length)} -C 16 -o {SAVE_DIR}"
    conditional_image.save(image_path)
    shutil.rmtree(SAVE_DIR)
    os.system(command)
    gif_file = glob.glob(SAVE_DIR + "/**/*.mp4")[0]
    os.system(
        f"cd Real-ESRGAN && python inference_realesrgan_video.py -i ../{gif_file} -n realesr-animevideov3 -s 2 -o ../{SAVE_DIR}/upscaled"
    )
    output_file = glob.glob(f"{SAVE_DIR}/upscaled/**")[0]
    return output_file

with gr.Blocks() as app:
    state = gr.State({})
    with gr.Row():
        with gr.Column():
            reference_image = gr.Image(label="Reference Image", type="pil", value="assets/images/conditional_image_comfyui.png")
            ratio = gr.Dropdown(["square", "wide", "tall"], value="square")
            v1_duration = gr.Slider(minimum=1.0, maximum=10.0, value=2.0)
            length = gr.Slider(label="length", minimum=8, maximum=512, value=16, step=16)
            with gr.Accordion("Prompt Travel Settings", open=True):
                head_prompt = gr.Textbox(
                    label="head_prompt",
                    value="1girl,yoimiya (genshin impact),fireworks,genshin impact,breasts,female,solo,sarashi,looking at viewer,gloves, shy, crying",
                )
                ci_button = gr.Button("Generate Prompt")
                use_v2 = gr.Checkbox(label="Use Prompt Travel (IMPORTANT)", value=False)
                prompt_map = gr.Dataframe(
                    headers=["start_frame", "prompt"],
                    datatype=["number", "str"],
                    row_count=1,
                    col_count=(2, "fixed"),
                    label="prompt_map",
                    value=[
                        [0, "((brown eyes)) ((sunny weather))"],
                        [32, "((blue eyes)), ((snow weather))"],
                        [64, "((yellow eyes)), ((summer weather)), hot sunny day"],
                        [96, "((summer)), ((under tree)), ((sunny day))"],
                    ],
                )
                denoise_strength = gr.Slider(label="denoise_strength", minimum=0.0, maximum=1.0, value=0.6, step=0.05)
                n_prompt = gr.Textbox(
                    label="n_prompt",
                    value="EasyNegativeV2",
                )
        with gr.Column():
            output_video = gr.Video(value="assets/video/output_out.mp4")
            btn_process = gr.Button("Generate")
            gr.Examples(
                examples=[
                    ["assets/images/conditional_image_comfyui.png","1girl,yoimiya (genshin impact),fireworks,genshin impact,breasts,female,solo,sarashi,looking at viewer,gloves, shy, crying","assets/video/output_out.mp4", False, 16],
#                    ["assets/images/003.jpeg", "anime girl with long hair and a flower in her hair", "assets/video/003.mp4", True, 128]
                ],
                inputs=[reference_image, head_prompt, output_video, use_v2, length],
                outputs=[output_video],
            )

    ci_button.click(
        fn=get_prompt,
        inputs=[reference_image],
        outputs=head_prompt
    )
    btn_process.click(
        fn=process_config,
        inputs=[reference_image, head_prompt, prompt_map, n_prompt, length, use_v2, denoise_strength, v1_duration, ratio],
        outputs=output_video
    )

# app.queue().launch(share=True, server_port=4449, server_name="0.0.0.0", root_path="/demo/img-to-video", auth=("sankaku-aiart", "!h&8EEX4AIyKp#f9"))
app.queue().launch(share=False, server_port=4449, server_name="0.0.0.0", debug=True, show_error=True)
