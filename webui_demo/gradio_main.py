import gradio as gr
from typing import List
import yaml
import pandas as pd
from PIL import Image
import os
import glob
import shutil
import json
from src.animatediff.cli import generate
import os

ckpt_config = yaml.load(open("webui_demo/model_maps.yaml"), Loader=yaml.FullLoader)
SAVE_DIR = "gradio_output"
CONTROL_IMAGE_DIR = "temp/control_image"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CONTROL_IMAGE_DIR, exist_ok=True)


# Define a function to process the input configuration
def process_config(
    conditional_image: Image.Image,
    head_prompt: str,
    prompt_map: pd.DataFrame,
    n_prompt: str,
    length: int,
):
    prompt_map = prompt_map.to_dict()
    prompt_map_dict = {
        str(k): v
        for k, v in zip(prompt_map["start_frame"].values(), prompt_map["prompt"].values())
        if str(k) != ""
    }
    image_path = f"{CONTROL_IMAGE_DIR}/conditional_image.png"
    config = {
        "name": "demo",
        "path": ckpt_config["base_model"]["counterfeit"],
        "motion_module": ckpt_config["motion_module"]["mm_sd_v15_v2"],
        "motion_lora_map": {
            # ckpt_config["motion_lora_module"][motion_lora_name]: motion_lora_scale,
        },
        "lcm_map": {
            "enable": False,
            "start_scale": 0.15,
            "end_scale": 0.75,
            "gradient_start": 0.2,
            "gradient_end": 0.75,
        },
        "seed": [42],
        "steps": 25,
        "guidance_scale": 7,
        "clip_skip": 1,
        "head_prompt": head_prompt,
        "prompt_map": prompt_map_dict,
        "tail_prompt": "",
        "n_prompt": [n_prompt],
        "img2img_map": {
            "enable": True,
            "init_img_dir": CONTROL_IMAGE_DIR,
            "save_init_image": True,
            "denoising_strength": 0.85,
        },
        "controlnet_map": {
            "input_image_dir": CONTROL_IMAGE_DIR,
            "max_samples_on_vram": 0,
            "max_models_on_vram": 1,
            "save_detectmap": True,
            "preprocess_on_gpu": True,
            "is_loop": True,
            "controlnet_tile": {
                "enable": True,
                "use_preprocessor": True,
                "preprocessor": {"type": "none", "param": {}},
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_ip2p": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_lineart_anime": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_openpose": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_softedge": {
                "enable": True,
                "use_preprocessor": True,
                "preprocessor": {"type": "softedge_pidsafe", "param": {}},
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_shuffle": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_depth": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_canny": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_inpaint": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_lineart": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_mlsd": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_normalbae": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_scribble": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_seg": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "qr_code_monster_v1": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "qr_code_monster_v2": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_mediapipe_face": {
                "enable": True,
                "use_preprocessor": True,
                "guess_mode": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "control_scale_list": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "controlnet_ref": {
                "enable": False,
                "ref_image": "ref_image/ref_sample.png",
                "attention_auto_machine_weight": 0.3,
                "gn_auto_machine_weight": 0.3,
                "style_fidelity": 0.5,
                "reference_attn": True,
                "reference_adain": False,
                "scale_pattern": [1.0],
            },
        },
        "upscale_config": {
            "scheduler": "k_dpmpp_sde",
            "steps": 20,
            "strength": 0.5,
            "guidance_scale": 10,
            "controlnet_tile": {
                "enable": True,
                "controlnet_conditioning_scale": 1.0,
                "guess_mode": False,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
            },
            "controlnet_line_anime": {
                "enable": False,
                "controlnet_conditioning_scale": 1.0,
                "guess_mode": False,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
            },
            "controlnet_ip2p": {
                "enable": False,
                "controlnet_conditioning_scale": 0.5,
                "guess_mode": False,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
            },
            "controlnet_ref": {
                "enable": False,
                "use_frame_as_ref_image": False,
                "use_1st_frame_as_ref_image": False,
                "ref_image": "ref_image/path_to_your_ref_img.jpg",
                "attention_auto_machine_weight": 1.0,
                "gn_auto_machine_weight": 1.0,
                "style_fidelity": 0.25,
                "reference_attn": True,
                "reference_adain": False,
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
    os.system(f"cd Real-ESRGAN && python inference_realesrgan_video.py -i ../{gif_file} -n realesr-animevideov3 -s 2 -o ../{SAVE_DIR}/upscaled")
    output_file = glob.glob(f"{SAVE_DIR}/upscaled/**")[0]
    return gr.Video.update(value=output_file)


# Create the Gradio interface
iface = gr.Interface(
    fn=process_config,
    inputs=[
        gr.Image(label="Reference Image", type="pil", value="assets/images/003.jpeg"),
        gr.Textbox(label="head_prompt", value="1girl, solo, flower, long hair, outdoors, letterboxed, school uniform, day, sky, looking up, short sleeves, parted lips, shirt, cloud, black hair"),
        gr.Dataframe(
            headers=["start_frame", "prompt"],
            datatype=["number", "str"],
            row_count=1,
            col_count=(2, "fixed"),
            label="prompt_map",
            value=[[0, "((red eyes))"], [32, "((blue eyes))"], [64, "((yellow eyes))"], [96, "((green eyes))"]],
        ),
        # gr.Textbox(label="tail_prompt", value="8k uhd, dslr, soft lighting, high quality"),
        gr.Textbox(
            label="n_prompt",
            value="mutated hands and fingers:1.4, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated",
        ),
        gr.Number(label="length", minimum=8, maximum=512, value=128),
    ],
    outputs=gr.Video(value="assets/video/003.mp4"),
    examples=[

    ]
)

# Run the Gradio app
iface.queue().launch(share=False, debug=True, show_error=True, server_port=10008)
