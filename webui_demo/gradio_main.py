import gradio as gr
from typing import List
import yaml
import pandas as pd
import os
import glob
import shutil
import json
from src.animatediff.cli import generate
import os

ckpt_config = yaml.load(open("webui_demo/model_maps.yaml"), Loader=yaml.FullLoader)
SAVE_DIR = "gradio_output"
os.makedirs(SAVE_DIR, exist_ok=True)
# Define a function to process the input configuration
def process_config(
    name: str,
    base_model_name: str,
    motion_module_name: str,
    motion_lora_name: str,
    motion_lora_scale: float,
    use_lcm: bool,
    seed: int,
    steps: int,
    guidance_scale: float,
    clip_skip: int,
    head_prompt,
    prompt_map: pd.DataFrame,
    tail_prompt: str,
    n_prompt: str,
    width: int,
    height: int,
    fps: int,
    length: int,
    context: int,
):
    prompt_map = prompt_map.to_dict()
    prompt_map_dict = {
        str(k): v for k, v in zip(prompt_map["start_frame"].values(), prompt_map["prompt"].values()) if str(k) != ""
    }
    config = {
        "name": name,
        "path": ckpt_config["base_model"][base_model_name],
        "motion_module": ckpt_config["motion_module"][motion_module_name],
        "motion_lora_map": {
            ckpt_config["motion_lora_module"][motion_lora_name]: motion_lora_scale,
        },
        "lcm_map":{
            "enable":use_lcm,
            "start_scale":0.15,
            "end_scale":0.75,
            "gradient_start":0.2,
            "gradient_end":0.75
        },
        "seed": [seed],
        "steps": steps,
        "guidance_scale": guidance_scale,
        "clip_skip": clip_skip,
        "head_prompt": head_prompt,
        "prompt_map": prompt_map_dict,
        "tail_prompt": tail_prompt,
        "n_prompt": [n_prompt],
    }
    config["output"] = {"format": "webp", "fps": fps, "encode_param": {"crf": 10}}
    with open("config.json", "w") as f:
        json.dump(config, f)
    command = f"animatediff generate -c config.json -W {int(width)} -H {int(height)} -L {int(length)} -C {int(context)} -o {SAVE_DIR}"
    shutil.rmtree(SAVE_DIR)
    os.system(command)
    gif_file = glob.glob(SAVE_DIR + "/**/*.webp")[0]
    return gr.Image(gif_file)


# Create the Gradio interface
iface = gr.Interface(
    fn=process_config,
    inputs=[
        gr.Textbox(label="name", value="Animatediff with Prompt Travel"),
        gr.Dropdown(label="base_model_name", choices=ckpt_config["base_model"].keys()),
        gr.Dropdown(label="motion_module_name", choices=ckpt_config["motion_module"].keys()),
        gr.Dropdown(label="motion_lora_name", choices=ckpt_config['motion_lora_module'].keys()),
        gr.Slider(label="motion_lora_scale", minimum=0.0, maximum=1.0, value=0.0),
        gr.Checkbox(label="use_lcm", value=False),
        gr.Number(label="seed", value=42),
        gr.Slider(minimum=1, maximum=50, label="steps", step=1, value=25),
        gr.Slider(minimum=0, maximum=10, label="guidance_scale", value=7.5),
        gr.Number(label="clip_skip", value=2),
        gr.Textbox(label="head_prompt", value="a mountain"),
        gr.Dataframe(
            headers=["start_frame", "prompt"],
            datatype=["number", "str"],
            row_count=1,
            col_count=(2, "fixed"),
            label="prompt_map",
            value=[[0, "((sunny weather))"], [64, "((snow weather))"]],
        ),
        gr.Textbox(label="tail_prompt", value="8k uhd, dslr, soft lighting, high quality"),
        gr.Textbox(
            label="n_prompt",
            value="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated",
        ),
        gr.Number(label="width", minimum=256, maximum=2048, value=512),
        gr.Number(label="height", minimum=256, maximum=2048, value=512),
        gr.Number(label="fps", minimum=4, maximum=24, value=8),
        gr.Number(label="length", minimum=8, maximum=512, value=128),
        gr.Number(label="context", minimum=8, maximum=32, value=16)
    ],
    outputs=gr.Image(),
)

# Run the Gradio app
iface.queue().launch(share=False, debug=True, show_error=True, server_port=10008)