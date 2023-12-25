import glob
import json
import random
import uuid
from urllib import parse, request


def random_uuid():
    return str(uuid.uuid4())

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)

def call_comfy(image_path, prompt, denoise_strength=0.6):
    prompt_workflow = json.load(open("webui_demo/workflow_api.json"))
    file_name_node = prompt_workflow['506']
    input_prompt_node = prompt_workflow['512']
    input_image_node = prompt_workflow['492']
    uuid = random_uuid()
    file_name_node["inputs"]["filename_prefix"] = uuid
    input_prompt_node["inputs"]["text"] = prompt
    prompt_workflow['321']['inputs']['text'] = prompt
    input_image_node["inputs"]["image"] = image_path
    prompt_workflow['500']['inputs']['number'] = 16
    prompt_workflow['528']['inputs']['denoise'] = denoise_strength
    queue_prompt(prompt_workflow)
    return f"/home/ai/ComfyUI/output/{uuid}"

if __name__ =="__main__":
    # prompt_workflow = json.load(open("webui_demo/prompt_workflow.json"))
    call_comfy(
        "/home/ai/luantt/animatediff-cli-prompt-travel/temp/conditional_image_comfyui.png",
        "1girl,flower,school uniform,solo,female,uniform,hair ornament,clothing"
    )
