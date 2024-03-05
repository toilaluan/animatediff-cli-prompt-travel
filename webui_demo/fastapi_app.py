import base64
import json

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

import base64
from base64 import b64encode
from io import BytesIO

import moviepy.editor as mp
from gradio_client import Client
from PIL import Image


def mp4_to_gif(gif_filename, mp4_filename):
    """
    Convert an MP4 file to a GIF file.

    :param gif_filename: The filename for the output GIF file.
    :param mp4_filename: The filename of the MP4 file to convert.
    """
    # Load the MP4 file
    clip = mp.VideoFileClip(mp4_filename)

    # Write the clip as a GIF file
    clip.write_gif(gif_filename)

def base64_to_pil(image_base64: str) -> Image.Image:
    # Decode the base64 string
    image_data = base64.b64decode(image_base64)

    # Convert to a PIL image
    image = Image.open(BytesIO(image_data))

    return image


class ImageData(BaseModel):
    image_base64: str
    prompt: str = ""
    length: float = 1.6
    ratio: str = "square"

@app.post("/animate")
async def upload_image(data: ImageData):
    # Decode the base64 string
    client = Client("http://localhost:4449/", serialize=True)
    image_bytes = base64.b64decode(data.image_base64)
    image = base64_to_pil(data.image_base64)
    image.save("temp/fastapi_image.png")
    if not prompt:
        prompt = client.predict("temp/fastapi_image.png", fn_index=1)
    print(prompt)
    dummy_dict = {"data": {"start_frame": ["0"], "prompt": ["high quality, anime, bright color"]}}
    with open("temp/dummy_dict.json", "w") as f:
        json.dump(dummy_dict, f)
    result = client.predict(
        "temp/fastapi_image.png",
        prompt,	# str in 'head_prompt' Textbox component
        "temp/dummy_dict.json",	# str in 'prompt_map' Textbox component
        "EasyNegativeV2",
        16,
        False,
        0.6,
        data.length,
        "square",
        fn_index=2,
    )

    mp4_file = result

    gif_file = "temp/fastapi_output.gif"
    mp4_to_gif(gif_file, mp4_file)

    with open(gif_file, "rb") as f:
        gif_data = f.read()

    # Step 2: Encode to Base64
    encoded_gif = b64encode(gif_data).decode('utf-8')
    return {"base64_encoded_gif": encoded_gif}
# You can run this app using uvicorn: uvicorn your_file_name:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4448)
