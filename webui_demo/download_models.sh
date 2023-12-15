mkdir webui_demo/models
# basemodel
curl -L "https://civitai.com/api/download/models/125771" --output "data/toonyou.safetensors"
curl -L "https://civitai.com/api/download/models/102222" --output "data/XXMix_9realistic.safetensors"
curl -L "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16" --output "data/realistic_vision.safetensors"
# motion & lora
curl -L "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt?download=true" --output "data/mm_sd_v15_v2.ckpt"
curl -L "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt?download=true" --output "data/v2_lora_PanLeft.ckpt"
