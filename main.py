import os
import torch
import time
import onnx
import onnxruntime as ort
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorrt as trt
from trt_unet_wrapper import TRTUNetWrapper

# --- INSTALLATION REQUIREMENTS ---
# Run the following commands in your environment:
# pip install torch torchvision diffusers transformers onnx onnxruntime numpy pillow accelerate
# git clone https://github.com/NVIDIA/TensorRT-LLM.git
# cd TensorRT-LLM
# pip install -r requirements.txt
# Follow additional build steps from TensorRT-LLM README to enable quantization

# --- SETTINGS ---
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DEVICE = "cuda"
NUM_IMAGES = 10
PROMPT = "a futuristic city at sunset in the style of cyberpunk"
ONNX_PATH = "unet.onnx"
ENGINE_PATH = "unet_fp4_svd.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --- 1. Load model and export UNet to ONNX ---
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
unet = pipe.unet
unet.eval()

dummy_input = torch.randn(1, 4, 64, 64).half().to(DEVICE)
with torch.no_grad():
    torch.onnx.export(unet, dummy_input, ONNX_PATH, input_names=["latent"], output_names=["output"],
                      dynamic_axes={"latent": {0: "batch_size"}}, opset_version=17)


# --- 2. Run SVD + FP4 quantization using TensorRT-LLM ---
# This will output a serialized TensorRT-LLM engine (.engine)

# Example command:
# python TensorRT-LLM/examples/quantization/quantize_model.py \\
#   --model_path ./unet.onnx \\
#   --quant_mode fp4 \\
#   --use_svd \\
#   --calib_data ./calib_data/ \\
#   --output_path ./unet_fp4_svd/



# --- 3. Run inference using TensorRT-LLM runtime ---
from tensorrt_llm.runtime import Engine as TRTLLMEngine

def profile_engine_llm(engine_path, input_tensor):
    engine = TRTLLMEngine.from_file(engine_path)
    inputs = {"latent": input_tensor.half().cuda()}
    outputs = engine.run(inputs)
    print("Inference completed with TensorRT-LLM.")
    return outputs["output"]


input_tensor = torch.randn(1, 4, 64, 64).half().to(DEVICE)
output = profile_engine_llm(ENGINE_PATH, input_tensor)


# --- 4. Generate images and compute CLIP Score ---
def generate_images(pipe, prompt, num_images):
    return [pipe(prompt, num_inference_steps=25).images[0] for _ in range(num_images)]

def compute_clip_score(images, prompt):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits_per_image.softmax(dim=1).mean().item()

# Original pipeline score
print("Generating images from original pipeline...")
images_orig = generate_images(pipe, PROMPT, NUM_IMAGES)
score_orig = compute_clip_score(images_orig, PROMPT)
print(f"Original CLIP Score: {score_orig:.4f}")

print("Generating images from quantized UNet...")

pipe_quant = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
pipe_quant.unet = TRTUNetWrapper(ENGINE_PATH)

images_quant = generate_images(pipe_quant, PROMPT, NUM_IMAGES)
score_quant = compute_clip_score(images_quant, PROMPT)
print(f"Quantized CLIP Score: {score_quant:.4f}")
