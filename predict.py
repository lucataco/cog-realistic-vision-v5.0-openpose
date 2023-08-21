# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, ControlNetModel, UniPCMultistepScheduler
import tempfile

MODEL_NAME = "SG161222/Realistic_Vision_V5.0_noVAE"
CONTROL_NAME = "lllyasviel/ControlNet"
OPENPOSE_NAME = "lllyasviel/control_v11p_sd15_openpose"
MODEL_CACHE = "cache"
CONTROL_CACHE = "control-cache"
POSE_CACHE = "pose-cache"
url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

class Predictor(BasePredictor):
    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.openpose = OpenposeDetector.from_pretrained(
            CONTROL_NAME,
            cache_dir=CONTROL_CACHE,
        )
        controlnet = ControlNetModel.from_pretrained(
            POSE_CACHE,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")
        

    def predict(
        self,
        image: Path = Input(description="Input pose image"),
        prompt: str = "RAW photo, a portrait photo of a latina woman in casual clothes, 8k uhd, high quality, film grain, Fujifilm XT3",
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        steps: int = Input(description=" num_inference_steps", ge=0, le=100, default=30),
        guidance: float = Input(description="Guidance scale (3.5 - 7)", default=5),
        scheduler: str = Input(
            default="MultistepDPM-Solver",
            choices=["EulerA", "MultistepDPM-Solver"],
            description="Choose a scheduler",
        ),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=728),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed is None) or (seed == 0):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        print("Seed is:"+str(seed))
        generator = torch.Generator('cuda').manual_seed(seed)
    
        # Control pose image
        image = Image.open(image)
        control_image = self.openpose(image, hand_and_face=True)

        width = self.base(width)
        height = self.base(height)

        if scheduler == "EulerA":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        elif scheduler == "MultistepDPM-Solver":
            self.pipe.scheduler = DPMSolverMultistepScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
    
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
            image=control_image
        ).images[0]

        output_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(output_path)

        return  Path(output_path)

