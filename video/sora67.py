import config
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


class VideoGenerator:
    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    def __init__(self):
        # VAE
        self._vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.float32
        )

        # Model
        self._pipe = WanPipeline.from_pretrained(
            self.MODEL_ID,
            vae=self._vae,
            torch_dtype=torch.float16
        )

        # Не загружаем все компоненты одновременно в VRAM
        self._pipe.enable_model_cpu_offload()

    def generate_and_save(self, prompt, negative_prompt, height=480, width=640, num_frames=81, num_inference_steps=30, guidance_scale=6.0):
        torch.cuda.empty_cache()
        print("Starting generation...")

        with torch.inference_mode():
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).frames[0]

        export_to_video(
            output,
            "result.mp4",
            fps=16
        )

        print("Video saved!")