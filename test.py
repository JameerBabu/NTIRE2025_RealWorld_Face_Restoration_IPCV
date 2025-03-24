# -*- coding: utf-8 -*-
from huggingface_hub import login

# Login with your Hugging Face token
login('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')

import os
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image

# Set up paths for the folder containing images
input_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/restoration_dataset/test/CelebA"
output_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/gfpgan/CelebA"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the FluxControlNet model and pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Iterate over all image files in the folder
for image_file in os.listdir(input_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Construct the full path to the image
        image_path = os.path.join(input_folder, image_file)

        # Load the control image
        control_image = load_image(image_path)

        # Resize the control image (upscale x4)
        w, h = control_image.size
        control_image = control_image.resize((w * 1, h * 1))

        # Process the image with the model
        result_image = pipe(
            prompt="",
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        # Save the processed image with the same name in the output folder
        output_image_path = os.path.join(output_folder, image_file)
        result_image.save(output_image_path)

        print(f"Processed and saved image: {output_image_path}")

print("Image processing completed for all images in the folder.")

input_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/restoration_dataset/test/CelebChild-Test"
output_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/Flux1/CelebChild-Test"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


for image_file in os.listdir(input_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Construct the full path to the image
        image_path = os.path.join(input_folder, image_file)

        # Load the control image
        control_image = load_image(image_path)

        # Resize the control image (upscale x4)
        w, h = control_image.size
        control_image = control_image.resize((w * 1, h * 1))

        # Process the image with the model
        result_image = pipe(
            prompt="",
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        # Save the processed image with the same name in the output folder
        output_image_path = os.path.join(output_folder, image_file)
        result_image.save(output_image_path)

        print(f"Processed and saved image: {output_image_path}")

print("Image processing completed for all images in the folder.")

input_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/restoration_dataset/test/LFW-Test"
output_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/Flux1/LFW-Test"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


for image_file in os.listdir(input_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Construct the full path to the image
        image_path = os.path.join(input_folder, image_file)

        # Load the control image
        control_image = load_image(image_path)

        # Resize the control image (upscale x4)
        w, h = control_image.size
        control_image = control_image.resize((w * 1, h * 1))

        # Process the image with the model
        result_image = pipe(
            prompt="",
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        # Save the processed image with the same name in the output folder
        output_image_path = os.path.join(output_folder, image_file)
        result_image.save(output_image_path)

        print(f"Processed and saved image: {output_image_path}")

print("Image processing completed for all images in the folder.")

input_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/restoration_dataset/test/WebPhoto-Test"
output_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/Flux1/WebPhoto-Test"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


for image_file in os.listdir(input_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Construct the full path to the image
        image_path = os.path.join(input_folder, image_file)

        # Load the control image
        control_image = load_image(image_path)

        # Resize the control image (upscale x4)
        w, h = control_image.size
        control_image = control_image.resize((w * 1, h * 1))

        # Process the image with the model
        result_image = pipe(
            prompt="",
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        # Save the processed image with the same name in the output folder
        output_image_path = os.path.join(output_folder, image_file)
        result_image.save(output_image_path)

        print(f"Processed and saved image: {output_image_path}")

print("Image processing completed for all images in the folder.")

input_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/restoration_dataset/test/Wider-Test"
output_folder = "/content/drive/MyDrive/NTIRE/Test/restoration_real_world/Flux1/Wider-Test"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


for image_file in os.listdir(input_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Construct the full path to the image
        image_path = os.path.join(input_folder, image_file)

        # Load the control image
        control_image = load_image(image_path)

        # Resize the control image (upscale x4)
        w, h = control_image.size
        control_image = control_image.resize((w * 1, h * 1))

        # Process the image with the model
        result_image = pipe(
            prompt="",
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        # Save the processed image with the same name in the output folder
        output_image_path = os.path.join(output_folder, image_file)
        result_image.save(output_image_path)

        print(f"Processed and saved image: {output_image_path}")

print("Image processing completed for all images in the folder.")

import os
import shutil
import zipfile

# Define source folders and destination
source_folders = ["folder1", "folder2", "folder3"]  # Update with actual folder names
destination = "merged_folder"
zip_filename = "images.zip"

# Create destination folder if not exists
os.makedirs(destination, exist_ok=True)

# Copy files from source folders
for folder in source_folders:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, destination)

# Zip the files without keeping the folder structure
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(destination):
        zipf.write(os.path.join(destination, file), arcname=file)

print(f"Created {zip_filename}, ready for download.")