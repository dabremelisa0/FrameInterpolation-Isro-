from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import requests
from PIL import Image
from io import BytesIO
import os
import cv2
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from typing import List
from moviepy.editor import ImageSequenceClip
from fastapi import Response
from fastapi.responses import FileResponse
import tensorflow_hub as hub
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("cloud_name"),
    api_key=os.getenv("api_key"),
    api_secret=os.getenv("api_secret")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, limit this to your known domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Directory to save the images and videos
IMAGE_SAVE_PATH = "images"
VIDEO_SAVE_PATH = "videos"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)

# Fixed layer prefix
LAYER_PREFIX = "MOSDAC_TIR_1"

# Load FILM model
print("Loading FILM model...")
local_model_path = r"C:\Project\SIH\film_model"
film_model = hub.load(local_model_path)
print("FILM model loaded.")

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def fetch_wms_image(
    geoserver_url: str,
    timestamp: str,
    bbox: str,
    width: int = 512,
    height: int = 512,
    image_format: str = "image/png"
) -> str:
    """Fetches a WMS image from the GeoServer URL using the provided parameters."""
    layer_name = f"{LAYER_PREFIX}:3RIMG_01DEC2024_{timestamp}_L1B_STD_V01R00_IMG_TIR1"
    min_x, min_y, max_x, max_y = map(lambda v: int(round(float(v))), bbox.split(","))


    wms_params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": layer_name,
        "bbox": f"{min_x},{min_y},{max_x},{max_y}",
        "width": width,
        "height": height,
        "srs": "EPSG:4326",
        "format": image_format,
    }
    try:
        response = requests.get(geoserver_url, params=wms_params, stream=True)
        response.raise_for_status()
        image_name = f"wms_image_{timestamp}_{min_x}_{min_y}_{max_x}_{max_y}.png"
        image_path = os.path.join(IMAGE_SAVE_PATH, image_name)
        image = Image.open(BytesIO(response.content))
        image.save(image_path)
        return image_path
    except Exception as e:
        return None

def load_image(image_path: str, target_size=(512,512)) -> np.ndarray:
    """Loads an image, resizes it to the target size, and normalizes pixels to the range [0, 1]."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image / _UINT8_MAX_F  # Normalize pixels to [0, 1]
    return image
def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Sharpens an image using a kernel filter.

    Parameters:
    - image (np.ndarray): Input image.

    Returns:
    - np.ndarray: Sharpened image.
    """
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened
# def interpolate_frames(image1: np.ndarray, image2: np.ndarray, times: list = [0.33, 0.66]) -> list:
#     """Interpolates intermediate frames between two images using the FILM model."""
#     interpolated_frames = []
#     for t in times:
#         time = np.array([t], dtype=np.float32)  # Correct dtype
#         input_data = {
#             'time': tf.convert_to_tensor(np.expand_dims(time, axis=0), dtype=tf.float32),  # Add batch dimension
#             'x0': tf.convert_to_tensor(np.expand_dims(image1, axis=0), dtype=tf.float32),  # Add batch dimension
#             'x1': tf.convert_to_tensor(np.expand_dims(image2, axis=0), dtype=tf.float32),  # Add batch dimension
#         }
#         result = film_model(input_data)
#         interpolated_frame = (result['image'][0].numpy() * _UINT8_MAX_F).astype(np.uint8)
#         interpolated_frames.append(interpolated_frame)
#     return interpolated_frames
def interpolate_frames(image1: np.ndarray, image2: np.ndarray, num_frames: int = 25) -> list:
    """
    Interpolates intermediate frames between two images using the FILM model.

    Parameters:
    - image1 (np.ndarray): First input image.
    - image2 (np.ndarray): Second input image.
    - num_frames (int): Number of intermediate frames to generate.

    Returns:
    - list: List of interpolated frames.
    """
    interpolated_frames = []
    times = np.linspace(0, 1, num_frames + 2)[1:-1]  # Generate evenly spaced times excluding 0 and 1
    for t in times:
        time = np.array([t], dtype=np.float32)  # Correct dtype
        input_data = {
            'time': tf.convert_to_tensor(np.expand_dims(time, axis=0), dtype=tf.float32),  # Add batch dimension
            'x0': tf.convert_to_tensor(np.expand_dims(image1, axis=0), dtype=tf.float32),  # Add batch dimension
            'x1': tf.convert_to_tensor(np.expand_dims(image2, axis=0), dtype=tf.float32),  # Add batch dimension
        }
        result = film_model(input_data)  # Assuming `film_model` is the pre-loaded FILM model
        interpolated_frame = (result['image'][0].numpy() * _UINT8_MAX_F).astype(np.uint8)
        interpolated_frames.append(interpolated_frame)
    return interpolated_frames


# def generate_video(image_paths: list, video_name: str, frame_rate: int = 10) -> str:
#     if not image_paths:
#         raise ValueError("No images provided for video generation.")
    
#     # Load the first image using load_image to ensure consistent sizing
#     first_frame = load_image(image_paths[0])  # This will be 256x256 by default
#     height, width, _ = first_frame.shape
#     video_path = os.path.join(VIDEO_SAVE_PATH, video_name)
#     video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    
#     for i in range(len(image_paths) - 1):
#         frame1 = load_image(image_paths[i])   # (256,256)
#         frame2 = load_image(image_paths[i+1]) # (256,256)

#         # Write the first frame
#         video_writer.write(cv2.cvtColor((frame1 * _UINT8_MAX_F).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
#         # Generate interpolated frames
#         interpolated_frames = interpolate_frames(frame1, frame2)
#         for interpolated_frame in interpolated_frames:
#             video_writer.write(cv2.cvtColor(interpolated_frame, cv2.COLOR_RGB2BGR))
    
#     # Write the last frame
#     last_frame = load_image(image_paths[-1])
#     video_writer.write(cv2.cvtColor((last_frame * _UINT8_MAX_F).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
#     video_writer.release()
#     return video_path

def generate_video(image_paths: List[str], video_name: str, frame_rate: int = 10) -> str:
    """
    Generates a video from a list of image paths using MoviePy.

    Args:
        image_paths (List[str]): List of paths to input images.
        video_name (str): Name of the output video file (e.g., 'output.mp4').
        frame_rate (int, optional): Frames per second for the video. Defaults to 10.

    Returns:
        str: Path to the generated video file.
    """
    if not image_paths:
        raise ValueError("No images provided for video generation.")

    frames = []

    for i in range(len(image_paths) - 1):
        # Load consecutive frames
        frame1 = load_image(image_paths[i])       # Expected shape: (H, W, 3), range: [0, 1]
        frame2 = load_image(image_paths[i + 1])

        # Append the first frame (converted to [0, 255] uint8)
        frame1_uint8 = (frame1).astype(np.uint8)
        frames.append(frame1_uint8)

        # Generate and append interpolated frames
        interpolated_frames = interpolate_frames(frame1, frame2)
        for interp_frame in interpolated_frames:
            interp_frame_uint8 = (interp_frame ).astype(np.uint8)
            frames.append(interp_frame_uint8)

    # Append the last frame
    last_frame = load_image(image_paths[-1])
    last_frame_uint8 = (last_frame).astype(np.uint8)
    frames.append(last_frame_uint8)

    # Create the video clip
    clip = ImageSequenceClip(frames, fps=frame_rate)

    # Define the output video path
    video_path = os.path.join(VIDEO_SAVE_PATH, video_name)

    # Write the video file using MoviePy's write_videofile method
    # You can adjust codec and other parameters as needed
    clip.write_videofile(video_path, codec='libx264', audio=False)

    return video_path

@app.get("/generate_video/")
async def generate_video_from_timestamps(
    geoserver_url: str = Query(..., description="GeoServer WMS service URL"),
    bbox: str = Query(..., description="Bounding box in 'minX,minY,maxX,maxY' format"),
    start_time: str = Query("0045", description="Start timestamp (e.g., '0045')"),
    end_time: str = Query("2115", description="End timestamp (e.g., '2115')"),
    interval: int = Query(30, description="Time interval in minutes"),
    width: int = Query(512, description="Width of the output images"),
    height: int = Query(512, description="Height of the output images"),
    frame_rate: int = Query(10, description="Frame rate for the video")
):
    """Generates a video from WMS images fetched over a range of timestamps."""
    try:
        # Generate timestamps
        time_format = "%H%M"
        current_time = datetime.strptime(start_time, time_format)
        end_time_dt = datetime.strptime(end_time, time_format)
        timestamps = []
        print(geoserver_url)
        print(type(start_time))
        print(type(end_time))
        print(type(width))
        print(type(height))
        print(type(frame_rate))


        while current_time <= end_time_dt:
            timestamps.append(current_time.strftime(time_format))
            current_time += timedelta(minutes=interval)

        # Fetch images, skipping missing timestamps
        image_paths = []
        for timestamp in timestamps:
            image_path = fetch_wms_image(
                geoserver_url=geoserver_url,
                timestamp=timestamp,
                bbox=bbox,
                width=width,
                height=height
            )
            if image_path:
                image_paths.append(image_path)
            else:
                print(f"Skipped missing image for timestamp: {timestamp}")

        # Ensure at least one image is available
        if not image_paths:
            return {"error": "No valid images were fetched for the specified range."}

        # Generate video
        video_name = f"wms_video_{start_time}_{end_time}.mp4"

        video_path = generate_video(image_paths, video_name, frame_rate=frame_rate)
        print(video_path)
        
        upload_response = cloudinary.uploader.upload(video_path, resource_type="video")
        print(upload_response)
        video_url = upload_response.get("secure_url")
        if video_url:
            print('cloudinary Link')
            return JSONResponse({"video_url": video_url})


        #return FileResponse(video_path, media_type="video/mp4", filename=video_name)
        

    except Exception as e:
        return {"error": str(e)}