# Copyright 2023 Natural Synthetics Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union
from io import BytesIO
import PIL
from PIL import ImageSequence, Image
import requests
import os
import numpy as np
import imageio


def get_image(img_path) -> PIL.Image.Image:
    if img_path.startswith("http"):
        return PIL.Image.open(requests.get(img_path, stream=True).raw)
    if os.path.exists(img_path):
        return Image.open(img_path)
    raise Exception("File not found")

def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        # Save the first image
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)  # 0 means the GIF will loop indefinitely

        # Get the byte array from the buffer
        gif_bytes = output_buffer.getvalue()

    return gif_bytes

def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))

def images_to_mp4_bytes(images: List[Image.Image], duration: int = 1000) -> bytes:
        with BytesIO() as output_buffer:
            with imageio.get_writer(output_buffer, format='mp4', fps=1/(duration/1000)) as writer:
                for img in images:
                    writer.append_data(np.array(img))
            mp4_bytes = output_buffer.getvalue()

        return mp4_bytes

def save_as_mp4(images: List[Image.Image], file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_mp4_bytes(images, duration))

def scale_aspect_fill(img, new_width, new_height):
    new_width = int(new_width)
    new_height = int(new_height)

    original_width, original_height = img.size
    ratio_w = float(new_width) / original_width
    ratio_h = float(new_height) / original_height

    if ratio_w > ratio_h:
        # It must be fixed by width
        resize_width = new_width
        resize_height = round(original_height * ratio_w)
    else:
        # Fixed by height
        resize_width = round(original_width * ratio_h)
        resize_height = new_height

    img_resized = img.resize((resize_width, resize_height), Image.LANCZOS)

    # Calculate cropping boundaries and do crop
    left = (resize_width - new_width) / 2
    top = (resize_height - new_height) / 2
    right = (resize_width + new_width) / 2
    bottom = (resize_height + new_height) / 2

    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped

def extract_gif_frames_from_midpoint(image: Union[str, PIL.Image.Image], fps: int=8, target_duration: int=1000) -> list:
    # Load the GIF
    image = get_image(image) if type(image) is str else image

    frames = []

    estimated_frame_time = None

    # some gifs contain the duration - others don't
    # so if there is a duration we will grab it otherwise we will fall back

    for frame in ImageSequence.Iterator(image):

        frames.append(frame.copy())
        if 'duration' in frame.info:
            frame_info_duration = frame.info['duration']
            if frame_info_duration > 0:
                estimated_frame_time = frame_info_duration

    if estimated_frame_time is None:
        if len(frames) <= 16:
            # assume it's 8fps
            estimated_frame_time = 1000 // 8
        else:
            # assume it's 15 fps
            estimated_frame_time = 70

    if len(frames) < fps:
        raise ValueError(f"fps of {fps} is too small for this gif as it only has {len(frames)} frames.")

    skip = len(frames) // fps
    upper_bound_index = len(frames) - 1

    best_indices = [x for x in range(0, len(frames), skip)][:fps]
    offset = int(upper_bound_index - best_indices[-1]) // 2
    best_indices = [x + offset for x in best_indices]
    best_duration = (best_indices[-1] - best_indices[0]) * estimated_frame_time

    while True:

        skip -= 1

        if skip == 0:
            break

        indices = [x for x in range(0, len(frames), skip)][:fps]

        # center the indices, so we sample the middle of the gif...
        offset = int(upper_bound_index - indices[-1]) // 2
        if offset == 0:
            # can't shift
            break
        indices = [x + offset for x in indices]

        # is the new duration closer to the target than last guess?
        duration = (indices[-1] - indices[0]) * estimated_frame_time
        if abs(duration - target_duration) > abs(best_duration - target_duration):
            break

        best_indices = indices
        best_duration = duration

    return [frames[index] for index in best_indices]

def get_crop_coordinates(old_size: tuple, new_size: tuple) -> tuple:
    """
    Calculate the crop coordinates after scaling an image to fit a new size.

    :param old_size: tuple of the form (width, height) representing the original size of the image.
    :param new_size: tuple of the form (width, height) representing the desired size after scaling.
    :return: tuple of the form (left, upper, right, lower) representing the normalized crop coordinates.
    """
    # Check if the input tuples have the right form (width, height)
    if not (isinstance(old_size, tuple) and isinstance(new_size, tuple) and
            len(old_size) == 2 and len(new_size) == 2):
        raise ValueError("old_size and new_size should be tuples of the form (width, height)")

    # Extract the width and height from the old and new sizes
    old_width, old_height = old_size
    new_width, new_height = new_size

    # Calculate the ratios for width and height
    ratio_w = float(new_width) / old_width
    ratio_h = float(new_height) / old_height

    # Determine which dimension is fixed (width or height)
    if ratio_w > ratio_h:
        # It must be fixed by width
        resize_width = new_width
        resize_height = round(old_height * ratio_w)
    else:
        # Fixed by height
        resize_width = round(old_width * ratio_h)
        resize_height = new_height

    # Calculate cropping boundaries in the resized image space
    left = (resize_width - new_width) / 2
    upper = (resize_height - new_height) / 2
    right = (resize_width + new_width) / 2
    lower = (resize_height + new_height) / 2

    # Normalize the cropping coordinates

    # Return the normalized coordinates as a tuple
    return (left, upper, right, lower)

aspect_ratio_to_1024_map = {
    "0.42": [640,  1536],
    "0.57": [768,  1344],
    "0.68": [832,  1216],
    "1.00": [1024, 1024],
    "1.46": [1216,  832],
    "1.75": [1344,  768],
    "2.40": [1536,  640]
}

res_to_aspect_map = {
    1024: aspect_ratio_to_1024_map,
    512: {key: [value[0] // 2, value[1] // 2] for key, value in aspect_ratio_to_1024_map.items()},
}

def best_aspect_ratio(aspect_ratio: float, resolution: int):

    map = res_to_aspect_map[resolution]

    d = 99999999
    res = None
    for key, value in map.items():
        ar = value[0] / value[1]
        diff = abs(aspect_ratio - ar)
        if diff < d:
            d = diff
            res = value

    ar = res[0] / res[1]
    return f"{ar:.2f}", res