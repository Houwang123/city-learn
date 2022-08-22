from PIL import Image
import numpy as np

frames = []

def append_one_frame(one_frame):
    frames.append(one_frame)

def get_total_frame_number():
    return len(frames)

def display_frame_at(index):
    assert 0 <= index and index < get_total_frame_number(), "In display_frame_at, index must be valid"
    data = frames[index]
    img = Image.fromarray(data, 'RGB')
    img.show()

def get_image_of_frame_at(index):
    assert 0 <= index and index < get_total_frame_number(), "In display_frame_at, index must be valid"
    data = frames[index]
    img = Image.fromarray(data, 'RGB')
    return img

def get_np_of_frame_at(index):
    assert 0 <= index and index < get_total_frame_number(), "In display_frame_at, index must be valid"
    data = frames[index]
    return data

import io
import PIL

def compress_to_bytes(data, fmt = 'png'):
    """
    Helper function to compress image data via PIL/Pillow.
    """
    buff = io.BytesIO()
    img = PIL.Image.fromarray(data)    
    img.save(buff, format=fmt)
    
    return buff.getvalue()

def get_bytes_of_frame_at(index):
    return compress_to_bytes(get_np_of_frame_at(index))