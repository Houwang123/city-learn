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

def image_to_byte_array(image: Image) -> bytes:
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def get_bytes_of_frame_at(index):
    return image_to_byte_array(get_image_of_frame_at(index))