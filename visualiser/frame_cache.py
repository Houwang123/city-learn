from PIL import Image
import numpy as np
import io
import PIL

class FrameCache:

    def __init__(self):
        self.frames = []

    def empty(self):
        self.frames = []

    def append_one_frame(self, one_frame):
        self.frames.append(one_frame)

    def get_total_frame_number(self):
        return len(self.frames)

    def display_frame_at(self, index):
        assert 0 <= index and index < self.get_total_frame_number(), "In display_frame_at, index must be valid"
        self.data = self.frames[index]
        self.img = Image.fromarray(self.data, 'RGB')
        self.img.show()

    def get_image_of_frame_at(self, index):
        assert 0 <= index and index < self.get_total_frame_number(), "In display_frame_at, index must be valid"
        self.data = self.frames[index]
        self.img = Image.fromarray(self.data, 'RGB')
        return self.img

    def get_np_of_frame_at(self, index):
        assert 0 <= index and index < self.get_total_frame_number(), "In display_frame_at, index must be valid"
        self.data = self.frames[index]
        return self.data

    def compress_to_bytes(self, data, fmt = 'png'):
        """
        Helper function to compress image data via PIL/Pillow.
        """
        self.buff = io.BytesIO()
        img = PIL.Image.fromarray(data)    
        img.save(self.buff, format=fmt)
        
        return self.buff.getvalue()

    def get_bytes_of_frame_at(self, index):
        return self.compress_to_bytes(self.get_np_of_frame_at(index))