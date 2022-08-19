import numpy as np
import cv2
from random import randint

duration = 10
fps = 25
out = cv2.VideoWriter('output t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1440, 720))
for _ in range(fps * duration):
    data = np.ones([720, 1440, 3], dtype='uint8') #np.random.randint(0, 256, (720, 1440, 3), dtype='uint8')
    data[:,:,0] = randint(0, 256)
    data[:,:,1] = randint(0, 256)
    data[:,:,2] = randint(0, 256)

    out.write(data)
out.release()