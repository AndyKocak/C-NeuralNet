from PIL import Image
import os

import numpy as np

#small script to convert the png fer data into .raw for easier reading in c file
def convert_to_raw(input_dir, output_dir, size=(48, 48)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg')):
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('L').resize(size)  # Convert to grayscale and resize
            raw_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.raw')
            img_arr = np.array(img)
            raw_data = img_arr.flatten()
            raw_bytes = raw_data.tobytes()
            with open(raw_path, 'wb') as f:
                f.write(raw_bytes)

#convert_to_raw("test/angry", "test/angry_raw")
#convert_to_raw("test/disgusted", "test/disgusted_raw")
#convert_to_raw("test/fearful", "test/fearful_raw")
#convert_to_raw("test/happy", "test/happy_raw")
#convert_to_raw("test/neutral", "test/neutral_raw")
#convert_to_raw("test/sad", "test/sad_raw")
#convert_to_raw("test/surprised", "test/surprised_raw")

# Example loading from a file
with open("test/sad_raw/im0.raw", "rb") as f:
   raw_data = f.read()
   img = Image.frombytes('L', (48, 48), raw_data)
   img.show()
