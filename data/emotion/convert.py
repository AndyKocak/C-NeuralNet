from PIL import Image
import os

#small script to convert the png fer data into .raw for easier reading in c file
def convert_to_raw(input_dir, output_dir, size=(48, 48)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg')):
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('L').resize(size)  # Convert to grayscale and resize
            raw_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.raw')
            img.save(raw_path, format='RAW')

