from PIL import Image
import os


#function to resize my images
def resize_images(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_dir, filename)
            with Image.open(image_path) as img:
                resized_img = img.resize(target_size)
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)

input_directory = r'C:\Users\victo\OneDrive\Desktop\whiskard\cat data set\beans'
output_directory = r'C:\Users\victo\OneDrive\Desktop\whiskard\cat data set\beans'
target_size = (299,299) 

resize_images(input_directory, output_directory, target_size)