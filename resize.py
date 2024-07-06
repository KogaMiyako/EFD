import os
from PIL import Image

def compare_and_resize_images(folder_a, folder_b):
    if not os.path.exists(folder_a) or not os.path.exists(folder_b):
        print("One or both folders do not exist.")
        return

    for filename_a in os.listdir(folder_a):
        if filename_a.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
            filename_b = os.path.join(folder_b, filename_a)

            if os.path.exists(filename_b):
                image_a = Image.open(os.path.join(folder_a, filename_a))
                image_b = Image.open(filename_b)

                if image_a.size != image_b.size:
                    image_a_resized = image_a.resize(image_b.size, Image.ANTIALIAS)
                    image_a_resized.save(os.path.join(folder_a, filename_a))
                    print(f"Resized {filename_a}({image_a.size}) to match {filename_b}({image_b.size})'s size.")

# 示例用法：
folder_a = './cpexpres/Real-ESRGAN-Res/manga109'  # 文件夹路径a
folder_b = './cpexpres/HR/manga109'  # 文件夹路径b

compare_and_resize_images(folder_a, folder_b)
