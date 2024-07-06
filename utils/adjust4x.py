import os
from PIL import Image

def adjust_png_image_sizes(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                image_path = os.path.join(foldername, filename)
                try:
                    img = Image.open(image_path)
                    original_width, original_height = img.size
                    new_width = original_width
                    new_height = original_height

                    # 调整图像宽度为最接近4的倍数
                    new_width = (new_width + 3) & ~3

                    # 调整图像高度为最接近4的倍数
                    new_height = (new_height + 3) & ~3

                    if (original_width != new_width) or (original_height != new_height):
                        img = img.resize((new_width, new_height), Image.BILINEAR)
                        img.save(image_path)
                        print(f"Resized {filename} to {new_width}x{new_height}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

# 示例用法：
root_folder = './dataset'  # 根文件夹路径

adjust_png_image_sizes(root_folder)
