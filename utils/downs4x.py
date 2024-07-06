from PIL import Image
import os

def resize_images_in_folder(input_folder, output_folder, scale_factor=0.25):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for foldername, subfolders, filenames in os.walk(input_folder):
        for filename in filenames:
            input_path = os.path.join(foldername, filename)
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)

            try:
                img = Image.open(input_path)
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                resized_img = img.resize((new_width, new_height), Image.BILINEAR)
                output_dir = os.path.dirname(output_path)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                resized_img.save(output_path)
                print(f"Resized and saved {filename} to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# 示例用法：
input_folder = 'dataset/HR/lsun_test_100'  # 文件夹A路径
output_folder = 'dataset_LR/HR/lsun_test_100'  # 文件夹B路径

resize_images_in_folder(input_folder, output_folder)
