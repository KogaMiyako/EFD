import os

def rename_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith("_SRF_4_HR.png"):
            original_path = os.path.join(folder_path, filename)
            new_filename = filename.replace("_SRF_4_HR.png", ".png")
            new_path = os.path.join(folder_path, new_filename)
            os.rename(original_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
    for filename in os.listdir(folder_path):
        if filename.__contains__("_"):
            original_path = os.path.join(folder_path, filename)
            new_filename = filename.replace("_", "")
            new_path = os.path.join(folder_path, new_filename)
            os.rename(original_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 示例用法：
folder_path = './cpexpres/HR-New/BSD'  # 输入文件夹路径

rename_images_in_folder(folder_path)
