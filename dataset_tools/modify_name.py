import os

def add_prefix_to_files(prefix, directory="/home/yifan/Documents/S8_SAR_Dataset/621_970/VH"):
    """
    给当前目录下的所有文件添加指定前缀。
    
    Args:
    - prefix (str): 要添加到文件名前面的字符串。
    - directory (str): 要修改文件名的目录，默认为当前目录。
    """
    # 获取目录中的所有文件
    for filename in os.listdir(directory):
        # 检查是否是文件（忽略目录）
        if os.path.isfile(os.path.join(directory, filename)):
            # 新文件名：在现有文件名前加上前缀
            new_filename = prefix + filename
            # 重命名文件
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"已将文件 {filename} 重命名为 {new_filename}")

# 示例：在当前目录下的所有文件名前加上 "prefix_"
if __name__ == "__main__":
    # 输入前缀
    prefix = input("请输入要添加的前缀: ")
    # 执行文件名修改操作
    add_prefix_to_files(prefix)