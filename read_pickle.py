import pickle

# 定义你的.p文件路径
# file_path = '/home/yrf/MPT/data/desert/map.p'  
file_path = '/home/yrf/MPT/data/sim_dataset/train/env000010/path_0.p'  
# file_path = 'your_data.p'  
# 请将 'your_data.p' 替换为你的实际文件路径# 打开并读取文件

with open(file_path, 'rb') as f:
    loaded_data = pickle.load(f)

# 打印读取到的数据
print(loaded_data)