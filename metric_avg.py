import pandas as pd

# 读取CSV文件
file_path = 'BART_Metrics.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 计算每一列的平均值
average_values = df.mean()
average_values = round(average_values, 4)

# 打印或使用平均值
print(average_values)