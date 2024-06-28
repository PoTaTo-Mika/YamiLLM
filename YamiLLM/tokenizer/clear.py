import os

file = "data.txt"

# 读取文件内容并删除空行
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 去除空行
lines = [line for line in lines if line.strip() != '']

# 将处理后的内容写回文件
with open(file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

    