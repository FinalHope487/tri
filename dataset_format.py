import os
import shutil
from pathlib import Path

tri_dir = r'C:\Users\sword\.vscode\vtb\my-projects\tri'

def format_data(sides, poses, k):
    for i in range(2):
        shutil.copy2(img_path, str(Path(tri_dir)/f'dataset/{sides[i]}/img{k}.jpg'))
        with open(f'{tri_dir}/dataset/{sides[i]}/img{k}.txt', 'w') as f:
            f.write(poses[i])

# print(os.listdir(Path(tri_dir)/'screenshots'))
k = 0
for folder in os.listdir(Path(tri_dir)/'screenshots'):
    images = [f'{tri_dir}/screenshots/{folder}/{file}' for file in os.listdir(f'{tri_dir}/screenshots/{folder}') if file.endswith('.jpg')]
    # print(images)
    for img_path in images:
        label_path = os.path.splitext(img_path)[0] + '.txt'
        with open(label_path, 'r') as f:
            first_pos, second_pos = f.readlines()

        if folder.startswith('sin'): # b/c
            format_data(('b', 'c'), (first_pos, second_pos), k)

        elif folder.startswith('cos'): # a/c
            format_data(('c', 'a'), (first_pos, second_pos), k)

        elif folder.startswith('tan'): # b/a
            format_data(('a', 'b'), (first_pos, second_pos), k)
        
        k += 1