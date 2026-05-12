import os
import shutil
from pathlib import Path

from config import MAIN_FOLDER, DATASET_FOLDER, SCREENSHOTS_FOLDER

def format_data(img_path, sides, poses, k):
    for i in range(2):
        target_dir = DATASET_FOLDER / sides[i]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, str(target_dir / f'img{k}.jpg'))
        with open(target_dir / f'img{k}.txt', 'w') as f:
            f.write(poses[i])

# print(os.listdir(SCREENSHOTS_FOLDER))
k = 0
for folder in os.listdir(SCREENSHOTS_FOLDER):
    folder_path = SCREENSHOTS_FOLDER / folder
    if not folder_path.is_dir():
        continue
    images = [str(folder_path / file) for file in os.listdir(folder_path) if file.endswith('.jpg')]
    # print(images)
    for img_path in images:
        label_path = os.path.splitext(img_path)[0] + '.txt'
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                continue
            first_pos, second_pos = lines[0], lines[1]

        if folder.startswith('sin'): # b/c
            format_data(img_path, ('b', 'c'), (first_pos, second_pos), k)

        elif folder.startswith('cos'): # a/c
            format_data(img_path, ('c', 'a'), (first_pos, second_pos), k)

        elif folder.startswith('tan'): # b/a
            format_data(img_path, ('a', 'b'), (first_pos, second_pos), k)
        
        k += 1