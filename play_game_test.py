import time
import os
import random
import multiprocessing as  mp
import torch
from torchvision import transforms
from PIL import Image
import mss
import numpy as np
import cv2
import pydirectinput as pyd
from screeninfo import get_monitors
import keyboard
from PIL import ImageDraw

from model import create_player, create_classifier

# sin: c->b
# cos: c->a
# tan: a->b

from config import MAIN_FOLDER, get_best_model_path

MODE = 'tan'
sides_dict = {'cos': ('c', 'a'), 'sin': ('c', 'b'), 'tan': ('a', 'b')}
modes_list = ['cos', 'sin', 'tan']

mode_model_path = get_best_model_path('classifier')
pos_models_path = {side: get_best_model_path(side) for side in ['a', 'b', 'c']}
monitor_number = 0
SAVE_DIR = MAIN_FOLDER / 'saved_images'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

prev_poses = [(0, 0), (0, 0)]

device = torch.device("cuda")

mode_model = create_classifier()
print(torch.load(mode_model_path, map_location=device))
mode_model.load_state_dict(torch.load(mode_model_path, map_location=device))
mode_model.to(device)
mode_model.eval()

pos_models_dict = {}
for side in ['a', 'b', 'c']:
    model = create_player()
    model.load_state_dict(torch.load(pos_models_path[side], map_location=device))
    model.to(device)
    model.eval()
    pos_models_dict[side] = model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def screenshot():
    # prev_sct_img = None
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_number + 1]
        # while True:
        #     sct_img = sct.grab(monitor)
        #     if prev_sct_img is None or prev_sct_img.bgra != sct_img.bgra:
        #         prev_sct_img = sct_img
        #         break
        sct_img = sct.grab(monitor)

        # 轉換為 PIL 圖像
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        resized_img = img.resize((224, 126), cv2.INTER_LINEAR)
            
        height, width = resized_img.size
        size = max(height, width)

        square_img = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - height) // 2
        x_offset = (size - width) // 2

        square_img[x_offset:x_offset + width, y_offset:y_offset + height] = resized_img
        return Image.fromarray(square_img)

def detect_mode(model, img):
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor).squeeze().cpu()  # tensor([x, y])

    pred_idx = torch.argmax(outputs, dim=0) # TODO
    return modes_list[pred_idx]

def detect_pos(model, img):
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor).squeeze().cpu()  # tensor([x, y])

    return tuple(output.tolist())

def action(pos1, pos2, add_offset=False, pause=False):
    m = get_monitors()[monitor_number]
    # pyd.moveTo(int(pos1[0] * m.width + m.x), int(pos1[1] * m.height + m.y), _pause=False)
    # pyd.mouseDown(_pause=False)
    # pyd.moveTo(int(pos2[0] * m.width + m.x), int(pos2[1] * m.height + m.y), _pause=False)
    # pyd.mouseUp(_pause=pause)
    pyd.moveTo(int((pos1[0] + random.uniform(-0.1, 0.1) * add_offset) * m.width + m.x), int((pos1[1] + random.uniform(-0.1, 0.1) * add_offset) * m.height + m.y), _pause=False)
    pyd.mouseDown(_pause=False)
    pyd.moveTo(int((pos2[0] + random.uniform(-0.1, 0.1) * add_offset) * m.width + m.x), int((pos2[1] + random.uniform(-0.1, 0.1) * add_offset) * m.height + m.y), _pause=False)
    pyd.mouseUp(_pause=False)

def detect_and_action(img_save_queue):
    poses = []
    img = screenshot()
    if not MODE:
        mode = detect_mode(mode_model, img)
    elif MODE in modes_list:
        mode = MODE
    else:
        raise Exception('invalid mode')

    for side in sides_dict[mode]:
        poses.append(detect_pos(pos_models_dict[side], img))

    if round((prev_poses[0][0] - poses[0][0]) * 2, 3) == 0 and round((prev_poses[0][1] - poses[0][1]) * 2, 3) == 0 and \
        round((prev_poses[1][0] - poses[1][0]) * 2, 3) == 0 and round((prev_poses[1][1] - poses[1][1]) * 2, 3) == 0:
        action(poses[0], poses[1], add_offset=True)
    else:
        action(poses[0], poses[1])

    prev_poses[0], prev_poses[1] = poses[0], poses[1]

    print('mode:', mode if not MODE else MODE, 'pos:', poses)

    # action(pos1, pos2)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # img_save_queue.put((img.copy(), pos1, pos2, timestamp))

def save_image_worker(queue):
    while True:
        img, pos1, pos2, timestamp = queue.get()

        draw = ImageDraw.Draw(img)
        size = img.size[0]

        # pos1: 綠色 (start)
        x1, y1 = int(pos1[0] * size), int(pos1[1] * size)
        draw.ellipse((x1-5, y1-5, x1+5, y1+5), fill=(0, 255, 0))

        # pos2: 藍色 (end)
        x2, y2 = int(pos2[0] * size), int(pos2[1] * size)
        draw.ellipse((x2-5, y2-5, x2+5, y2+5), fill=(0, 0, 255))

        filename = f"{timestamp}.png"
        img.save(os.path.join(SAVE_DIR, filename))

def toggle_loop(running):
    was_pressed = False
    while True:
        if keyboard.is_pressed('\\') and not was_pressed:
            was_pressed = True
            running.value = not running.value  # 切換狀態

        elif not keyboard.is_pressed('\\') and was_pressed:
            was_pressed = False

        time.sleep(1/60)

if __name__ == '__main__':
    running = mp.Value('b', False)
    mp.Process(target=toggle_loop, args=(running,), daemon=True).start()

    manager = mp.Manager()
    img_save_queue = manager.Queue()
    # mp.Process(target=save_image_worker, args=(img_save_queue,), daemon=True).start()

    if MODE:
        print('current mode:', MODE)

    print(r'press \ to toggle detection loop')

    while True:
        if running.value:
            detect_and_action(img_save_queue)
            time.sleep(1/25)
        else:
            time.sleep(1/60)