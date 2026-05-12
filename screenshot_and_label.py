from multiprocessing import Queue, Process, Lock
from threading import Thread
import time
import mss
from PIL import Image
from pynput.mouse import Listener
import os
from pathlib import Path
import cv2
import numpy as np
from screeninfo import get_monitors

task_queue = Queue()
result_queue = Queue()
pos_queue = Queue()
mss_lock = Lock()
was_pressed = False
press_pos = (0, 0)
release_pos = (0, 0)
monitor_number = 0

latest_image = None  # 儲存目前要顯示的圖片
image_display_running = True  # 控制顯示視窗的 thread 是否繼續運作

m = get_monitors()[monitor_number]

from config import SCREENSHOTS_FOLDER

def screenshot_worker(task_queue, result_queue, worker_id):
    with mss.mss() as sct:
        with mss_lock:
            while True:    
                task = task_queue.get()
                if task == "screenshot":
                    monitor = sct.monitors[monitor_number + 1]
                    sct_img = sct.grab(monitor)

                    # 轉換為 PIL 圖像
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    resized_img = img.resize((224, 126), cv2.INTER_AREA)

                    height, width = resized_img.size
                    size = max(height, width)

                    square_img = np.zeros((size, size, 3), dtype=np.uint8)
                    y_offset = (size - height) // 2
                    x_offset = (size - width) // 2

                    square_img[x_offset:x_offset + width, y_offset:y_offset + height] = resized_img

                    print(f"[Worker {worker_id}] 截圖完成")
                    result_queue.put((worker_id, square_img))
                    
                elif task == "exit":
                    print(f"[Worker {worker_id}] 結束")
                    break

def mouse_listener():
    def on_click(x, y, button, pressed):
        global was_pressed, press_pos, release_pos
        if pressed and not was_pressed:
            print("收到截圖請求...")
            print(f"Pressed at ({x}, {y})")
            press_pos = (x, y)
            was_pressed = True
            task_queue.put("screenshot")

        elif not pressed and was_pressed:
            print(f"Released at ({x}, {y})")
            release_pos = (x, y)
            was_pressed = False
            pos_queue.put({'press_pos': press_pos, 'release_pos': release_pos})
    
    with Listener(on_click=on_click) as listener:
        listener.join()

def image_display_loop():
    global latest_image, image_display_running
    while image_display_running:
        if latest_image is not None:
            cv2.imshow("Latest Screenshot", latest_image)
        key = cv2.waitKey(100)
        if key == 27:  # 按下 ESC 也可結束（選擇性）
            image_display_running = False
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    screenshot_folder = SCREENSHOTS_FOLDER / timestamp
    screenshot_folder.mkdir(parents=True, exist_ok=True)

    num_workers = 3
    workers = []
    for i in range(num_workers):
        p = Process(target=screenshot_worker, args=(task_queue, result_queue, i + 1), daemon=True)
        p.start()
        workers.append(p)

    time.sleep(0.2)
    print("按下滑鼠左鍵以截圖 (Ctrl+C 中止)")

    t = Thread(target=mouse_listener, daemon=True)
    t.start()

    # display_thread = Thread(target=image_display_loop, daemon=True)
    # display_thread.start()

    i = 0
    try:
        while True:
            while not result_queue.empty() and not pos_queue.empty():
                worker_id, img = result_queue.get()
                pos_dict = pos_queue.get()

                print(f"來自 Worker {worker_id} 的圖像，大小：{img.shape}")

                Image.fromarray(img).save(f'{screenshot_folder}/img{i}.jpg')

                latest_image = img.copy()

                with open(f'{screenshot_folder}/img{i}.txt', 'w') as f:
                    f.write(f"{(pos_dict['press_pos'][0] - m.x) / m.width:.5f}, {(pos_dict['press_pos'][1] - m.y) / m.height:.5f}\n")
                    f.write(f"{(pos_dict['release_pos'][0] - m.x) / m.width:.5f}, {(pos_dict['release_pos'][1] - m.y) / m.height:.5f}\n")

                i += 1

    except KeyboardInterrupt:
        print("程式終止中...")

        for _ in workers:
            task_queue.put("exit")
        for p in workers:
            p.join()

        image_display_running = False
        # display_thread.join()