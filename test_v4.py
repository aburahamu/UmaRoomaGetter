import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import threading
import time

class UmamusumeCapture:
    def __init__(self, root):
        self.root = root
        self.root.title("Umamusume Capture")
        self.is_capturing = False
        self.frames = []
        
        self.start_button = tk.Button(root, text="開始", command=self.start_capture)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(root, text="終了", command=self.stop_capture)
        self.stop_button.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=400, height=300)
        self.canvas.pack()
        
    def start_capture(self):
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.start()
        
    def stop_capture(self):
        self.is_capturing = False
        self.root.after(100, self.check_thread)
        
    def check_thread(self):
        if self.capture_thread.is_alive():
            self.root.after(100, self.check_thread)
        else:
            self.process_frames()
        
    def capture_frames(self):
        # Wait for the user to switch to the umamusume window
        time.sleep(2)
        window_app = gw.getWindowsWithTitle('umamusume')[0]
        while self.is_capturing:
            bbox = (window_app.left, window_app.top, window_app.right - window_app.left, window_app.bottom - window_app.top)
            screenshot = pyautogui.screenshot(region=bbox)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.frames.append(frame)
            
            # Display the frame in Tkinter window
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((400, int(400 * img.height / img.width)), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk
            
            time.sleep(1/60)
        
    def process_frames(self):
        if not self.frames:
            messagebox.showerror("エラー", "キャプチャされたフレームがありません。")
            return
        
        result_image = np.zeros_like(self.frames[0])
        for frame in self.frames:
            result_image = np.maximum(result_image, frame)
        
        cv2.imwrite("results.png", result_image)
        messagebox.showinfo("完了", "キャプチャと処理が完了しました。")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = UmamusumeCapture(root)
    root.mainloop()
