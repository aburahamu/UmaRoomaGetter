import cv2
import pandas as pd
import pygetwindow as gw
import easyocr
import tkinter as tk
from datetime import datetime
from multiprocessing import Process, freeze_support, Manager
from PIL import Image, ImageTk, ImageGrab
import os
import numpy as np

def delete_df():
    global df_ranking
    df_ranking = df_zero.copy()
    lbl_ranking.config(text=df_ranking.to_string())
    return True

def regist_result():
    global df_ranking
    df_ranking.to_csv('results.csv', mode='a', header=False, index=False)
    delete_df()

def get_screen():
    global frame
    bbox = (window_app.left, window_app.top, window_app.right, window_app.bottom)
    frame = cv2.cvtColor(np.array(ImageGrab.grab(bbox)), cv2.COLOR_BGR2GRAY)
    lbl_frames.config(image=ImageTk.PhotoImage(image=Image.fromarray(frame)))
    for key in positions.keys():
        if key not in df_ranking['position'].values:
            Process(target=judge_position, args=(frame, key, positions[key], faces, ranks)).start()

def judge_position(frame, position_key, position_value, faces, ranks):
    res = cv2.matchTemplate(frame, position_value, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.8:
        x_stt, y_stt = max_loc
        y_stt -= int(frame.shape[0] * 0.05)
        y_end = y_stt + int(frame.shape[0] * 0.10)
        x_end = frame.shape[1]
        img_result = frame[y_stt:y_end, x_stt:x_end]
        Process(target=get_text, args=(img_result, position_key)).start()
        Process(target=get_name, args=(img_result, position_key, faces)).start()
        Process(target=get_rank, args=(img_result, position_key, ranks)).start()

def get_text(img_result, position):
    reader = easyocr.Reader(['ja'])
    result = reader.readtext(img_result)
    text_easyocr = ' '.join([res[1] for res in result])
    if len(text_easyocr) > 0:
        add_text(position, text_easyocr)

def add_text(text_position, text_easyocr):
    result = text_easyocr.split()
    global df_ranking
    if text_position not in df_ranking['position'].values:
        df_ranking = df_ranking.append({
            'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': text_position,
            'frame_no': result[0],
            'player': result[1],
            'plan': result[2],
            'time': result[3],
            'popularity': result[4]
        }, ignore_index=True)
    else:
        df_ranking.loc[df_ranking['position'] == text_position, ['frame_no', 'player', 'plan', 'time', 'popularity']] = result

def get_name(img_result, position, faces):
    for face_name, face_value in faces.items():
        Process(target=judge_face, args=(img_result, position, face_name, faces)).start()

def judge_face(img, position, face_name, faces):
    res = cv2.matchTemplate(img, faces[face_name], cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.8:
        add_face(position, face_name)

def add_face(text_position, face_name):
    global df_ranking
    if text_position not in df_ranking['position'].values:
        df_ranking = df_ranking.append({
            'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': text_position,
            'name': face_name
        }, ignore_index=True)
    else:
        df_ranking.loc[df_ranking['position'] == text_position, 'name'] = face_name

def get_rank(img_result, position, ranks):
    for rank_name, rank_value in ranks.items():
        Process(target=judge_rank, args=(img_result, position, rank_name, ranks)).start()

def judge_rank(img, position, rank_name, ranks):
    res = cv2.matchTemplate(img, ranks[rank_name], cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.8:
        add_rank(position, rank_name)

def add_rank(text_position, rank_name):
    global df_ranking
    if text_position not in df_ranking['position'].values:
        df_ranking = df_ranking.append({
            'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': text_position,
            'rank': rank_name
        }, ignore_index=True)
    else:
        df_ranking.loc[df_ranking['position'] == text_position, 'rank'] = rank_name

if __name__ == '__main__':
    freeze_support()

    # 前処理
    if not os.path.exists('results.csv'):
        with open('results.csv', 'w') as f:
            f.write('create,position,name,rank,frame_no,player,plan,time,popularity\n')

    positions = {os.path.splitext(f)[0]: cv2.imread(os.path.join('images/positions', f), cv2.IMREAD_GRAYSCALE) for f in os.listdir('images/positions')}
    faces = {os.path.splitext(f)[0]: cv2.imread(os.path.join('images/faces', f), cv2.IMREAD_GRAYSCALE) for f in os.listdir('images/faces')}
    ranks = {os.path.splitext(f)[0]: cv2.imread(os.path.join('images/ranks', f), cv2.IMREAD_GRAYSCALE) for f in os.listdir('images/ranks')}

    window_app = gw.getWindowsWithTitle('umamusume')[0]

    df_zero = pd.DataFrame(columns=['create', 'position', 'name', 'rank', 'frame_no', 'player', 'plan', 'time', 'popularity'])
    df_ranking = df_zero.copy()

    if len(pd.read_csv('results.csv')) == 0:
        df_ranking.to_csv('results.csv', index=False)

    # tkinterウィンドウの設定
    root = tk.Tk()
    root.geometry(f'300x{window_app.height}+{window_app.right+10}+{window_app.top}')
    btn_delete = tk.Button(root, text='消去', command=lambda: delete_df())
    btn_delete.pack()
    lbl_ranking = tk.Label(root, text=df_ranking.to_string())
    lbl_ranking.pack()
    btn_register = tk.Button(root, text='登録', command=lambda: regist_result())
    btn_register.pack()
    lbl_frames = tk.Label(root)
    lbl_frames.pack()

    root.after(1000 // 60, get_screen)
    root.mainloop()

    df_ranking.to_csv('results.csv', mode='a', header=False, index=False)
    print("プログラム終了")