import cv2
import pandas as pd
import pygetwindow as gw
import easyocr
import tkinter as tk
from datetime import datetime
from multiprocessing import freeze_support, Manager
from PIL import Image, ImageTk, ImageGrab
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

# 対象アプリ名
app_title = "umamusume"

# レース情報のCSVファイル名
infos_file_name = "race_infos.csv"

# 着順情報のCSVファイル名
results_file_name = "race_results.csv"

# 各種画像ファイルのパス
images_folder_path = "images"

# グローバル変数の初期化
uuid = ""
df_race_zero = pd.DataFrame(columns=['create', 'uuid', 'race_name', 'condition', 'season', 'weather', 'timezone'])
df_raceinfo = df_race_zero.copy()
df_rank_zero = pd.DataFrame(columns=['create', 'race_id', 'position', 'name', 'rank', 'frame_no', 'player', 'plan', 'time', 'popularity'])
df_ranking = df_rank_zero.copy()

# 画像ディレクトリからデータを読み込む関数
def load_images(folder_name):
    return {os.path.splitext(f)[0]: cv2.imread(os.path.join(images_folder_path, folder_name, f), cv2.IMREAD_GRAYSCALE) 
            for f in os.listdir(os.path.join(images_folder_path, folder_name))}

# 初期化処理
positions = load_images('positions')
race_grades = load_images('race_grades')
race_names = load_images('race_names')
race_seasons = load_images('race_seasons')
race_places = load_images('race_places')
race_surfaces = load_images('race_surfaces')
race_distances = load_images('race_distances')
race_directions = load_images('race_directions')
race_weathers = load_images('race_weathers')
race_conditions = load_images('race_conditions')
race_timezones = load_images('race_timezones')
chara_faces = load_images('chara_faces')
chara_ranks = load_images('chara_ranks')
chara_nums = load_images('chara_nums')
chara_plans = load_images('chara_plans')

# データフレームをクリアする関数
def delete_df():
    global df_ranking, df_raceinfo
    # レース情報を初期化
    df_raceinfo = df_race_zero.copy()
    # 着順情報を初期化
    df_ranking = df_rank_zero.copy()
    # Tkinterのラベルをリフレッシュ
    refresh_labels()

# Tkinterのラベルをリフレッシュする関数
def refresh_labels():
    # 集計したレースの情報を上書き
    lbl_raceinfo.config(text=df_raceinfo.to_string())
    # 集計した着順の情報を上書き
    lbl_ranking.config(text=df_ranking.to_string())

# 抽出結果をCSVとして保存する関数
def regist_result():
    global df_ranking, df_raceinfo  
    # レース情報を上書き
    df_raceinfo.to_csv(infos_file_name, mode='a', header=False, index=False)
    # 着順情報を上書き
    df_ranking.to_csv(results_file_name, mode='a', header=False, index=False)
    # データフレームを初期化
    delete_df()

# 画像内の文字列を取得する関数
def get_text(img_result, position):
    reader = easyocr.Reader(['ja'])
    result = reader.readtext(img_result)
    text_easyocr = ' '.join(res[1] for res in result)

    if text_easyocr:
        add_text(position, text_easyocr)

# 取得した文字列をデータフレームに上書きする関数
def add_text(text_position, text_easyocr):
    global df_ranking
    result = text_easyocr.split()
    frame_no, player_name, plan_name, time_str, popularity = ["unknown"] * 5  # デフォルト値

    for i in range(min(len(result), 5)):  # 最大5項目まで取得
        locals()[['frame_no', 'player_name', 'plan_name', 'time_str', 'popularity'][i]] = result[i]

    update_dataframe(df_ranking, text_position, {
        'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'position': text_position,
        'frame_no': frame_no,
        'player': player_name,
        'plan': plan_name,
        'time': time_str,
        'popularity': popularity
    })

def update_rankDF(dataframe, position, new_data):
    global df_ranking
    if position not in dataframe['position'].values:
        df_ranking = dataframe.append(new_data, ignore_index=True)
    else:
        dataframe.loc[dataframe['position'] == position, list(new_data.keys())] = list(new_data.values())

def update_raceDF(dataframe, position, new_data):
    global df_ranking
    if position not in dataframe['position'].values:
        df_ranking = dataframe.append(new_data, ignore_index=True)
    else:
        dataframe.loc[dataframe['position'] == position, list(new_data.keys())] = list(new_data.values())

# キャラクター名を判定する関数
def get_chara_name(img_result, position):
    with ThreadPoolExecutor(max_workers=1) as executor:
        for face_name in chara_faces.keys():
            executor.submit(judge_face, img_result, position, face_name)

# キャラクター画像との一致を判定する関数
def judge_face(img, position, face_name):
    match_template_and_add(img, position, face_name, chara_faces[face_name], 0.9, add_chara_name)

def match_template_and_add(img, position, name, template_img, threshold, add_function):
    result = cv2.matchTemplate(img, template_img, cv2.TM_CCOEFF_NORMED)
    max_val = np.max(result)
    
    if max_val > threshold:
        add_function(position, name)

# キャラクター名をデータフレームに上書きする関数
def add_chara_name(text_position, face_name):
    update_dataframe(df_ranking, text_position, {'name': face_name})

def get_chara_rank(img_result, position):
    with ThreadPoolExecutor() as executor:
        for rank_name in chara_ranks.keys():
            executor.submit(judge_rank, img_result, position, rank_name)

def judge_rank(img, position, rank_name):
    match_template_and_add(img, position, rank_name, chara_ranks[rank_name], 0.8, add_rank)

def add_rank(text_position, rank_name):
    update_rankDF(df_ranking, text_position, {'rank': rank_name})

# 着順の画像を探して、あればその行を画像として切り取ってキャラ情報の取得関数を呼び出す
def judge_position(frame, position_key, position_value):
    result = cv2.matchTemplate(frame, position_value, cv2.TM_CCOEFF_NORMED)
    max_val = np.max(result)
    
    if max_val > 0.8:
        x_stt, y_stt = np.unravel_index(np.argmax(result), result.shape)        
        y_stt -= int(frame.shape[0] * 0.05)
        y_end = y_stt + int(frame.shape[0] * 0.10)
        img_result = frame[y_stt:y_end, x_stt:]
        execute_multiple_processes(img_result, position_key)

# 着順で切り取られた画像から、キャラ名などの情報を取得する関数を呼び出す
def execute_multiple_processes(img_result, position_key):
    processes = [
        Process(target=get_chara_name, args=(img_result, position_key)),
        Process(target=get_chara_rank, args=(img_result, position_key)),
        Process(target=get_chara_num, args=(img_result, position_key)),
        Process(target=get_chara_plan, args=(img_result, position_key))
    ]
    for process in processes:
        process.start()

def judge_race_info(frame):
    if not uuid:
        x_stt, y_stt = 0, int(frame.shape[0] * 0.4)
        y_end = y_stt + int(frame.shape[0] * 0.5)
        img_result = frame[y_stt:y_end, x_stt:]

        processes = [
            Process(target=get_race_grade, args=(img_result)),
            Process(target=get_race_name, args=(img_result)),
            Process(target=get_race_season, args=(img_result)),
            Process(target=get_race_place, args=(img_result)),
            Process(target=get_race_surface, args=(img_result)),
            Process(target=get_race_distance, args=(img_result)),
            Process(target=get_race_direction, args=(img_result)),
            Process(target=get_race_weather, args=(img_result)),
            Process(target=get_race_condition, args=(img_result)),
            Process(target=get_race_timezone, args=(img_result))
        ]
        
        for process in processes:
            process.start()

def get_screen():
    try:
        window_app = gw.getWindowsWithTitle(app_title)[0]
    except Exception as e:
        print(f"アプリが見つからなくなりました")

    # モニターに合わせてbboxを補正
    bbox = (window_app.left, window_app.top, window_app.right, window_app.bottom)
    frame = cv2.cvtColor(np.array(ImageGrab.grab(all_screens=True, bbox = bbox)), cv2.COLOR_BGR2GRAY)

    global img_tk
    original_width, original_height = frame.shape[1], frame.shape[0]
    label_width = root.winfo_width()
    aspect_ratio = original_height / original_width
    new_width = label_width
    new_height = int(new_width * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))
    lbl_frames.config(image=img_tk)
    lbl_frames.image = img_tk
    root.update()
    
    for key in positions.keys():
        if key not in df_ranking['position'].values:
            judge_position(frame, key, positions[key])
    
    if key not in df_raceinfo['race_name'].values:
        judge_race_info(frame)

# 出力用のCSVファイルが無ければ作る関数
def initialize_csv_files():
    for filename, headers in [(results_file_name, 'create,race_id,position,name,rank,frame_no,player,plan,time,popularity\n'),
                              (infos_file_name, 'create,uuid,race_name,condition,season,weather,timezone\n')]:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(headers)

# デフォルト処理
if __name__ == '__main__':
    print("[UMA_ROOMA_GETTER]")
    freeze_support()
    initialize_csv_files()

    try:
        window_app = gw.getWindowsWithTitle(app_title)[0]
    except Exception as e:
        print(f"アプリが見つかりません")

    root = tk.Tk()
    root.geometry(f'300x{window_app.height-35}+{window_app.right+5}+{window_app.top}')

    lbl_raceinfo = tk.Label(root, text=df_raceinfo.to_string())
    lbl_raceinfo.pack()

    lbl_ranking = tk.Label(root, text=df_ranking.to_string()) 
    lbl_ranking.pack()

    btn_register = tk.Button(root, text='登録', command=regist_result)
    btn_register.pack()

    btn_delete = tk.Button(root, text='消去', command=delete_df)
    btn_delete.pack()

    lbl_frames = tk.Label(root)
    lbl_frames.pack()

    print("設定完了")
    root.after(1000 // 60, get_screen)
    root.mainloop()
    print("プログラム終了")
