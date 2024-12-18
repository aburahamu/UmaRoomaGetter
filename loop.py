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
from concurrent.futures import ThreadPoolExecutor

# 対象アプリ名
app_title = "umamusume"
# レース情報のCSVファイル名
infos_file_name = "race_infos.csv"
# 着順情報のCSVファイル名
results_file_name = "race_results"
# 各種画像ファイルのパス
images_folder_path = "images"

## データフレームをクリアする関数
def delete_df():
    uuid = ""
    df_ranking = df_rank_zero.copy()
    df_raceinfo = df_race_zero.copy()
    # 結果ウィンドウの情報も初期化
    lbl_ranking.config(text=df_ranking.to_string())
    lbl_raceinfo.config(text=df_raceinfo.to_string())
    return True

## 抽出結果をCSVとして保存する関数
def regist_result():
    # データフレームの中身をCSVファイルに追加
    df_raceinfo.to_csv(infos_file_name, mode='a', header=False, index=False)
    df_ranking.to_csv(results_file_name, mode='a', header=False, index=False)
    # 既存のデータフレームを初期化
    delete_df()

## 画像内の文字列を取得する関数
def get_text(img_result, position):
    reader = easyocr.Reader(['ja'])
    result = reader.readtext(img_result)
    text_easyocr = ' '.join([res[1] for res in result])
    # 文字が取れた場合の処理
    if len(text_easyocr) > 0:
        # 取得した文字列をデータフレームに上書きする関数を呼び出す
        add_text(position, text_easyocr)

## 取得した文字列をデータフレームに上書きする関数
def add_text(text_position, text_easyocr):
    # 渡された文字列を分割
    result = text_easyocr.split()
    # 結果の要素数を取得
    result_length = len(result)
    # resultの各要素を変数に格納
    frame_no = result[0] if result_length > 0 else "unknown"
    player_name = result[1] if result_length > 1 else "unknown"
    plan_name = result[2] if result_length > 2 else "unknown"
    time_str = result[3] if result_length > 3 else "unknown"
    popularity = result[4] if result_length > 4 else "unknown"

    # n着の情報が入っていなければ処理する
    if text_position not in df_ranking['position'].values:
        # 着順情報のデータフレームに着順情報を追加する
        df_ranking = df_ranking.append({
            'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': text_position,
            'frame_no': frame_no,
            'player': player_name,
            'plan': plan_name,
            'time': time_str,
            'popularity': popularity
        }, ignore_index=True)
    # n着の情報が入っていれば処理する
    else:
        # 同じ着順の各データを上書きする
        df_ranking.loc[df_ranking['position'] == text_position, ['frame_no', 'player', 'plan', 'time', 'popularity']] = \
            [text_position, frame_no, player_name, plan_name, time_str, popularity]

## キャラクター名を判定する関数
def get_name(img_result, position):
    # 顔画像の数だけ顔判定処理を呼び出す
    for face_name, face_value in faces.items():
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge_faces") as executor:
            results = list(executor.map(judge_face, img_result, position, face_name))

## キャラクター画像との一致を判定する関数
def judge_face(img, position, face_name):
    # 渡された顔画像が画像内にあるかを判定する
    result = cv2.matchTemplate(img, faces[face_name], cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # 一致率が高ければ処理する
    if max_val > 0.9:
        # n着のキャラクター名を上書きする関数を呼び出す
        add_face(position, face_name)

## キャラクター名をデータフレームに上書きする関数
def add_face(text_position, face_name):
    # n着の情報がデータフレーム内になければ作る
    if text_position not in df_ranking['position'].values:
        df_ranking = df_ranking.append({
            'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': text_position,
            'name': face_name
        }, ignore_index=True)
    # n着の情報がデータフレーム内にあればキャラクター名を上書きする
    else:
        df_ranking.loc[df_ranking['position'] == text_position, 'name'] = face_name

def get_rank(img_result, position):
    for rank_name, rank_value in ranks.items():
        Process(target=judge_rank, args=(img_result, position, rank_name, ranks)).start()

def judge_rank(img, position, rank_name):
    res = cv2.matchTemplate(img, ranks[rank_name], cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.8:
        add_rank(position, rank_name)

def add_rank(text_position, rank_name):
    if text_position not in df_ranking['position'].values:
        df_ranking = df_ranking.append({
            'create': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position': text_position,
            'rank': rank_name
        }, ignore_index=True)
    else:
        df_ranking.loc[df_ranking['position'] == text_position, 'rank'] = rank_name

## 着順情報を読み取る関数を呼び出す関数
def judge_position(frame, position_key, position_value):
    # n着の画像があるかを判定
    res = cv2.matchTemplate(frame, position_value, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # n着の画像があれば画像を切り出して他の情報を読み取る関数を呼び出す
    if max_val > 0.8:
        x_stt, y_stt = max_loc
        y_stt -= int(frame.shape[0] * 0.05)
        y_end = y_stt + int(frame.shape[0] * 0.10)
        x_end = frame.shape[1]
        # 画像の切り出し
        img_result = frame[y_stt:y_end, x_stt:x_end]
        # 文字列の取り出し
        Process(target=get_text, args=(img_result, position_key)).start()
        # キャラ名の判定
        Process(target=get_name, args=(img_result, position_key)).start()
        # キャラランクの判定
        Process(target=get_rank, args=(img_result, position_key)).start()

## レース情報を判定しデータフレームに入れる関数
def judge_race_info(frame):
    if len(uuid) == 0:
        # レースの固有IDを生成
        uuid = ""
        # レース情報の画像範囲を設定
        x_stt, y_stt = 0
        y_stt -= int(frame.shape[0] * 0.4)
        y_end = y_stt + int(frame.shape[0] * 0.5)
        x_end = frame.shape[1]
        # レース情報の画像範囲を切り取り
        img_result = frame[y_stt:y_end, x_stt:x_end]
        # レース名
        Process(target=get_rank, args=(img_result)).start()
        # 馬場状態
        Process(target=get_rank, args=(img_result)).start()
        # 季節
        Process(target=get_rank, args=(img_result)).start()
        # 天候
        Process(target=get_rank, args=(img_result)).start()
        # 時間帯
        Process(target=get_rank, args=(img_result)).start()

## アプリウィンドウをキャプチャする関数
########################################################################################################
def get_screen():
    # ゲームウィンドウを取得
    window_app = gw.getWindowsWithTitle(app_title)[0]
    # ウィンドウ範囲を切り取り
    bbox = (window_app.left, window_app.top, window_app.right, window_app.bottom)
    # 切り取ったゲーム画面をグレースケールに変換
    frame = cv2.cvtColor(np.array(ImageGrab.grab(bbox)), cv2.COLOR_BGR2GRAY)
    # ステータスウィンドウのキャプチャ結果を書き換え
    lbl_frames.config(image=ImageTk.PhotoImage(image=Image.fromarray(frame)))
    # 着順結果のファイル名を順番に参照
    for key in positions.keys():
        # n着のデータが取れていなければ判定させる
        if key not in df_ranking['position'].values:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge_position") as executor:
                results = list(executor.map(judge_position, frame, key, positions[key]))
    # レース情報のデータが取れていなければ判定させる
    if key not in df_raceinfo['race_name'].values:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="judge_race_info") as executor:
                results = list(executor.map(judge_race_info, frame))
########################################################################################################

## メイン処理
########################################################################################################
if __name__ == '__main__':
    print("[UMA_ROOMA_GETTER]")
    ### 前処理
    freeze_support()

    # レース結果を保存するCSVファイルを準備
    if not os.path.exists(results_file_name):
        with open(results_file_name, 'w') as f:
            if os.path.getsize(results_file_name) == 0:
                f.write('create,race_id,position,name,rank,frame_no,player,plan,time,popularity\n')

    # レース情報を保存するCSVファイルを準備
    if not os.path.exists(infos_file_name):
        with open(infos_file_name, 'w') as f:
            if os.path.getsize(infos_file_name) == 0:
                f.write('create,uuid,race_name,condition,season,weather,timezone\n')

    # レースの固有ID
    global uuid
    uuid =  ""

    ## 判定用画像をオブジェクト化
    # キャラ顔
    global faces
    path_faces = os.path.join(images_folder_path, 'faces')
    faces = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_faces, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_faces)}
    # 着順
    global positions
    path_positions = os.path.join(images_folder_path, 'positions')
    positions = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_positions, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_positions)}
    # ステータスランク
    global ranks
    path_ranks = os.path.join(images_folder_path, 'ranks')
    ranks = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_ranks, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_ranks)}
    # 馬場状態
    global conditions
    path_conditions = os.path.join(images_folder_path, 'conditions')
    conditions = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_conditions,f ), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_conditions)}
    # 季節
    global seasons
    path_seasons = os.path.join(images_folder_path, 'seasons')
    seasons = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_seasons, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_seasons)}
    # 天候
    global weathers
    path_weathers = os.path.join(images_folder_path, 'weathers')
    weathers = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_weathers,f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_weathers)}
    # 時間帯
    global timezones
    path_timezones = os.path.join(images_folder_path, 'timezones')
    timezones = {os.path.splitext(f)[0]: cv2.imread(os.path.join(path_timezones, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path_timezones)}

    ## 着順用データフレーム
    global df_rank_zero
    df_rank_zero = pd.DataFrame(columns=['create', 'race_id', 'position', 'name', 'rank', 'frame_no', 'player', 'plan', 'time', 'popularity'])
    global df_ranking
    df_ranking = df_rank_zero.copy()

    ## レース情報用データフレーム
    global df_race_zero
    df_race_zero = pd.DataFrame(columns=['create', 'uuid', 'race_name', 'condition', 'season', 'weather', 'timezone'])
    global df_raceinfo
    df_raceinfo = df_race_zero.copy()

    ## tkinterウィンドウの設定
    root = tk.Tk()
    ## ゲーム画面を取得
    window_app = gw.getWindowsWithTitle(app_title)[0]
    ## 結果表示画面のサイズ設定
    root.geometry(f'300x{window_app.height}+{window_app.right+10}+{window_app.top}')
    ## レース情報の表示用ラベルを定義
    global lbl_raceinfo
    lbl_raceinfo = tk.Label(root, text=df_raceinfo.to_string())
    lbl_raceinfo.pack()
    ## 着順情報の表示用ラベルを定義
    global lbl_ranking
    lbl_ranking = tk.Label(root, text=df_ranking.to_string()) 
    lbl_ranking.pack()
    ## 取得データの登録ボタンを定義
    btn_register = tk.Button(root, text='登録', command=lambda: regist_result())
    btn_register.pack()
    ## 取得データの消去ボタンを定義
    btn_delete = tk.Button(root, text='消去', command=lambda: delete_df())
    btn_delete.pack()
    ## アプリ画面のフレーム表示用領域を定義
    global lbl_frames
    lbl_frames = tk.Label(root)
    lbl_frames.pack()
    print("設定完了")

    ## 関数get_screen()を60分の1秒ごとに呼び出す
    root.after(1000 // 60, get_screen)
    root.mainloop()
    print("プログラム終了")
########################################################################################################