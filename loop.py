import os
import cv2
import uuid
import numpy as np
import pandas as pd
import tkinter as tk
import datetime
import threading
import pygetwindow as gw
from PIL import Image, ImageTk, ImageGrab
from multiprocessing import Process
# from multiprocessing import freeze_support, Manager
from concurrent.futures import ThreadPoolExecutor

# 対象アプリ名
APP_TITLE = "umamusume"

# レース情報のCSVファイル名
INFOS_FILE_NAME = "race_infos.csv"

# 着順情報のCSVファイル名
RESULTS_FILE_NAME = "race_results.csv"

# 各種画像ファイルのパス
IMAGES_FOLDER_PATH = "images"

# レース情報を格納するデータフレーム
df_race_zero = pd.DataFrame(columns=[
    'create', 
    'race_id', 
    'grade', 
    'name', 
    'season', 
    'place', 
    'surface', 
    'distance', 
    'direction', 
    'weather', 
    'condition', 
    'timezone'
])
df_raceinfo = df_race_zero.copy()

# 着順情報を格納するデータフレーム
df_rank_zero = pd.DataFrame(columns=[
    'create', 
    'race_id', 
    'position', 
    'name', 
    'rank', 
    'frame_no', 
    'plan'
])
df_ranking = df_rank_zero.copy()

# スレッドロック用の変数
lock = threading.Lock()

#############################################################################
### 画像の読み込み用処理
#############################################################################
# 画像ディレクトリからデータを読み込む関数
# def load_images(folder_name):
#     return {os.path.splitext(f)[0]: cv2.imread(os.path.join(IMAGES_FOLDER_PATH, folder_name, f), cv2.IMREAD_GRAYSCALE) 
#             for f in os.listdir(os.path.join(IMAGES_FOLDER_PATH, folder_name))}
def load_images(folder_name):
    images = {}
    for f in os.listdir(os.path.join(IMAGES_FOLDER_PATH, folder_name)):
        # 完全なファイルパスを作成
        file_path = os.path.join(IMAGES_FOLDER_PATH, folder_name, f)

        # PILで画像を読み込む
        img = Image.open(file_path).convert('L')  # Lモードはグレースケール
        images[os.path.splitext(f)[0]] = img

    return images

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


class App:
    def __init__(self, root):
        # ウマ娘のアプリが見つからなければ終了
        try:
            window_app = gw.getWindowsWithTitle(APP_TITLE)[0]
        except Exception as e:
            print(f"ウマ娘のゲーム画面が見つかりません")

        self.root = root
        self.running = True  # 実行中フラグ
        self.counter = 0     # 呼び出し回数
        self.root.geometry(f'300x{window_app.height-40}+{window_app.right+5}+{window_app.top}')

        # 開始ボタン
        self.start_button = tk.Button(root, text="解析開始", command=self.start)
        self.start_button.pack()

        # 再開ボタン
        self.resume_button = tk.Button(root, text="再開", command=self.resume)
        self.resume_button.pack()

        # 登録ボタン
        self.btn_register = tk.Button(root, text='登録', command=self.regist_result)
        self.btn_register.pack()

        # 消去ボタン
        self.btn_delete = tk.Button(root, text='消去', command=self.delete_df)
        self.btn_delete.pack()

        # 取得したレース情報を表示するラベル
        self.lbl_raceinfo = tk.Label(root, text=df_raceinfo.to_string())
        self.lbl_raceinfo.pack()

        # 取得した着順情報を表示するラベル
        self.lbl_ranking = tk.Label(root, text=df_ranking.to_string()) 
        self.lbl_ranking.pack()

        # 取得したスクリーンショットを表示するラベル
        self.lbl_frames = tk.Label(root)
        self.lbl_frames.pack()

        # CSVファイルを初期化（なければ作る）
        self.initialize_csv_files()

        # ループ開始
        self.get_screen()

    #############################################################################
    ### その他の処理用
    #############################################################################
    # データフレームをクリアする関数
    def delete_df(self):
        global df_ranking, df_raceinfo, race_uuid
        # レース情報を初期化
        df_raceinfo = df_race_zero.copy()
        # 着順情報を初期化
        df_ranking = df_rank_zero.copy()
        # uuidを初期化
        race_uuid = ''
        # Tkinterのラベルをリフレッシュ
        self.refresh_labels()

    # Tkinterのラベルをリフレッシュする関数
    def refresh_labels(self):
        # 集計したレースの情報を上書き
        self.lbl_raceinfo.config(text=df_raceinfo.to_string())
        # 集計した着順の情報を上書き
        self.lbl_ranking.config(text=df_ranking.to_string())

    # 抽出結果をCSVとして保存する関数
    def regist_result(self):
        global df_ranking, df_raceinfo  
        # レース情報を上書き
        df_raceinfo.to_csv(INFOS_FILE_NAME, mode='a', header=False, index=False)
        # 着順情報を上書き
        df_ranking.to_csv(RESULTS_FILE_NAME, mode='a', header=False, index=False)
        # データフレームを初期化
        self.delete_df()

    #############################################################################
    ### データフレームの処理用
    #############################################################################
    # 着順用のデータフレームを更新する関数
    def update_rankDF(position, new_data):
        global df_ranking
        # 書き込み用処理をスレッドロック
        with lock:
            # 着順が入っていなければ追加
            if position not in df_ranking['position'].values:
                create_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_data['create'] = create_dt
                new_data['race_id'] = race_uuid
                new_data['position'] = position
                df_ranking = pd.concat([df_ranking, new_data], ignore_index=True)
            # 着順が入っていれば上書き
            else:
                df_ranking.loc[df_ranking['position'] == position, list(new_data.keys())] = list(new_data.values())

    # レース情報のデータフレームを更新する関数
    def update_raceDF(new_data):
        global df_raceinfo, race_uuid
        # 書き込み用処理をスレッドロック
        with lock:
            if race_uuid not in df_raceinfo['race_id'].values:
                create_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_data['create'] = create_dt
                new_data['race_id'] = race_uuid
                df_raceinfo = pd.concat([df_raceinfo, new_data], ignore_index=True)
            else:
                df_raceinfo.loc[df_raceinfo[1], list(new_data.keys())] = list(new_data.values())

    #############################################################################
    ### 着順情報の処理用
    #############################################################################
    # キャラクター名をデータフレームに上書きする関数
    def add_chara_name(self, position, chara_name):
        self.update_rankDF(position, {'name': chara_name})

    # ランクをデータフレームに上書きする関数
    def add_chara_rank(self,position, chara_rank):
        self.update_rankDF(position, {'rank': chara_rank})

    # 枠番をデータフレームに上書きする関数
    def add_chara_num(self, position, chara_num):
        self.update_rankDF(position, {'frame_no': chara_num})

    # 作戦をデータフレームに上書きする関数
    def add_chara_plan(self, position, chara_plan):
        self.update_rankDF(position, {'plan': chara_plan})

    # 渡された画像がimgに含まれる場合は着順用のデータフレームに情報を追加する関数
    def match_chara_and_add(img, position, key, img_temp, threshold, add_function):
        result = cv2.matchTemplate(img, img_temp, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)
        if max_val > threshold:
            add_function(position, key)

    # キャラクター名を取得する関数
    def get_chara_name(self, img, position):
        with ThreadPoolExecutor() as executor:
            for key in chara_faces.keys():
                executor.submit(self.match_chara_and_add, img, position, key, chara_faces[key], 0.9, self.add_chara_name)

    # キャラクターのランクを取得する関数
    def get_chara_rank(self, img, position):
        with ThreadPoolExecutor() as executor:
            for key in chara_ranks.keys():
                executor.submit(self.match_chara_and_add, img, position, key, chara_ranks[key], 0.9, self.add_chara_rank)

    # キャラクターの枠番を取得する関数
    def get_chara_num(self, img, position):
        with ThreadPoolExecutor() as executor:
            for key in chara_nums.keys():
                executor.submit(self.match_chara_and_add, img, position, key, chara_nums[key], 0.9, self.add_chara_num)

    # キャラクターの作戦を取得する関数
    def get_chara_plan(self, img, position):
        with ThreadPoolExecutor() as executor:
            for key in chara_plans.keys():
                executor.submit(self.match_chara_and_add, img, position, key, chara_plans[key], 0.9, self.add_chara_plan)

    # 着順で切り取られた画像から、キャラ名などの情報を取得する関数を呼び出す
    def analysis_chara_data(self, img, key):
        processes = [
            Process(target=self.get_chara_name, args=(img, key,)),
            Process(target=self.get_chara_rank, args=(img, key,)),
            Process(target=self.get_chara_num, args=(img, key,)),
            Process(target=self.get_chara_plan, args=(img, key,))
        ]
        for process in processes:
            process.start()

    #############################################################################
    ### レース情報の処理用
    #############################################################################

    # レースの格をデータフレームに上書きする関数
    def add_race_grade(self, key):
        self.update_raceDF({'grade': key})

    # レース名をデータフレームに上書きする関数
    def add_race_name(self, key):
        self.update_raceDF({'name': key})

    # レースの季節をデータフレームに上書きする関数
    def add_race_season(self, key):
        self.update_raceDF({'season': key})

    # レース場をデータフレームに上書きする関数
    def add_race_place(self, key):
        self.update_raceDF({'place': key})

    # レースのコースをデータフレームに上書きする関数
    def add_race_surface(self, key):
        self.update_raceDF({'surface': key})

    # レースのをデータフレームに上書きする関数
    def add_race_distance(self, key):
        self.update_raceDF({'distance': key})

    # レースの距離をデータフレームに上書きする関数
    def add_race_direction(self, key):
        self.update_raceDF({'direction': key})

    # レースのをデータフレームに上書きする関数
    def add_race_weather(self, key):
        self.update_raceDF({'weather': key})

    # レースのをデータフレームに上書きする関数
    def add_race_condition(self, key):
        self.update_raceDF({'condition': key})

    # レースのをデータフレームに上書きする関数
    def add_race_timezone(self, key):
        self.update_raceDF({'timezone': key})

    # 渡された画像がimgに含まれる場合は着順用のデータフレームに情報を追加する関数
    def match_race_and_add(img, key, img_temp, threshold, add_function):
        result = cv2.matchTemplate(img, img_temp, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)
        if max_val > threshold:
            add_function(key)

    # レースの格を取得する関数
    def get_race_grade(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_grades.keys():
                executor.submit(self.match_race_and_add, img, key, race_grades[key], 0.9, self.add_race_grade)

    # レース名を取得する関数
    def get_race_name(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_names.keys():
                executor.submit(self.match_race_and_add, img, key, race_names[key], 0.9, self.add_race_name)

    # レースの季節を取得する関数
    def get_race_season(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_seasons.keys():
                executor.submit(self.match_race_and_add, img, key, race_seasons[key], 0.9, self.add_race_name)

    # レース場を取得する関数
    def get_race_place(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_places.keys():
                executor.submit(self.match_race_and_add, img, key, race_places[key], 0.9, self.add_race_place)

    # レースのコースを取得する関数
    def get_race_surface(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_surfaces.keys():
                executor.submit(self.match_race_and_add, img, key, race_surfaces[key], 0.9, self.add_race_surface)

    # レースの距離を取得する関数
    def get_race_distance(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_distances.keys():
                executor.submit(self.match_race_and_add, img, key, race_distances[key], 0.9, self.add_race_distance)

    # レースの向きを取得する関数
    def get_race_direction(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_directions.keys():
                executor.submit(self.match_race_and_add, img, key, race_directions[key], 0.9, self.add_race_direction)

    # レースの天候を取得する関数
    def get_race_weather(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_weathers.keys():
                executor.submit(self.match_race_and_add, img, key, race_weathers[key], 0.9, self.add_race_weather)

    # レースの馬場状態を取得する関数
    def get_race_condition(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_conditions.keys():
                executor.submit(self.match_race_and_add, img, key, race_conditions[key], 0.9, self.add_race_condition)

    # レースの時間帯を取得する関数
    def get_race_timezone(self, img):
        with ThreadPoolExecutor() as executor:
            for key in race_timezones.keys():
                executor.submit(self.match_race_and_add, img, key, race_timezones[key], 0.9, self.add_race_timezone)

    # 切り取られたレース情報の画像から、レース情報を取得する関数を呼び出す
    def analysis_race_data(self, img):
        global race_uuid
        # uuidが未発行なら処理する
        if len(race_uuid) == 0:
            # レース情報の着順情報を紐づけるためのUUIDを発行する
            race_uuid = str(uuid.uuid4())
            # レース情報の取得用関数のプロセスに登録
            processes = [
                Process(target=self.get_race_grade, args=(img,)),
                Process(target=self.get_race_name, args=(img,)),
                Process(target=self.get_race_season, args=(img,)),
                Process(target=self.get_race_place, args=(img,)),
                Process(target=self.get_race_surface, args=(img,)),
                Process(target=self.get_race_distance, args=(img,)),
                Process(target=self.get_race_direction, args=(img,)),
                Process(target=self.get_race_weather, args=(img,)),
                Process(target=self.get_race_condition, args=(img,)),
                Process(target=self.get_race_timezone, args=(img,))
            ]
            # 解析プロセスを開始
            for process in processes:
                process.start()

    #############################################################################
    ### ゲーム画面の切り取り用
    #############################################################################
    # 着順の画像を探して、あればその行を画像として切り取ってキャラ情報の取得関数を呼び出す
    def judge_position(self, frame, position_key, position_value):
        result = cv2.matchTemplate(frame, position_value, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)
        
        if max_val > 0.9:
            # uuidが発行済みかで処理を分ける
            if len(race_uuid) > 0: 
                # 切り取る範囲を算出
                y_stt, x_stt = np.unravel_index(np.argmax(result), result.shape)
                y_stt -= 10
                y_end = y_stt + int(frame.shape[0] * 0.1)
                img_result = frame[y_stt:y_end, x_stt:]

                ## 切り取った画像の確認用
                # cv2.imshow('img_result', img_result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # 切り取った画像をキャラ情報の解析関数に渡す
                self.analysis_chara_data(img_result, position_key)
            else:
                # 切り取る範囲を算出
                x_stt = 0
                y_stt = int(frame.shape[0] * 0.4)
                y_end = y_stt + int(frame.shape[0] * 0.1)
                img_result = frame[y_stt:y_end, x_stt:]

                ## 切り取った画像の確認用
                # cv2.imshow('img_result', img_result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # 切り取った画像をレース情報の解析関数に渡す
                self.analysis_race_data(img_result)

    ### ウィジェット用のサムネイル画像を作成する関数
    def get_img_thumbnail(self, frame, root_width):
        original_width, original_height = frame.shape[1], frame.shape[0]
        label_width = root_width
        aspect_ratio = original_height / original_width
        new_width = label_width
        new_height = int(new_width * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))
        return img_tk

    # アプリのスクリーンショットを取得してレース情報と着順を判定させる関数
    def get_screen(self):
        if self.running:
            self.counter += 1
            print(f"Function called {self.counter} times")

            # 1秒間に60回呼ぶための設定
            self.root.after(1000 // 60, self.get_screen)

        # アプリを探す
        try:
            window_app = gw.getWindowsWithTitle(APP_TITLE)[0]
        except Exception as e:
            print(f"アプリが見つからなくなりました")

        # アプリの範囲を取得
        bbox = (window_app.left, window_app.top, window_app.right, window_app.bottom)

        # アプリのスクリーンショットを取得
        frame = cv2.cvtColor(np.array(ImageGrab.grab(all_screens=True, bbox = bbox)), cv2.COLOR_BGR2GRAY)

        ## ウィジェット用の画像を加工して表示
        global img_tk
        img_tk = self.get_img_thumbnail(frame, root.winfo_width())
        self.lbl_frames.config(image=img_tk)
        self.lbl_frames.image = img_tk
        root.update()

        # uuidを初期化
        global race_uuid
        race_uuid =''
        
        ## レース情報の取得関数を呼び出し
        if len(race_uuid) == 0:
            p01key = 'position01'
            self.judge_position(frame, p01key, positions[p01key])
        else:
            ## 着順情報の判定関数を呼び出し
            for key in positions.keys():
                if key not in df_ranking['position'].values:
                    self.judge_position(frame, key, positions[key])
        
    # 出力用のCSVファイルが無ければ作る関数
    def initialize_csv_files(self):
        for filename, headers in [(RESULTS_FILE_NAME, 'create,race_id,position,name,rank,number,plan,\n'),
                                (INFOS_FILE_NAME, 'create,uuid,grade,name,season,place,surface,distance,direction,weather,condition,timezone\n')]:
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write(headers)

    def stop(self):
        self.running = False
        print("Stopped after 10 seconds")

    def start(self):
        self.running = True
        self.counter = 0  # カウンタリセット
        self.get_screen()
        self.root.after(1000, self.stop)

    def resume(self):
        if not self.running:
            self.running = True
            self.get_screen()
            self.root.after(1000, self.stop)

# デフォルト処理
if __name__ == '__main__':
    print("[UMA_ROOMA_GETTER]")
    # アプリを開始
    root = tk.Tk()
    app = App(root)
    root.mainloop()