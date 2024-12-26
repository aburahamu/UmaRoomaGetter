import os
import cv2
import uuid
import numpy as np
import pandas as pd
import tkinter as tk
import datetime as dt
import threading
import pygetwindow as gw
from PIL import Image, ImageTk, ImageGrab
from concurrent.futures import ThreadPoolExecutor

# 対象アプリ名
APP_TITLE = "umamusume"

# CSVファイル名
INFOS_FILE_NAME = "race_infos.csv"
RESULTS_FILE_NAME = "race_results.csv"
IMAGES_FOLDER_PATH = "images"

# データフレーム初期化
df_raceinfo = pd.DataFrame(columns=[
    'create', 'race_id', 'grade', 'name', 'season',
    'place', 'surface', 'distance', 'direction',
    'weather', 'condition', 'timezone'
])
df_ranking = pd.DataFrame(columns=[
    'create', 'race_id', 'position', 'name', 'rank',
    'frame_no', 'plan'
])

# レース情報を判定する際の一致率ボーダー
race_thresholds = {
    'grade': 0.9, 'name': 0.85, 'season': 0.8, 'place': 0.9,
    'surface': 0.9, 'distance': 0.9, 'direction': 0.9,
    'weather': 0.7, 'condition': 0.85, 'timezone': 0.85
}

# 着順情報を判定する際の一致率ボーダー
chara_thresholds = {
    'name': 0.9, 'rank': 0.95, 'frame_no': 0.92, 'plan': 0.85
}

# スレッドロック用の変数
lock = threading.Lock()

## PIL版
def load_images(folder_name):
    """画像ディレクトリからデータを読み込む関数"""
    images = {}
    folder_path = os.path.join(IMAGES_FOLDER_PATH, folder_name)
    for f in os.listdir(folder_path):
        img_path = os.path.join(folder_path, f)
        # PILを使って画像を開く
        with Image.open(img_path) as img:
            # グレースケールに変換
            img_gray = img.convert('L')
            # NumPy配列に変換
            images[os.path.splitext(f)[0]] = np.array(img_gray)
    return images

# 画像の初期化
positions = load_images('positions')
race_replay = load_images('race_replay')
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

# レース情報解析用の画像配列を定義
race_dict = {
    'grade': race_grades, 'name': race_names, 'season': race_seasons, 'place': race_places,
    'surface': race_surfaces, 'distance': race_distances, 'direction': race_directions,
    'weather': race_weathers, 'condition': race_conditions, 'timezone': race_timezones
}

# 着順解析用の画像配列を定義
chara_dict = {'name': chara_faces, 'rank': chara_ranks, 'frame_no': chara_nums, 'plan': chara_plans}

### アプリのクラスを作成
class App:
    ###################################################################################################
    ### UIに関する処理を定義したブロック
    ###################################################################################################
    def __init__(self, root):
        # ウマ娘のアプリが見つからなければ終了
        try:
            window_app = gw.getWindowsWithTitle(APP_TITLE)[0]
            # window_app.resize(514, 924)
            self.root = root
            self.is_running = True
            self.running = True
            self.counter = 0
            self.root.geometry(f'300x886+{window_app.right+5}+{window_app.top}')

            global race_uuid
            race_uuid = ''
            print(f"App Start.")

            # ボタンとラベルの設定
            self.init_ui()
            self.initialize_csv_files()
            self.get_screen()
        except IndexError:
            print("ウマ娘のゲーム画面が見つかりません")

    def init_ui(self):
        """ユーザーインターフェイスを初期化する関数"""
        # ボタンのtextを書き換えるため変数化
        self.toggle_button = tk.Button(self.root, text="一時停止", width=20, fg="black", bg="yellow", command=self.toggle_resume_stop)
        self.toggle_button.pack(pady=10)

        tk.Button(self.root, text='登録', width=20, fg="white", bg="green", command=self.regist_result).pack(pady=10)
        tk.Button(self.root, text='消去', width=20, fg="white", bg="gray", command=self.delete_df).pack(pady=10)

        self.lbl_raceinfo = tk.Label(self.root, text=df_raceinfo.to_string())
        self.lbl_raceinfo.pack()
        self.lbl_ranking = tk.Label(self.root, text=df_ranking.to_string())
        self.lbl_ranking.pack()
        self.lbl_frames = tk.Label(self.root)
        self.lbl_frames.pack()

    def refresh_labels(self):
        # レース情報を文字列化
        df_race_filtered = df_raceinfo.drop(columns=['race_id', 'create'])
        self.lbl_raceinfo.config(text=df_race_filtered.T.to_string())

        # 着順情報を文字列化
        df_ranking_filtered = df_ranking.drop(columns=['race_id', 'create'])
        self.lbl_ranking.config(text=df_ranking_filtered.sort_values(by='position').to_string(index=False))

    def toggle_resume_stop(self):
        """再開/停止ボタンのトグル処理"""
        if self.is_running:
            self.stop()
            self.toggle_button.config(text="再開", fg="white", bg="blue")
        else:
            self.resume()
            self.toggle_button.config(text="一時停止", fg="black", bg="yellow")
        self.is_running = not self.is_running  # 状態を反転

    def start(self):
        self.running = True
        print(f"App Start.")
        self.counter = 0
        self.get_screen()

    def stop(self):
        self.running = False
        print(f"App Stop.")

    def resume(self):
        if not self.running:
            self.running = True
            print(f"App ReStart.")
            self.get_screen()
    ###################################################################################################

    ###################################################################################################
    ### データフレームに関する処理を定義したブロック
    ###################################################################################################
    def delete_df(self):
        global race_uuid, df_raceinfo, df_ranking
        with lock:
            df_raceinfo.drop(df_raceinfo.index, inplace=True)
            df_ranking.drop(df_ranking.index, inplace=True)
            race_uuid = ''
        print(f"DataFrame is deleted.")
        self.refresh_labels()

    def regist_result(self):
        global df_raceinfo, df_ranking
        df_raceinfo.to_csv(INFOS_FILE_NAME, mode='a', header=False, index=False)
        print(f"レース情報をCSVファイルに追加しました")
        df_ranking.to_csv(RESULTS_FILE_NAME, mode='a', header=False, index=False)
        print(f"着順情報をCSVファイルに追加しました")
        self.delete_df()

    def initialize_csv_files(self):
        """出力用のCSVファイルが無ければ作る関数"""
        for filename, headers in [(RESULTS_FILE_NAME, 'create,race_id,position,name,rank,frame_no,plan,\n'),
                                   (INFOS_FILE_NAME, 'create,race_id,grade,name,season,place,surface,distance,direction,weather,condition,timezone\n')]:
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write(headers)
                print(f"データ出力用のCSVファイルを作成しました")
    ###################################################################################################
    
    ###################################################################################################
    ### 着順情報に関する処理を定義したブロック
    ###################################################################################################
    ## 着順情報のデータフレームを更新する関数
    def update_rankDF(self, position, new_data):
        global df_ranking
        with lock:
            # 着順が入っていなければ追加
            if position not in df_ranking['position'].values:
                create_dt = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_data['create'] = create_dt
                new_data['race_id'] = race_uuid
                new_data['position'] = position
                df_new_data = pd.DataFrame([new_data])
                df_ranking = pd.concat([df_ranking, df_new_data], ignore_index=True)
            # 着順が入っていれば上書き
            else:
                for col, val in new_data.items():
                    df_ranking.loc[df_ranking['position'] == position, col] = val

    ## 画像の一致を判定する関数
    def match_and_add_rank(self, img, position, typ, key, img_temp, threshold):
        img_temp_new = self.get_new_template(img_temp)
        result = cv2.matchTemplate(img, img_temp_new, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)
        # if typ == "name":
        #     print(f"[rank]【{typ}】{key} = {format(max_val, '.2f')}")
        if max_val > threshold:
            # print(f"[rank]【{typ}】{key} = {format(max_val, '.2f')}")
            self.update_rankDF(position, {typ: key})
    
    ## 着順に関する情報を判定させるスレッドを作る関数
    def get_chara_info(self, img, position, type_key):
        with ThreadPoolExecutor() as executor:
            # keyごとにマッチングする
            if type_key in chara_dict:
                for key in chara_dict[type_key].keys():
                    threshold = chara_thresholds.get(type_key, 0.9)    # デフォルト値は0.9
                    img_origin = img
                    img_temp = chara_dict[type_key][key]
                    # 切り取る範囲を算出
                    if type_key == "frame_no":
                        ## 切り取った画像の確認用
                        # cv2.imshow('img', img_origin)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        height, width = img_origin.shape
                        x_start = int(width * 0.30)
                        x_end = int(width * 0.40)
                        img_origin = img_origin[:, x_start:x_end]
                        ## 切り取った画像の確認用
                        # cv2.imshow('img_origin', img_origin)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    executor.submit(self.match_and_add_rank, img_origin, position, type_key, key, img_temp, threshold)

    ## 取得する情報の種類を判定用スレッドに渡す関数
    def analysis_chara_data(self, img, key):
        types = ['name', 'rank', 'frame_no', 'plan']
        for typ in types:
            self.get_chara_info(img, key, typ)

    ## 着順の画像を見つけてその段を切り取る関数
    def judge_position(self, frame, position_key, position_value):
        h, w = position_value.shape[:2]
        result = cv2.matchTemplate(frame, position_value, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)

        if max_val > 0.9:
            if race_uuid:
                # 切り取る範囲を算出
                y_stt, x_stt = np.unravel_index(np.argmax(result), result.shape)
                y_stt = y_stt + int(h / 2) - int(frame.shape[0] * 0.05)
                y_end = y_stt + int(frame.shape[0] * 0.1)
                img_result = frame[y_stt:y_end, x_stt:]
                ## 切り取った画像の確認用
                # cv2.imshow('img_result', img_result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                ## 切り取った画像をキャラ情報の解析関数に渡す
                self.analysis_chara_data(img_result, position_key)
    ###################################################################################################

    ###################################################################################################
    ### レース情報に関する処理を定義したブロック
    ###################################################################################################
    ## レース情報のデータフレームを更新する関数 
    def update_raceDF(self, new_data):
        global df_raceinfo, race_uuid
        with lock:
            # レースIDが入っていない場合はレコード追加
            if race_uuid not in df_raceinfo['race_id'].values:
                new_data['create'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_data['race_id'] = race_uuid
                df_new_data = pd.DataFrame([new_data])
                df_raceinfo = pd.concat([df_raceinfo, df_new_data], ignore_index=True)
            # レースIDが入っていればレコード上書き
            else:
                for typ, key in new_data.items():
                        if typ in df_raceinfo.columns:
                            df_raceinfo.at[0, typ] = key
    
    ## 一致する画像が含まれるかを判定する関数
    def match_and_add_race(self, img, typ, key, img_temp, threshold):
        img_temp_new = self.get_new_template(img_temp)
        result = cv2.matchTemplate(img, img_temp_new, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)
        if max_val > threshold:
            self.update_raceDF({typ: key})

    ## レース情報の判定用スレッドを作る関数
    def get_race_info(self, img, type_key):
        with ThreadPoolExecutor() as executor:
            if type_key in race_dict:
                for key in race_dict[type_key].keys():
                    threshold = race_thresholds.get(type_key, 0.9)    # デフォルト値は0.9
                    executor.submit(self.match_and_add_race, img, type_key, key, race_dict[type_key][key], threshold)

    ## 取得する情報の種類を判定用スレッドに渡す関数
    def analysis_race_data(self, img):
        global race_uuid
        if len(race_uuid) == 0:
            race_uuid = str(uuid.uuid4())
            types = ['grade', 'name', 'season', 'place', 
                     'surface', 'distance', 'direction', 
                     'weather', 'condition', 'timezone']
            for typ in types:
                self.get_race_info(img, typ)

    # レース情報が記載された場所を見つけて切り取る関数
    def judge_raceinfo(self, frame, img):
        result = cv2.matchTemplate(frame, img, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(result)

        if max_val > 0.9:
            # 切り取る範囲を算出
            x_stt = 0
            y_stt = int(frame.shape[0] * 0.4)
            y_end = y_stt + int(frame.shape[0] * 0.1)
            img_result = frame[y_stt:y_end, x_stt:]
            # 切り取った画像の確認用
            # cv2.imshow('img_result', img_result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            ## 切り取った画像をレース情報の解析関数に渡す
            self.analysis_race_data(img_result)
    ###################################################################################################

    ###################################################################################################
    ### ゲーム画面を取得するブロック
    ###################################################################################################
    def get_new_template(self, img_old):
        """ゲームウィンドウの幅に応じてテンプレート画像のサイズを変更する"""
        img_temp_new = cv2.resize(img_old, None, fx=zoom_ratio, fy=zoom_ratio, interpolation=cv2.INTER_CUBIC)
        return img_temp_new

    def get_img_thumbnail(self, frame):
        """ウィジェット用のサムネイル画像を作成する関数"""
        original_width, original_height = frame.shape[1], frame.shape[0]
        label_width = self.root.winfo_width()
        aspect_ratio = original_height / original_width
        new_width = int(label_width * 0.7)
        new_height = int(new_width * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return ImageTk.PhotoImage(image=Image.fromarray(resized_frame))

    def get_screen(self):
        """ゲーム画面のスクショを取得して情報の判定処理をさせる関数"""
        if self.running:
            self.counter += 1
            # print(f"Function called {self.counter} times", end='\r')

            # スクリーンショット取得処理
            try:
                window_app = gw.getWindowsWithTitle(APP_TITLE)[0]
                bbox = (window_app.left, window_app.top, window_app.right, window_app.bottom)
                frame = cv2.cvtColor(np.array(ImageGrab.grab(all_screens=True, bbox=bbox)), cv2.COLOR_BGR2GRAY)
                
                ## ゲーム画面のサムネイルを作る
                # img_tk = self.get_img_thumbnail(frame)
                # self.lbl_frames.config(image=img_tk)
                # self.lbl_frames.image = img_tk
                # self.root.update()

                ## 画面の比率を覚えておく
                # キャプチャ時のウィンドウ幅が500pxになるサイズを基準にしている
                global zoom_ratio
                zoom_ratio = window_app.width / 514 

                # 情報の取得要否フラグを初期化
                need_raceinfo = False
                need_positions = False

                # 着順情報の取得要否を判定する
                if len(race_uuid) > 0:
                    need_positions = True
                # 着順情報の取得が必要な場合は処理する
                if need_positions:
                    for key in positions.keys():
                        self.judge_position(frame, key, positions[key])

                # レース情報の取得要否を判定する
                if len(race_uuid) == 0:
                    need_raceinfo = True
                if len(df_raceinfo) > 0:
                    # レース情報に1つでも抜けがあれば取得要とする
                    if df_raceinfo.isnull().any().any() or (df_raceinfo == '').any().any():
                        need_raceinfo = True
                # レース情報の取得が必要な場合は処理する
                if need_raceinfo:
                    self.judge_raceinfo(frame, race_replay['リプレイ'])

            except Exception as e:
                print("アプリを探す時にエラーが発生しました:", e)

            # UI上のラベルを更新
            self.refresh_labels()
            # 一定時間後に再帰的に呼び出す
            self.root.after(1000 // 4, self.get_screen)
    ###################################################################################################

# デフォルト処理
if __name__ == '__main__':
    print(f"[UMA_ROOMA_GETTER]")
    root = tk.Tk()
    app = App(root)
    root.mainloop()
