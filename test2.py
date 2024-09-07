import pygetwindow as gw
import pyautogui
import easyocr
import pandas as pd
import cv2
import numpy as np
import os
import time

# ゲームウィンドウの取得
window = gw.getWindowsWithTitle('umamusume')[0]
window.activate()

# ゲーム画面のキャプチャー
screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
screenshot_np = np.array(screenshot)

# OCRの初期化
reader = easyocr.Reader(['ja'])

# 画像の前処理
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

# レース概要の取得
race_info_region = (0, int(window.height * 0.4), window.width, int(window.height * 0.5))
race_info_image = screenshot_np[int(window.height * 0.4):int(window.height * 0.5), 0:window.width]
race_info_image = preprocess_image(race_info_image)

# リトライ回数
max_retries = 5

def get_race_info_text(image, retries=0):
    if retries >= max_retries:
        return ["不明"] * 8  # 代替文字を使用
    try:
        text = reader.readtext(image, detail=0)
        if len(text) < 2:
            raise ValueError("OCR failed to extract sufficient text")
        return text
    except Exception as e:
        print(f"OCR failed: {e}, retrying... ({retries + 1}/{max_retries})")
        time.sleep(1)
        return get_race_info_text(image, retries + 1)

race_info_text = get_race_info_text(race_info_image)

# レース概要のデータフレーム化
race_info = {
    'レースの格': race_info_text[0].split()[0] if len(race_info_text[0].split()) > 0 else "不明",
    'レース名': race_info_text[0].split()[1] if len(race_info_text[0].split()) > 1 else "不明",
    '場所': race_info_text[1].split()[0] if len(race_info_text[1].split()) > 0 else "不明",
    '馬場': race_info_text[1].split()[1] if len(race_info_text[1].split()) > 1 else "不明",
    '距離': race_info_text[1].split()[2] if len(race_info_text[1].split()) > 2 else "不明",
    '距離名': race_info_text[1].split()[3] if len(race_info_text[1].split()) > 3 else "不明",
    '周回方向': race_info_text[1].split()[4] if len(race_info_text[1].split()) > 4 else "不明",
    '天候': race_info_text[1].split()[5] if len(race_info_text[1].split()) > 5 else "不明",
    '馬場の状態': race_info_text[1].split()[6] if len(race_info_text[1].split()) > 6 else "不明",
    '時間帯': race_info_text[1].split()[7] if len(race_info_text[1].split()) > 7 else "不明"
}
df_raceinfo = pd.DataFrame([race_info])

# キャラクターのレース結果の取得
character_results = []
for i in range(9, 19):
    character_region = (0, int(window.height * 0.5) + (i - 9) * int(window.height * 0.1), window.width, int(window.height * 0.1))
    character_image = screenshot_np[int(window.height * 0.5) + (i - 9) * int(window.height * 0.1):int(window.height * 0.5) + (i - 8) * int(window.height * 0.1), 0:window.width]
    
    if character_image.size == 0:
        continue  # スキップ
    
    character_image = preprocess_image(character_image)
    character_text = reader.readtext(character_image, detail=0)
    
    # キャラクターの画像判定
    character_face = character_image[:, :int(character_image.shape[1] * 0.2)]
    best_match = None
    best_score = 0
    for face_file in os.listdir('images/faces'):
        face_image = cv2.imread(os.path.join('images/faces', face_file), cv2.IMREAD_GRAYSCALE)
        if face_image.shape[0] > character_face.shape[0] or face_image.shape[1] > character_face.shape[1]:
            continue  # スキップ
        res = cv2.matchTemplate(character_face, face_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_match = face_file.split('.')[0]
    
    character_result = {
        '着順': character_text[0] if len(character_text) > 0 else "不明",
        'キャラクターの衣装': best_match,
        '枠番号': character_text[2] if len(character_text) > 2 else "不明",
        'キャラクターの2つ名': character_text[3].split()[0] if len(character_text) > 3 and len(character_text[3].split()) > 0 else "不明",
        'キャラクター名': character_text[3].split()[1] if len(character_text) > 3 and len(character_text[3].split()) > 1 else "不明",
        'トレーナー名': character_text[3].split()[2] if len(character_text) > 3 and len(character_text[3].split()) > 2 else "不明",
        '作戦': character_text[4].split()[0] if len(character_text) > 4 and len(character_text[4].split()) > 0 else "不明",
        'タイム': character_text[4].split()[1] if len(character_text) > 4 and len(character_text[4].split()) > 1 else "不明",
        '人気': character_text[4].split()[2] if len(character_text) > 4 and len(character_text[4].split()) > 2 else "不明"
    }
    character_results.append(character_result)

df_raceresult = pd.DataFrame(character_results)

# データフレームの表示
print(df_raceinfo)
print(df_raceresult)
