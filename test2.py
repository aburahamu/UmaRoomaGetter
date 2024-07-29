import cv2
import pandas as pd
import pytesseract
import pyautogui
from PIL import Image
import numpy as np
import os

# Tesseractのパスを設定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ゲーム画面のキャプチャー
def capture_game_window(window_name):
    window = pyautogui.getWindowsWithTitle(window_name)[0]
    window.activate()
    screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    screenshot.save('screenshot.png')
    return 'screenshot.png'

# レース概要の情報を取得
def get_race_info(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    race_info_region = image[int(height*0.4):int(height*0.5), :]
    race_info_text = pytesseract.image_to_string(race_info_region, lang='jpn')
    race_info_lines = race_info_text.split('\n')
    race_info = {
        'レースの格': race_info_lines[0].split()[0],
        'レース名': race_info_lines[0].split()[1],
        '場所': race_info_lines[1].split()[0],
        '馬場': race_info_lines[1].split()[1],
        '距離': race_info_lines[1].split()[2],
        '距離名': race_info_lines[1].split()[3],
        '周回方向': race_info_lines[1].split()[4],
        '天候': race_info_lines[1].split()[5],
        '馬場の状態': race_info_lines[1].split()[6],
        '時間帯': race_info_lines[1].split()[7]
    }
    return pd.DataFrame([race_info])

# キャラクターのレース結果を取得
def get_race_results(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    race_results_region = image[int(height*0.5):, :]
    race_results_text = pytesseract.image_to_string(race_results_region, lang='jpn')
    race_results_lines = race_results_text.split('\n')
    race_results = []
    for i in range(0, len(race_results_lines), 5):
        result = {
            '着順': race_results_lines[i],
            'キャラクターのランク': race_results_lines[i+1],
            '枠番号': race_results_lines[i+2],
            'キャラクターの2つ名': race_results_lines[i+3],
            'キャラクター名': race_results_lines[i+4],
            'トレーナー名': race_results_lines[i+5],
            '作戦': race_results_lines[i+6],
            'タイム': race_results_lines[i+7],
            '人気': race_results_lines[i+8]
        }
        race_results.append(result)
    return pd.DataFrame(race_results)

# メイン処理
def main():
    image_path = capture_game_window('umamusume')
    df_raceinfo = get_race_info(image_path)
    df_raceresult = get_race_results(image_path)
    print(df_raceinfo)
    print(df_raceresult)

if __name__ == '__main__':
    main()
