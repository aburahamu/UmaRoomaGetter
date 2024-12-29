# UmaRoomaGetter
このアプリはウマ娘のルームマッチのレース結果画面をキャプチャしてレース情報と着順情報を読み取って記録します。
精度は微妙です。枠番やランクは壊滅的です。が、ひとまずリリースします。多分誰かが改良してくれる。
This App is get and records the results of Umamusume's room matches.

## インストール方法
**前提**
- Pythonをインストールしておいてください。推奨バージョンは3.10.6です。
- GITをインストールしておいてください。推奨バージョンは2.43.0です。
**開発環境**
- OS：Windows11 Pro 24H2
- Pythonバージョン：3.10.6
- Gitバージョン：2.43.0
- GPU：Geforce RTX4080
- エディタ：VisualStudioCode
**手順**
* この画面の上部にある緑色のボタン「<> Code」をクリック
* HTTPSのURLをコピー
* アプリをインストールしたいフォルダをエクスプローラーで開く
* パス欄を消して「cmd」と入力しエンターキーを押す（コマンドプロンプトが開く）
* コマンドプロンプトに「git 」（最後に半角スペース）と入力し、コピーしたURLを張り付けて実行（エンターキーを押す）
* 完了です
## 使い方
* run.batをダブルクリックする（必要なモジュールがダウンロードされ仮想環境が作られます）
* アプリ「UmaRoomaGetter」が機動する
* ウマ娘でルームマッチのリザルト画面まで移動する
* 着順情報をスクロールする（速いと精度が落ちます）
* アプリにレース情報と着順情報が入ったことを確認する
* アプリ上のボタン「登録」を押す
* アプリの実行ファイル（app.py）と同じ場所にレース情報と着順情報が記録されたCSVファイルが作られて情報が入っていることを確認する
* 以降はルームマッチの結果表示と着順情報のスクロールと結果登録を繰り返して良い。
## アップデート方法
update.batをダブルクリックするだけで完了します。
（git pullしているだけです）
## 機能
- **一時停止**：データ取得の処理を一時停止します。CPU負荷が下がるのと結果を固定できます。
- **再開**：データ取得の処理を再開します。
- **登録**：取得したデータをCSVファイルに追加します。
- **消去**：取得したデータをクリアします。
- ログ出力（未実装）：必要だなと判断したらloggerで出せるようにします。
- 設定値のYAML化（未実装）：必要だなと判断したら実装します。
## ライセンス
AGPL-3.0
## 開発者
[aburahamu](https://twitter.com/aburahamu_aa)
改良できましたらプルリクお願いします。
## 謝辞
ハル：協力ありがとう。