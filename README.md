# UmaRoomaGetter
このアプリはウマ娘のルームマッチのレース結果画面をキャプチャしてレース情報と着順情報を読み取って記録します。
精度は微妙です。枠番やランクは壊滅的です。が、ひとまずリリースします。多分誰かが改良してくれる。
This App is get and records the results of Umamusume's room matches.
![アプリ動作デモ](https://github.com/user-attachments/assets/b33efdb7-518d-400e-8d5a-d136c4e6c970)
![レース情報CSV](https://github.com/user-attachments/assets/ca9bd2a9-42c4-4b61-ad66-2a1c201cd1b8)
![着順情報CSV](https://github.com/user-attachments/assets/7a8c1811-239a-49f3-857a-c03401054d5e)

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
1. run.batをダブルクリックする（必要なモジュールがダウンロードされ仮想環境が作られます）
2. アプリ「UmaRoomaGetter」が機動する
3. ウマ娘でルームマッチのリザルト画面まで移動する
4. 着順情報をスクロールする（速いと精度が落ちます）
5. アプリにレース情報と着順情報が入ったことを確認する
6. アプリ上のボタン「登録」を押す
7. アプリの実行ファイル（app.py）と同じ場所にレース情報と着順情報が記録されたCSVファイルが作られて情報が入っていることを確認する
8. 3以降を繰り返す
※CSVファイルは肥大化するので、大きくなったら削除してください。
※レース情報と着順情報はCSVファイル内の列「race_id」で紐付けることが出来ます。

## テンプレート画像の増やし方
* フォルダ「images」の各フォルダに検知された場合の名称を付けた画像ファイルを入れてください。
* 画像のサイズはゲーム画面のスクショをタイトルバーも含めて取得した再に高さが1,000pxになる状態のサイズが良いです。
* アプリの高さを1,000pxにしてあるので、そこに合わせてゲーム画面のサイズを調整すると良いと思います。

## FAQ
|質問|回答|
|-----------|-----------|
|「cv::resize」のエラーが出る|ゲーム画面のサイズを少し変えて再度実行してみてください|
|出走していない着順が表示される|仕様です。2桁着順は変化が小さいの16着と18着などは誤検知されがちです|
|検知されるキャラが少ない|「images/chara_faces」にキャラ名を付けたキャラ画像を追加してください|
|検知される育成ランクが少ない|「images/chara_ranks」に育成ランク名を付けた育成ランク画像を追加してください|

## アップデート方法
update.batをダブルクリックするだけで完了します。
（git pullしているだけです）

## 機能
- [x] **一時停止**：データ取得の処理を一時停止します。CPU負荷が下がるのと結果を固定できます。
- [x] **再開**：データ取得の処理を再開します。
- [x] **登録**：取得したデータをCSVファイルに追加します。
- [x] **消去**：取得したデータをクリアします。
- [ ] （未実装）ログ出力：必要だなと判断したらloggerで出せるようにします。
- [ ] （未実装）設定値のYAML化：必要だなと判断したら実装します。
- [ ] （未実装）ローカルDB化：需要あれば考えます。sqlite3で良いよねとは思っている。
- [ ] （未実装）クラウドDB化：需要あれば考えます。BigQueryに貯めて集計とか面白そうではある。

## ライセンス
AGPL-3.0

## 開発者
[aburahamu](https://twitter.com/aburahamu_aa)
改良できましたらプルリクお願いします。

## 謝辞
ハル：協力ありがとう。