# -*- coding: utf-8 -*-

from flask import Flask, request, redirect, url_for, render_template, flash, session, Response
import cv2
import numpy as np
import mysql.connector
import pickle
from cv2.face import LBPHFaceRecognizer_create
import requests
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os.path
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from pytz import timezone
import numpy as np


app = Flask(__name__)
app.secret_key = 'あなたの秘密鍵'  # 必ずランダムで秘密の値を使用してください


# MySQL接続設定
def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='face_recognition'
    )
    return conn


# 顔データのリサイズ
def resize_face(face, size=(200, 200)):
    if face is not None and face.size > 0:
        try:
            return cv2.resize(face, size)
        except Exception as e:
            print(f"顔のリサイズエラー: {e}")
    return None


# Google Calendar APIの認証とスケジュールの取得
def get_google_calendar_events():
    SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_609420033632-a6vpqqsfhkqm91haelki3upcni796soi.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('calendar', 'v3', credentials=creds)
        japan_tz = timezone('Asia/Tokyo')
        now = datetime.now(japan_tz).isoformat()  # 現在の日付と時間
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=10, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])
    except Exception as e:
        print(f"Google Calendar APIエラー: {e}")
        return ['スケジュール取得エラー']

    event_list = []
    if not events:
        event_list.append('今日の予定はありません。')

    # 今日の日付のイベントのみを表示
    today = datetime.now(japan_tz).date()  # 今日の日付
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        start_datetime = datetime.fromisoformat(start).astimezone(japan_tz)

        # 今日の日付と一致するイベントを表示
        if start_datetime.date() == today:
            formatted_time = start_datetime.strftime('%Y年%m月%d日 %H:%M')
            event_list.append(f"{formatted_time}: {event['summary']}")

    # 今日の予定がない場合
    if not event_list:
        event_list.append('今日の予定はありません。')

    return event_list


# OpenWeatherMap APIから天気予報を取得
def get_weather(city):
    api_key = "cbd8eabd537c70b4101eeae3561df4b0"  # OpenWeatherMapのAPIキーを設定
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ja"
    try:
        response = requests.get(url)
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        icon_code = data['weather'][0]['icon']
        weather_icon_url = f"http://openweathermap.org/img/wn/{icon_code}.png"
        return f"天気: {weather_description}, 気温: {temperature}°C", weather_icon_url
    except Exception as e:
        print(f"天気予報取得エラー: {e}")
        return '天気情報取得エラー', None


# 文字化け対策
def add_text_to_frame(frame, text, position, font_size=25, background_opacity=0.3):
    try:
        # 画像をRGBAモードに変換してアルファチャンネルを使用可能にする
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        draw = ImageDraw.Draw(img_pil)

        # フォントのパスとサイズを指定
        font_path = r"C:/Users/Reoh/OneDrive/opencv_faces/NotoSansJP-VariableFont_wght.ttf"
        font = ImageFont.truetype(font_path, font_size)

        # テキストのサイズを計算（textsizeの代わりにtextbboxを使用）
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # 左から右の幅
        text_height = text_bbox[3] - text_bbox[1]  # 上から下の高さ

        # 背景矩形の描画位置とサイズ
        rect_x1, rect_y1 = position
        rect_x2 = rect_x1 + text_width + 15  # テキストの幅に少し余白を加える
        rect_y2 = rect_y1 + text_height + 15  # テキストの高さに少し余白を加える

        # 背景矩形の描画 (薄い黒、アルファチャンネルを含む)
        draw.rectangle([(rect_x1, rect_y1), (rect_x2, rect_y2)], fill=(0, 0, 0, int(255 * background_opacity)))

        # テキストを描画
        draw.text(position, text, font=font, fill=(255, 255, 255, 255))  # 文字の色（白）

        # BGR形式に戻す
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
    except Exception as e:
        print(f"テキスト描画エラー: {e}")
    return frame


# キャッシュのためのアイコン画像
cached_icon_image = None
def get_icon_image(icon_url, size=(100, 100)):
    global cached_icon_image
    if cached_icon_image is None:
        try:
            icon_response = requests.get(icon_url)
            icon_array = np.asarray(bytearray(icon_response.content), dtype=np.uint8)
            icon_image = cv2.imdecode(icon_array, cv2.IMREAD_UNCHANGED)  # アルファチャンネルも読み込む

            if icon_image.shape[2] == 4:  # 4チャンネル（アルファチャンネルあり）
                # アルファチャンネルを使って透明部分を処理
                alpha_channel = icon_image[:, :, 3]  # アルファチャンネル
                rgb_channels = icon_image[:, :, :3]  # RGBチャンネル

                # アルファチャンネルを背景色（黒）で置き換える
                white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
                alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
                icon_image = cv2.convertScaleAbs(rgb_channels * alpha_factor + white_background * (1 - alpha_factor))

            # アイコンサイズをリサイズ
            cached_icon_image = cv2.resize(icon_image, size)
        except Exception as e:
            print(f"アイコン画像の取得エラー: {e}")
            return None
    return cached_icon_image


# 天気アイコン
def add_icon_to_frame(frame, icon_url, position, border_color=(255, 0, 0), border_thickness=2):
    try:
        icon_image = get_icon_image(icon_url)
        if icon_image is not None:
            icon_height, icon_width = icon_image.shape[:2]
            y1, y2 = position[1], position[1] + icon_height
            x1, x2 = position[0], position[0] + icon_width

            # アイコンの描画
            if y2 <= frame.shape[0] and x2 <= frame.shape[1]:  # フレームサイズ内にアイコンが収まるか確認
                frame[y1:y2, x1:x2] = icon_image

                # 枠線を追加（枠線の色と太さを指定）
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

    except Exception as e:
        print(f"アイコン画像の描画エラー: {e}")
    return frame


# モデルのロード
model = load_model('emotion_model.h5')

# 顔検出用のカスケード分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 感情ラベル
emotion_labels = ['怒り', '嫌悪', '恐怖', '悲しみ', '幸福', '驚き', '中立']

def recognize_emotion(frame):
    emotion = "未検出"  # 顔が検出されない場合のデフォルト値
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  # 48x48にリサイズ
        
        # ResNet50が期待する224x224のサイズに変更
        face_resized = cv2.resize(face_resized, (224, 224))  # 224x224にリサイズ
        face_array = np.array(face_resized) / 255.0  # 正規化
        face_array = np.expand_dims(face_array, axis=0)  # バッチ次元
        face_array = np.expand_dims(face_array, axis=-1)  # チャンネル次元

        # 感情認識
        emotion_pred = model.predict(face_array)
        emotion_idx = np.argmax(emotion_pred)
        emotion = emotion_labels[emotion_idx]

        # 顔枠を描画したい場合はそのままOpenCVでOK（枠は文字化けしない）
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ここでOpenCVの putText ではなく、PILベースの add_text_to_frame を使う
        frame = add_text_to_frame(frame, emotion, (x, y - 30))

    return frame, emotion


# ---カメラ画面---に天気とスケジュールを表示
FPS = 15  # フレームレートを15fpsに制限
frame_interval = 1 / FPS

# 健康記録を10秒ごとに更新
def save_health_record(user_name, emotion, health_status):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO health_records (name, emotion, health_status) VALUES (%s, %s, %s)',
                       (user_name, emotion, health_status))  # INSERT文を修正
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"健康記録の保存エラー: {e}")


# 健康記録を10秒ごとに更新
last_save_time = time.time()  # 初期化はグローバルスコープで行う
save_interval = 10  # 保存間隔（秒）

def generate_frames_with_info(user_name=None):
    global last_save_time  # グローバル変数を使用するために宣言

    city = "Tokyo"
    weather_info, weather_icon_url = get_weather(city)
    schedule_info = get_google_calendar_events()

    last_update_time = time.time()
    update_interval = 60  # 60秒ごとに情報を更新

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    y_schedule = 90  # スケジュールの開始位置を設定
    y_health = y_schedule + 50  # 健康状態の位置はスケジュールの下に設定

    while True:
        success, frame = video_capture.read()
        if not success:
            print("カメラの読み込みに失敗しました")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()
        if current_time - last_update_time > update_interval:
            weather_info, weather_icon_url = get_weather(city)
            schedule_info = get_google_calendar_events()
            last_update_time = current_time

        try:
            frame = add_text_to_frame(frame, weather_info, (10, 30))
        except Exception as e:
            print(f"天気情報の描画エラー: {e}")

        if weather_icon_url:
            try:
                frame = add_icon_to_frame(frame, weather_icon_url, (frame.shape[1] - 100 - 10, 10), border_color=(255, 0, 0), border_thickness=2)
            except Exception as e:
                print(f"アイコンの描画エラー: {e}")

        # スケジュール情報の描画
        for item in schedule_info:
            try:
                frame = add_text_to_frame(frame, item, (10, y_schedule))
            except Exception as e:
                print(f"スケジュール情報の描画エラー: {e}")

        # 健康状態の表示
        frame, emotion = recognize_emotion(frame)
        health_status = "健康状態: 良好" if emotion == "幸福" else "健康状態: "
        frame = add_text_to_frame(frame, health_status, (10, y_health))

        # 10秒ごとに健康状態と感情を保存
        if user_name and current_time - last_save_time > save_interval:
            save_health_record(user_name, emotion, health_status)
            last_save_time = current_time  # 最後に保存した時刻を更新

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("フレームのエンコードに失敗しました")
                continue
            frame = buffer.tobytes()
        except Exception as e:
            print(f"エンコードエラー: {e}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


# カメラ画面 
def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # カメラ映像をJPEG形式にエンコード
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # MIME形式でストリーミング
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_info')
def video_feed_info():
    # ログインしているユーザー名を取得
    user_name = session.get('user')
    
    # ユーザー名が存在する場合のみ処理を進める
    if user_name:
        # user_nameをgenerate_frames_with_infoに渡して呼び出す
        return Response(generate_frames_with_info(user_name=user_name), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # ログインしていない場合はログインページにリダイレクト
        flash('ログインしてください')
        return redirect(url_for('login'))


# 顔登録ページ
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        video_capture = cv2.VideoCapture(0)

        face_data_list = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("画像のキャプチャに失敗しました")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            print(f"検出された顔の数: {len(faces)}")

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = resize_face(face)
                if face_resized is not None:
                    face_data_list.append(face_resized)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.imshow('Face Registration', frame)
                    cv2.waitKey(1000)  # 1秒間表示

            if len(face_data_list) > 0:
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        if len(face_data_list) > 0:
            face_data_list_resized = [resize_face(face) for face in face_data_list if resize_face(face) is not None]
            if len(face_data_list_resized) > 0:
                # 顔データをNumPyとして保存
                np.save(f"face_data_{name}.npy", np.array(face_data_list_resized))

                # MySQLに顔データのパスを保存
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('INSERT INTO users (name, face_data) VALUES (%s, %s)', 
                               (name, f"face_data_{name}.npy"))
                conn.commit()
                conn.close()
                return redirect(url_for('index'))

    return render_template('register.html')


# 顔認識の設定
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        video_capture = cv2.VideoCapture(0)

        detected_faces = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("画像のキャプチャに失敗しました")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            print(f"検出された顔の数: {len(faces)}")

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = resize_face(face)
                if face_resized is not None:
                    detected_faces.append(face_resized)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.imshow('Face Detection', frame)
                    cv2.waitKey(1000)  # 1秒間表示

            if len(detected_faces) > 0:
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

        if len(detected_faces) > 0:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT name, face_data FROM users')
            users = cursor.fetchall()
            conn.close()

            recognizer = LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

            face_labels = []
            face_images = []

            for name, face_data_path in users:
                try:
                    # NumPy配列として顔データを読み込む
                    stored_faces = np.load(face_data_path)  # NumPyデータの読み込み
                    face_labels.extend([name] * len(stored_faces))
                    face_images.extend(stored_faces)
                except Exception as e:
                    print(f"顔データの読み込みエラー: {e}")
                    continue    

            if len(face_images) > 0:
                face_labels_encoded = list(set(face_labels))
                labels = np.array([face_labels_encoded.index(label) for label in face_labels])

                recognizer.train(face_images, labels)
                
                for detected_face in detected_faces:
                    try:
                        label, confidence = recognizer.predict(detected_face)
                        print(f"認識結果: ラベル={label}, 信頼度={confidence}")

                        if confidence < 100:  # 信頼度の閾値を調整
                            user_name = face_labels_encoded[label]
                            session['user'] = user_name
                            flash(f'ログイン成功: {user_name}')
                            return redirect(url_for('index'))

                    except Exception as e:
                        print(f"顔認識中のエラー: {e}")

            flash('ログイン失敗')
    
    return render_template('login.html')


# ログアウト処理
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('ログアウトしました')
    return redirect(url_for('index'))


@app.route('/smart_mirror')
def smart_mirror():
    return render_template('smart_mirror.html')


@app.route('/health_record')
def health_record():
    user_name = session.get('user')
    if user_name:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT emotion, health_status, updated_at FROM health_records WHERE name = %s ORDER BY updated_at DESC', (user_name,))
        records = cursor.fetchall()
        conn.close()
        return render_template('health_record.html', user=user_name, records=records)
    else:
        flash('ログインしてください')
        return redirect(url_for('login'))


# ホームページ
@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)


if __name__ == '__main__':
    app.run(debug=True)