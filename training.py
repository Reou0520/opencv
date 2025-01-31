import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Flatten, Dense, Dropout, BatchNormalization
) 
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# --- 1. データセットのディレクトリ指定 ---
train_dir = './fer2013/train'      # 訓練データ
validation_dir = './fer2013/test'  # 検証データ

# --- 2. 画像の前処理・データ拡張 ---
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input  # ResNet50推奨の前処理
)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # バッチサイズを16に変更
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# --- 3. ResNet50をベースとした転移学習モデルの構築 ---
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# いったん全層を凍結 (学習しない) しておく
for layer in base_model.layers:
    layer.trainable = False

# --- 4. 転移学習用に上の数層を追加 ---
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7つの感情クラス
])

# --- 5. コンパイル ---
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # 学習率を少し大きく
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. 学習 ---
# EarlyStopping を使って、バリデーション精度が改善しない場合に早期終了
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=10,  # エポック数を10に変更
    validation_data=validation_generator,
    callbacks=[early_stopping]  # EarlyStoppingを追加
)

# --- 7. (オプション) 一部の上位層をアンロックしてファインチューニング ---
# より高精度を狙いたい場合は、ある程度学習が進んだ後に、ResNet50の後半ブロックを数層だけtrainable=Trueにして再学習する
for layer in base_model.layers[-10:]:
    layer.trainable = True  # 最後の10層を学習対象にする

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # 学習率を少し小さく
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=10,  # さらに10エポック学習
    validation_data=validation_generator
)

# --- 8. モデルの保存 ---
model.save('emotion_model.h5')
