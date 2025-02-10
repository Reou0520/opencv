# -*- coding: utf-8 -*-
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import sys
import cv2
from scipy import signal
from scipy import fftpack
import time

from pulse_face import Face

capture = cv2.VideoCapture(0)
myFace = Face()

def get_gbr():
    ret, frame = capture.read()
        
    try:
        point = myFace.get_face_point(frame)
        nose = point[30]
        nose_color = frame[nose[1]][nose[0]]
        nose_area = frame[nose[1]-5:nose[1]+5,nose[0]-5:nose[0]+5]

        rsum,gsum,bsum = 0.0,0.0,0.0
        for raster in nose_area:
            for px in raster:
                rsum += px[2]
                gsum += px[0]
                bsum += px[1]
        # 平均値
        ravg = rsum / 100
        gavg = gsum / 100 
        bavg = bsum / 100

        nose_ave = [gavg, bavg, ravg]


    # 顔が取得できなかった時の例外処理
    except:
        print('err')
        return [0,0,0]

    return nose_ave

class PlotGraph:
    def __init__(self):
        # メインウィンドウの設定
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('Pulse Monitor')
        self.central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(self.central_widget)
        
        # メインレイアウトの設定
        self.layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        # カメラ表示用のコンテナウィジェット
        self.camera_container = QtWidgets.QWidget()
        self.camera_layout = QtWidgets.QHBoxLayout()
        self.camera_container.setLayout(self.camera_layout)
        
        # カメラ表示用のラベル
        self.camera_label = QtWidgets.QLabel()
        self.camera_layout.addWidget(self.camera_label)
        
        # グラフウィジェットの設定
        self.plt = pg.PlotWidget(height=80)  # 高さを50から80に変更
        self.plt.setMaximumHeight(80)        # 最大高さも80に変更
        self.plt.setMaximumWidth(200)
        self.plt.setBackground('transparent')
        self.plt.setYRange(0, 255)
        self.plt.showAxis('left', False)
        self.plt.showAxis('bottom', False)
        self.curve_r = self.plt.plot(pen=(255, 0, 0))
        self.curve_g = self.plt.plot(pen=(0, 255, 0))
        self.curve_b = self.plt.plot(pen=(0, 0, 255))
        
        self.plt2 = pg.PlotWidget(height=120)  # 高さを50から80に変更
        self.plt2.setMaximumHeight(120)        # 最大高さも80に変更
        self.plt2.setMaximumWidth(200)
        self.plt2.setBackground('transparent')
        self.plt2.setYRange(0, 255)
        self.plt2.showAxis('left', False)
        self.plt2.showAxis('bottom', False)
        self.curve_g_smg = self.plt2.plot(pen=(0, 255, 255))
        self.curve_g_peak = self.plt2.plot(pen=(0, 255, 255))
        
        # グラフを右上に配置するためのレイアウト
        graph_layout = QtWidgets.QVBoxLayout()
        graph_layout.addWidget(self.plt)
        graph_layout.addWidget(self.plt2)
        graph_layout.addStretch()
        graph_layout.setContentsMargins(5, 5, 5, 5)
        
        # カメララベルにグラフをオーバーレイ
        self.camera_label.setLayout(graph_layout)
        
        # カメラコンテナをメインレイアウトに追加
        self.layout.addWidget(self.camera_container)
        
        # ウィンドウサイズを設定
        self.win.resize(800, 600)
        self.win.show()
        
        # タイマーの設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        
        # データの初期化
        self.data_r = np.zeros((100))
        self.data_g = np.zeros((100))
        self.data_b = np.zeros((100))
        self.data = []
        self.peak_count = 0
        self.last_check_time = time.time()

    def check_pulse_rate(self):
        current_time = time.time()
        if current_time - self.last_check_time >= 10:  # 10秒ごとにチェック
            pulse_rate = self.peak_count * 6  # 1分あたりの脈拍数に換算
            print(f"測定結果:")
            print(f"10秒間のピーク検出数: {self.peak_count}回")
            print(f"推定脈拍数: {pulse_rate}/分")
            if 60 <= pulse_rate <= 96:
                print("判定: 正常")
            elif pulse_rate > 96:
                print("判定: 頻脈")
            else:
                print("判定: 徐脈")
            print("-" * 30)
            
            self.peak_count = 0  # カウントをリセット
            self.last_check_time = current_time

    def update(self):
        ret, frame = capture.read()
        if ret:
            # OpenCV（BGR）からQt（RGB）の形式に変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            scaled_pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
                self.camera_label.size(), 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            self.camera_label.setPixmap(scaled_pixmap)

        # 既存のグラフ更新処理
        self.data_r = np.delete(self.data_r, 0)
        self.data_g = np.delete(self.data_g, 0)
        self.data_b = np.delete(self.data_b, 0)
        gbr = get_gbr()
        
        self.data_r = np.append(self.data_r, gbr[2])
        self.data_g = np.append(self.data_g, gbr[0])
        self.data_b = np.append(self.data_b, gbr[1])
        self.data.append(gbr[0])

        self.curve_r.setData(self.data_r)
        self.curve_g.setData(self.data_g)
        self.curve_b.setData(self.data_b)

        window = 8  # 移動平均の範囲を増やして、ノイズを減らす
        w = np.ones(window)/window
        x = np.convolve(self.data_g, w, mode='same')
        self.curve_g_smg.setData(x)

        N = 100
        threshold = 0.6  # 振幅の閾値を上げて、より顕著なピークのみを検出

        x = np.fft.fft(self.data_g)
        x_abs = np.abs(x)
        x_abs = x_abs / N * 2
        x[x_abs < threshold] = 0

        x = np.fft.ifft(x)
        x = x.real  # 複素数から実数部だけ取り出す
        self.curve_g_smg.setData(x)

        #ピーク値のインデックスを赤色で描画
        maxid = signal.find_peaks(x, 
                             distance=5,     # 最小ピーク間距離を短く（約0.25秒）
                             height=np.mean(x) + 0.3 * np.std(x),  # 平均値+0.3標準偏差に下げてより多くのピークを検出
                             prominence=0.2    # ピークの顕著さの閾値を下げる
                             )[0]
        if len(maxid) > 0:  # ピークが検出された場合
            self.curve_g_peak.setData(maxid, x[maxid], pen=None, symbol='o', 
                                    symbolPen=None, symbolSize=4, symbolBrush=('r'))
            # 検出されたピークの数だけカウントを増やす（変更箇所）
            self.peak_count = len(maxid)  # 赤い点として表示されたピークの数を直接カウント
        else:
            self.curve_g_peak.setData([], [])  # ピークがない場合は表示をクリア
        
        # 脈拍チェック
        self.check_pulse_rate()

# メインの実行部分を修正
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    graphWin = PlotGraph()
    sys.exit(app.exec_())

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()  # QtGuiからQtWidgetsに変更