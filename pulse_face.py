# -*- coding: utf-8 -*-
import cv2
import dlib
from imutils import face_utils


class Face():
  def __init__(self):
    self.detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し.ただ顔だけを検出する
    self.load_model()
  
  def load_model(self, path="./model/shape_predictor_68_face_landmarks.dat"):
    try:
      self.predictor = dlib.shape_predictor(path) #顔から目鼻などランドマークを出力する
    except Exception as e:
      raise ValueError('can not load predict model.')

  def get_face_point(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #gray scaleに変換する
    rects = self.detector(gray, 0) #grayから顔を検出

    if not rects:
      return 

    else:
      #顔が認識できればポイントを特定
      shape = self.predictor(gray, rects[0])
      shape = face_utils.shape_to_np(shape)
      return shape