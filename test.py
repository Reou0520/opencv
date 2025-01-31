from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# フレームの作成
frame = np.zeros((500, 500, 3), dtype=np.uint8)

# 日本語フォントの指定
font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 30)

# 文字を描画
img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
draw.text((50, 50), "こんにちは", font=font, fill=(255, 255, 255))

# フレームをOpenCVで表示
frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cv2.imshow("Text Test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
