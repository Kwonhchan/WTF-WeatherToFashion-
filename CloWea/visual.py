import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model

model_path = r'Models\outer\best_model_30_1.25_0.43.keras'

# 모델 파일이 존재하는지 확인
if not os.path.exists(model_path):
    raise ValueError(f"File not found: {model_path}")

# .keras 파일로부터 모델 불러오기
model = tf.keras.models.load_model(model_path)

# 모델 구조 시각화
plot_model(model, to_file='bottom_model_structure.png', show_shapes=True, show_layer_names=True, dpi=600)

# 이미지의 크기와 해상도를 모두 조정
fig = plt.figure(figsize=(20, 20))  # 크기 조정
img = mpimg.imread('bottom_model_structure.png')
imgplot = plt.imshow(img)
plt.axis('off')  # 축 제거
plt.show()
