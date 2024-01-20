from tensorflow.keras.models import load_model
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

classes = os.listdir("/Users/yoheiyamaguchi/GeekSalon/products/train_products")
model = load_model("ftmodel.h5")

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image


# 認識させたい画像の読み込み
input_filename = input("画像のパスを入力してください")

input_image = image.load_img(input_filename, target_size=(224, 224))

# 画像の前処理
input_image = image.img_to_array(input_image)

input_image = np.expand_dims(input_image, axis=0)

input_image = preprocess_input(input_image)

results = model.predict(input_image)


# 2個の候補を順番に出力
for decode_result in results:
    print(decode_result)
label = classes[np.argmax(results[0])]
print(label)

probability = results[0][np.argmax(results[0])] * 100 + "%"
print(probability)