import io
from PIL import Image
from keras.models import load_model
import numpy as np
import tensorflow as tf


class Predict:

    def model_load(self):
        model = load_model('mobilenet.h5')  # 모델 load

        return model

    def prepare_img(self, img, target):
        img = Image.open(io.BytesIO(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target)
        img = img.to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        return img

    def predict(self, img):
        global class_name

        # input_image = img
        # raw_image = Image.open(io.BytesIO(input_image)).convert('RGB')
        # img = np.asarray(raw_image)
        # img = Image.fromarray(img).resize((224, 224))
        # img = np.expand_dims(img, axis=0)
        # img = img / 255

        model = self.model_load()
        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])
        prediction = probability_model.predict(img)

        # 클래스 이름이 영어로 되어있어서 정확한 번역이 필요
        li = ['사과_검은별무늬병', '사과_가지검은마름병 ', '사과_붉은별무늬병', '사과_정상', '체리_흰가루병', '체리_정상', '옥수수_잎마름병',
              '옥수수_녹병', '옥수수_둥근무늬병', '옥수수_정상', '포도_꼭지마름병', '포도_갈색무늬병', '포도_그을음무늬병', '포도_정상',  '복숭아_세균성구멍병',
              '복숭아_정상',
              '후추_세균성점무늬병', '후추_정상', '감자_겹둥근무늬병', '감자_역병', '감자_정상', '딸기_그을음병', '딸기_정상',
              '토마토_흰무늬병', '토마토_겹무늬병',
              '토마토_잎마름역병', '토마토_잎곰팡이병', '토마토_반점위조바이러스', '토마토_점박이응애', '토마토_갈색무늬병', '토마토_황화잎말림바이러스', '토마토_모자이크병', '토마토_정상']

        d = prediction.flatten()
        j = d.max()  # 신뢰도값이 가장 큰 클래스 값

        confidence = 100 * np.max(prediction[0])

        for index, item in enumerate(d):  # 인덱스 추출하는 함수
            if item == j:
                class_name = li[index]

        return class_name, confidence
