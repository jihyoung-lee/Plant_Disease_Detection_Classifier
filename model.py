import io
from PIL import Image
<<<<<<< Updated upstream
from keras.model import load_model
=======
from keras.models import load_model
>>>>>>> Stashed changes
import numpy as np


class Predict:
    def predict(self, img):
        global class_name
        model = load_model('mobilenet.h5')  # 모델 load

        input_image = img
        raw_image = Image.open(io.BytesIO(input_image)).convert('RGB')
        img = np.asarray(raw_image)
        img = Image.fromarray(img).resize((224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255

        prediction = model.predict(img)
        d = prediction.flatten()
        j = d.max()  # 신뢰도값이 가장 큰 클래스 값
        # 클래스 이름이 영어로 되어있어서 정확한 번역이 필요
        li = ['사과_검은별무늬병', '사과_검은썩음병', '사과_붉은별무늬병', '사과_건강함', '블루베리_건강', '체리_흰가루병', '체리_건강', '옥수수_잎마름병',
              '옥수수_녹병', '옥수수_그을음무늬병', '옥수수_건강함', '포도_검은썩음병', '포도_홍역', '포도_그을음무늬병', '포도_건강', '오렌지_황룡병', '복숭아_세균성구멍병',
              '복숭아_건강',
              '후추_세균성점무늬병', '후추_건강', '감자_겹무늬병', '감자_역병', '감자_건강', '라즈베리_건강', '콩_건강', '호박_흰가루병', '딸기_그을음병', '딸기_건강',
              '토마토_세균성점무늬병', '토마토_겹무늬병',
              '토마토_역병', '토마토_잎곰팡이병', '토마토_반점위조바이러스', '토마토_점박이응애', '토마토_갈색무늬병', '토마토_황화잎말림바이러스', '토마토_모자이크병', '토마토_건강']
        for index, item in enumerate(d):  # 인덱스 추출하는 함수
            if item == j:
                class_name = li[index]
        return class_name
