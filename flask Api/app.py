import json

from flask import Flask, request, make_response
from flask_restx import Api, Resource
from model import Predict

app = Flask(__name__)
api = Api(app,
          version='0.1',
          title="병해충판별 API Server",
          description="Mobilenet",
          terms_url="/predict",
          contact="wlgud3412@naver.com")

app.config['UPLOAD_FOLDER'] = 'uploaded\\image'
model = Predict()


@api.route('/predict')
class Test(Resource):
    def post(self):  # put application's code here

        input_img = request.files['image']
        img = input_img.read()  # image 는 키값 value는 파일
        res = model.predict(img=img)
        label = str(res[0])  # split 함수 사용을 위해 tuple 을 str 으로 변경
        cropName, sickNameKor = label.split('_')  # 문자열 나누기
        confidence = '{:2.0f}'.format(res[1])
        confidence = int(confidence)
        json_data = {

            'cropName': cropName,
            'sickNameKor': sickNameKor,
            'confidence': confidence
        }
        return make_response(json.dumps(json_data, ensure_ascii=False))


if __name__ == '__main__':
    model.model_load()
    app.run(host='0.0.0.0',port=80)
