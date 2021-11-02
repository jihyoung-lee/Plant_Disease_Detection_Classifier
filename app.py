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


@api.route('/predict')
class Test(Resource):
    def post(self):  # put application's code here

        input_img = request.files['image']
        img = input_img.read()# image 는 키값 value는 파일
        model = Predict()
        res = str(model.predict(img=img))  # split 함수 사용을 위해 tuple 을 str 으로 변경
        cropName, sickNameKor = res.split('_')  # 문자열 나누기
        json_data = {
            'cropName': cropName,
            'sickNameKor': sickNameKor
        }
        return make_response(json.dumps(json_data, ensure_ascii=False))


if __name__ == '__main__':
    app.run(debug=True)
