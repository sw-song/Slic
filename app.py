import io
import json
import os

from PIL import Image
from flask import Flask, jsonify, request
from flask.templating import render_template
from model import predictor

app = Flask(__name__)
app.config['UPLOAD'] = 'static/uploads'

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_name = file.filename 
            if '.' in file_name and file_name.rsplit('.', 1)[1] in ['jpg', 'png', 'jpeg']:
                src_path = os.path.join(app.config['UPLOAD'], file_name)
                file.save(src_path)
                pred = predictor.img_prediction(predictor.img_to_tensor(file))
                return render_template('index.html', src=src_path, pred=str(pred))
            return render_template('index.html', pred="파일 형식을 jpg, png, jpeg로 맞춰주세요.")
        return render_template('index.html', pred="파일 에러")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
            



    