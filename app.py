import io
import json
import os
import argparse

from PIL import Image
from flask import Flask, jsonify, request
from flask.templating import render_template
from model import predictor
from model import trainer
from data import creater

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
            return render_template('index.html', pred="파일 형식은 jpg, png, jpeg로 맞춰주세요.")
        return render_template('index.html', pred="파일을 선택해주세요.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--init', type=bool, default=False)
    parser.add_argument('-c', '--class_name', type=str, nargs='+', default=None)
    parser.add_argument('-ni', '--num_imgs', type=int, default=50)
    parser.add_argument('-sf', '--save_folder', type=str, default="datasets")
    parser.add_argument('-l', '--limit_time', type=int, default=10)
    parser.add_argument('-f', '--force_replace', type=bool, default=False)
    parser.add_argument('-t', '--train', type=bool, default=True)
    parser.add_argument('-ts', '--train_size', type=int, default=40)
    parser.add_argument('-p', '--pre', type=bool, default=False)
    parser.add_argument('-m', '--model_path', type=str, default="./")
    parser.add_argument('-df', '--data_folder', type=str, default="datasets")
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-s', '--shuffle', type=bool, default=True)
    parser.add_argument('-ne', '--num_epochs', type=int, default=16)

    args = parser.parse_args()
    if args.init and args.class_name:
        for i in range(len(args.class_name)):
            creater.create_dataset(args.class_name[i],
                args.num_imgs,
                args.save_folder,
                args.limit_time,
                args.force_replace,
                args.train,
                args.train_size)

        t = trainer.Trainer(
            args.pre,
            args.model_path,
            args.data_folder,
            args.batch_size,
            args.shuffle,
            args.num_epochs
        )
        
        print("#################################")
        print("[Info] Auto run - training start")
        if args.pre == True:
            t.run(pre=True)
        else:
            t.run()
            
        print("##################################")
        print("[Info] Auto run - evaluation start")
        t.test()
        app.run(host='0.0.0.0', port=8888, debug=True)

    elif args.init and (args.class_name is None):
        print("초기화를 위해서 class name을 지정해주세요.")
    
    else:
        app.run(host='0.0.0.0', port=8888, debug=True)
            