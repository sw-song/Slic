import io
import json
import os

from PIL import Image
from flask import Flask, jsonify, request

from model.trainer import Trainer
from model.transformer import transformer

app = Flask(__name__)

t = Trainer(pre=True)

# HTTP 요청(POST)으로 받은 img를 PyTorch 모델이 받을 수 있도록 변환
def img_to_tensor(img):
    img = Image.open(img)
    # 받은 image를 (3, 224, 224)로 변환
    tensor_img = t.trans_test(img)
    # 4차원으로 변환 (1, 3, 224, 224)
    tensor_img.unsqueeze_(0)
    return tensor_img

def img_prediction(tensor_img):
    outputs = t.model(tensor_img)
    _, pred_idx = outputs.max(1)
    return t.class_list[pred_idx.item()]




            



    