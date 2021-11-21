import torch
# torch.nn : 신경망 모듈 제공(커스텀하여 사용)
import torch.nn as nn
# torch.optim : optimizer 제공(adam, sgd 등)
import torch.optim as optim
# torchvision : 이미지 변환 유틸리티 제공(데이터셋, 데이터 변환 등)
import torchvision
# from torchvision import datasets, models, transforms
from transformer import transformer

import numpy as np
# time : 시간 불러오기 등, 모니터링을 위함
import time

class Trainer():

    def __init__(self):
        # 사용할 장치 지정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # transfer learning을 위해 resnet34 사용
        self.model = torchvision.models.resnet34(pretrained=True)
        # fully connected layer 가 입력받는 feature 갯수를 확인해서,
        # 해당 레이어의 출력 feature 수를 우리의 class 수로 변경한다.
        in_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(in_dim, 3)
        self.model = self.model.to(self.device)
    
        # 다중분류를 위한 손실함수로 CrossEntropyLoss 사용
        # 해당 함수는 내부적으로 클래스별 예측 확률 총합을 1로 만드는 Softmax 함수가 포함되어 있어,
        # 별도로 다중분류를 수행하기 위해 활성화 함수(Softmax)를 지정하지 않아도 된다.
        self.criterion = nn.CrossEntropyLoss()
    
        # 일반적으로 가중치 계산이 많은 딥러닝은 학습 시간 단축을 위해 SGD(확률적 경사 하강법)을 최적화함수(optimizer)로 사용한다.
        # 본 프로그램도 상용 목적이 아니므로 SGD를 사용한다.
        # 최적화함수는 모델의 파라미터를 받아 최적의 가중치(w)를 적용한다.
        # 또한 lr(learning rate)는 낮게 설정하는데, 전이학습의 경우 이미 loss가 크게 줄어있는 상태이므로, 
        # loss를 큰 폭으로 줄일 이유가 없고, underfitting의 가능성이 낮기 때문이다.
        # momentum은 0.9가 최소다. local minima에 빠지지 않도록 학습에 최소한의 관성을 추가한다.
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    
    # def load_dataset(save_path, num_classes):
        
        
    
    # def calculate(num_epochs):
    #     nu

    
    

    

    