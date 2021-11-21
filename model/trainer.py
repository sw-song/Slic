import os
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

class Trainer:

    def __init__(self, data_folder, out_dims, batch_size=4, shuffle=True, num_epochs=16):
        # 사용할 장치 지정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # transfer learning을 위해 resnet34 사용
        self.model = torchvision.models.resnet34(pretrained=True)
        # fully connected layer 가 입력받는 feature 갯수를 확인해서,
        # 해당 레이어의 출력 feature 수를 우리의 class 수로 변경한다.
        in_dims = self.model.fc.in_features
        self.model.fc = nn.Linear(in_dims, out_dims)
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # 모델 학습을 위한 데이터 전처리 객체(Compose) 저장
        self.trans_train, self.trans_test = transformer()
        
        # 데이터 폴더 지정
        cwd = os.getcwd()
        self.data_path = os.path.join(cwd, data_folder)
        
        # 학습에 필요한 파라미터들
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = shuffle

    def load_dataset(self):

        # ---- 특정 폴더를 지정하고, 데이터 전처리를 통해 데이터를 불러오는 구간 ----
        print("[Info] 데이터 로드 중 ..")
        from_dataset = torchvision.datasets.ImageFolder()

        train_datasets = from_dataset(
            os.path.join(self.data_path, 'train'),
            self.trans_train
       )

        test_datasets = from_dataset(
            os.path.join(self.data_path, 'test'), 
            self.trans_test
        )
        print(f"[Info] 데이터 로드 완료")
        print(f"\t- train set : {train_datasets.classes}")
        print(f"\t- test set : {test_datasets.classes}")
        # ----------------------------------------------------------
    
        # ---- 데이터들을 미니배치로 모델에 전달하기 위해 dataloader api 사용 ----
        dataloader = torch.utils.data.DataLoader()
        
        train_dataloader = dataloader(
            train_datasets,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        test_dataloader = dataloader(
            test_datasets,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        return (train_dataloader, test_dataloader)

    
    def calculate(self):
        
        # 모델 레이어들에 대해 train 모드를 적용한다
        # 모드는 train 혹은 evel로 설정 가능하며,
        # eval 모드로 설정하는 경우 dropout, batchnorm layer 등을 제한한다.
        self.model.train()

        # (모니터링)학습 시간 측정을 위해 학습 시작 시간을 기록한다.
        start_time = time.time()

        train_dataloader, test_dataloader = self.load_dataset()

        # epoch(default : 16)을 1씩 증가하며 순회한다.       
        for epoch in range(self.num_epochs):
            loss = 0.

            # mini-batch(default : 4)를 1씩 증가하며 순회한다.
            for batch_imgs, batch_labels in train_dataloader:
                # tensor로 표현된 이미지를 batch size(개) 만큼 device에 올림
                inputs = batch_imgs.to(self.device)
                # 배치 내 각 이미지에 대한 label tensor를 device에 올림
                labels = batch_labels.to(self.device)

                # 최적화함수를 초기화(매 배치마다)
                self.optimizer.zero_grad()

                # 모델에 inputs(이미지 텐서)를 넣으면 각 이미지가 각각의 label일 확률을 반환한다.
                # 즉, label이 [a,b,c]가 있을 때, a에 해당하는 이미지가 a일 확률, b일 확률, c일 확률을 반환한다.
                outputs = self.model(inputs)
                
                # torch.max 메서드는 첫번째 인자로 최대값의 인덱스를, 두번째 인자로 최대값을 반환한다.
                # 여기서는 '최대값'만 필요하므로 인덱스는 무시한다.
                # 또한, 최대값은 batch 수만큼 넘어온다. 이미지 4개를 받았으면 최대값도 4개다.
                _, preds = torch.max(outputs, 1)

                # 아래 손실함수는 내부적으로 softmax 활성화 함수가 사용하므로,
                # 전달받은 outputs들의 총합을 1이 되도록 각 확률값을 변환한 다음 labels와 비교한다.
                # labels는 원-핫 형태의 텐서를 총 batch-size만큼 가지고 있다.
                loss = self.criterion(outputs, labels)




        

    
    

    

    