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
import copy

import argparse

class Trainer:

    def __init__(self, pre=False, model_path="./", data_folder="datasets", batch_size=4, shuffle=True, num_epochs=16):

        # 데이터 폴더 지정
        cwd = os.getcwd()
        self.data_path = os.path.join(cwd, data_folder)
        self.class_list = os.listdir(f"{self.data_path}/train")
        self.class_list = [f for f in self.class_list if not '.' in f]
        
        out_dims = len(self.class_list)

        # 사용할 장치 지정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 훈련된 모델 위치
        self.model_path = model_path

        
        if pre:
            self.model = torch.load(self.model_path + "model.pt")
        else:
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
        
        # 학습에 필요한 파라미터들
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = shuffle

    def load_dataset(self):

        # ---- 특정 폴더를 지정하고, 데이터 전처리를 통해 데이터를 불러오는 구간 ----
        print("[Info] 데이터 로드 중 ..")
        train_datasets = torchvision.datasets.ImageFolder(
            os.path.join(self.data_path, 'train'),
            self.trans_train
       )

        test_datasets = torchvision.datasets.ImageFolder(
            os.path.join(self.data_path, 'test'), 
            self.trans_test
        )
        print(f"[Info] 데이터 로드 완료")
        print(f"\t- train set : {train_datasets.classes}")
        print(f"\t- test set : {test_datasets.classes}")
        # ----------------------------------------------------------
    
        # ---- 데이터들을 미니배치로 모델에 전달하기 위해 dataloader api 사용 ----
        train_dataloader = torch.utils.data.DataLoader(
            train_datasets,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_datasets,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        return (train_dataloader, len(train_datasets), 
                test_dataloader, len(test_datasets))

    
    def run(self, pre=False):
        
        # 최초 학습이 아닌 경우, 학습된 모델을 가져옴
        if pre:
            self.model = torch.load(self.model_path + "model.pt")
            
        # 모델 레이어들에 대해 train 모드를 적용한다
        # 모드는 train 혹은 evel로 설정 가능하며,
        # eval 모드로 설정하는 경우 dropout, batchnorm layer 등을 제한한다.
        self.model.train()

        train_dataloader, train_len, _, _ = self.load_dataset()

        # model save 조건처리를 위한 변수
        loss_min = 100

        # epoch(default : 16)을 1씩 증가하며 순회한다.       
        for epoch in range(self.num_epochs):
            # (모니터링)학습 시간 측정을 위해 학습 시작 시간을 기록한다.
            start_time = time.time()
            
            running_loss = 0.
            running_acc = 0

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
                
                # torch.max 메서드는 첫번째 인자로 최댓값을, 두번째 인자로 최댓값의 위치(인덱스)를 반환한다.
                # 여기서는 '최대값'은 값 자체로는 의미가 없으므로(softmax 처리 전이기 때문),
                # 최댓값의 위치(인덱스)만 가져온다.
                # 또한, 최대값은 batch 수만큼 넘어온다. 이미지 4개를 받았으면 최대값도 4개다.
                _, preds = torch.max(outputs, 1)

                # 아래 손실함수는 내부적으로 softmax 활성화 함수가 사용하므로,
                # 전달받은 outputs들의 총합을 1이 되도록 각 확률값을 변환한 다음 labels와 비교한다.
                # 위에서 label을 [a,b,c]로 가정했으나 실제로 [0,1,2] 처럼 인덱스 배열로 되어있다.
                # 이미지 1장에 대해 단순화한 예) outputs : [0.02, 0.01, 0.97] --> labels : [2]
                # 그래서 아래 손실함수를 계산하게 되면(실제 수식이 아님, 아주 단순화한 형태)
                # sum(outputs) - outputs[labels[0]] 로 계산되고 제대로 예측했다면 loss는 0에 가깝게 나올 것이다.
                loss = self.criterion(outputs, labels)
                
                # 역전파 구간 - loss를 줄이는 방향의 기울기(미분값) 수집
                loss.backward()
                # 수집한 기울기와 여러 최적화 파라미터로 가중치(w) 업데이트
                self.optimizer.step()

                # --- 모니터링을 위해 실시간(배치별) loss, accuracy 수집 ---
                # 위에서 구한 loss는 내부적으로 배치값으로 나눈 평균이므로 다시 배치사이즈를 곱해줘서 배치 전체에 대한 loss 총손실을 구해준다.
                running_loss += loss.item() * inputs.size(0)
                # preds도 인덱스, labels도 인덱스므로 비교가능하고, 배치사이즈만큼 비교해서 일치하는 갯수 만큼 올려준다.
                running_acc += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_len
            epoch_acc = running_acc / train_len * 100

            # ---- 모델 학습 모니터링 ----
            print(f"[Info] Epoch : {epoch + 1} Loss : {epoch_loss:.2f} Accuracy : {epoch_acc:.2f} Time : {time.time() - start_time}")
            # -----------------------

            # model 전체 저장
            if loss_min > loss:
                torch.save(self.model, self.model_path + "model.pt")

    def test(self):

        # 평가 모드 적용
        self.model.eval()
        
        start_time = time.time()

        _, _, test_dataloader, test_len = self.load_dataset()

        batch_count = 0

        # 가중치 추적을 하지 않는다(학습이 아니므로)
        with torch.no_grad():
            running_loss = 0.
            running_acc = 0

            # run()과 동일한 매커니즘으로, 가중치 업데이트 부분만 제외한다.
            for batch_imgs, batch_labels in test_dataloader:

                inputs = batch_imgs.to(self.device)
                labels = batch_labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
                
                batch_count += 1
                print(f"[Info] Batch : {batch_count}")
                for i in range(len(preds)):
                    print(f"\t- predict : {self.class_list[preds[i]]} - real : {self.class_list[labels[i]]}")
            
            eval_loss = running_loss / test_len
            eval_acc = running_acc / test_len * 100

            # ---- 평가 모니터링 ----
            print(f"[Info] Loss : {eval_loss:.2f} Accuracy : {eval_acc:.2f} Time : {time.time() - start_time}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pre', type=bool, default=False)
    parser.add_argument('-m', '--model_path', type=str, default="./")
    parser.add_argument('-d', '--data_folder', type=str, default="datasets")
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-s', '--shuffle', type=bool, default=True)
    parser.add_argument('-n', '--num_epochs', type=int, default=16)

    args = parser.parse_args()
    print(args.data_folder)
    
    t = Trainer(
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