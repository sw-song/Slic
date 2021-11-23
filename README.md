# Slic

## 통합 실행

본 패키지는 `Single line image classifier`입니다. 단 한 줄의 명령어로 나만의 이미지 데이터셋을 생성해내고 구성된 데이터셋에 따라 자동으로 이미지 분류 모델을 학습시키며 api를 개방시켜 외부 사용자의 이미지를 만들어낸 모델로 판별할 수 있도록 합니다. 패키지 구성은 아래와 같습니다.(datasets 폴더와 model.pt의 경우 본 패키지의 사용 방식에 따라 다르게 생성됩니다. 코드 파일이나 아래 "별도 실행"을 참고해주세요.)

```bsh
|--Slic/
|   |--data/
|       |--creater.py
|       |--scraper.py
|   |--model/
|       |--trainer.py
|       |--transformer.py
|   |--serve/
|   |-(datasets/)
|   |-(model.pt)

```
---
## 별도 실행

본 패키지는 3개의 하위 패키지를 연결하도록 구성되어 있습니다. 따라서 각 기능별로 하위 패키지를 별도로 실행할 수 있으며 데이터셋만 필요한 경우 `data`를, 데이터셋은 이미 폴더별로 구성되어 있고 모델 학습이 필요한 경우 `model`을, 모델 학습까지 완료되었다면 `server` 패키지를 사용하면 되겠습니다.

### 1. data

root(./slic) directory에서 아래 명령을 실행하면 각 class 이름('class_A', 'class_B', 'class_C')에 해당하는 이미지를 웹에서 자동으로 다운로드합니다. 각 class별로 수집할 이미지 수는 -n 옵션으로 지정할 수 있으며 default는 50입니다. 또한, 저장되는 폴더명은 -d 옵션으로 지정할 수 있으며 default는 "datasets"입니다. 즉, ./slic/datasets 안에 각 class별 폴더가 생성됩니다.
```sh
$python ./data/creater.py -c "class_A" "class_B" "class_C"
```

만약 모델 학습용 데이터셋을 구축하고자 한다면 아래와 같이 -t 옵션을 true로 추가해줍니다. train_size는 -s 옵션으로 지정할 수 있으며 default는 40입니다. 이렇게 train 모드로 creater.py를 실행하면 ./slic/datasets 안에는 바로 하위 폴더로 train, test 폴더가 생성되며 그 하위에 각각 class별 폴더가 생성됩니다.
```sh
$python ./data/creater.py -c "class_A" "class_B" "class_C" -t True
```

참고할 옵션 파라미터는 아래와 같습니다.
```python
parser.add_argument('-c', '--class_name', type=str, nargs='+')
parser.add_argument('-n', '--num_imgs', type=int, default=50)
parser.add_argument('-d', '--save_folder', type=str, default="datasets")
parser.add_argument('-l', '--limit_time', type=int, default=10)
parser.add_argument('-f', '--force_replace', type=bool, default=False)
parser.add_argument('-t', '--train', type=bool, default=False)
parser.add_argument('-s', '--train_size', type=int, default=40)
```
