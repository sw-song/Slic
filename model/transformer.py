from torchvision import transforms

def transformer(num_classes):

    # transforms : 이미지 데이터 처리(변형, 조작, 증강)
    transforms_train = transforms.Compose([
        # Resnet의 입력 이미지 사이즈와 동일하게 맞춰줌
        transforms.Resize((224, 224)),
        # 데이터 증강 - 수평반전한 이미지들을 추가로 생성
        transforms.RandomHorizontalFlip(),
        # numpy 배열을 torch tensor 형태로 변경해준다.
        transforms.ToTensor(),
        # Resnet이 default로 요구하는 정규화 방식 사용([평균], [표준편차])
        transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        ])
    
    # test 데이터이므로 규격만 맞추고 데이터 증강이나 변형은 하지 않음.
    transforms_test = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])
    
    return (transforms_train, transforms_test)