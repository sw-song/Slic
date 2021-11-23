from PIL import Image
import argparse

if __name__ == '__main__':
    from trainer import Trainer
else:
    from model.trainer import Trainer

t = Trainer(pre=True)

# img를 PyTorch 모델이 받을 수 있도록 변환
def img_to_tensor(img):
    img = Image.open(img).convert('RGB')
    # 받은 image를 (3, 224, 224)로 변환
    tensor_img = t.trans_test(img)
    # 4차원으로 변환 (1, 3, 224, 224)
    tensor_img.unsqueeze_(0)
    print(tensor_img.size())
    return tensor_img

def img_prediction(tensor_img):
    outputs = t.model(tensor_img)
    _, pred_idx = outputs.max(1)
    return t.class_list[pred_idx.item()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', nargs=1, type=str)

    args = parser.parse_args()
    print(img_prediction(img_to_tensor(args.file[0])))




