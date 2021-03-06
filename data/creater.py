# operating system (시스템 및 환경 작업)
import os
# shell utility (파일 및 디렉토리 작업)
import shutil
# cli 명령 실행시 인자 전달
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.scraper import Scraper

def create_dataset(class_name, num_imgs=50, save_folder="datasets", limit_time=10, \
                   force_replace=False, train=False, train_size=40):

    # 파이썬 파일이 실행되는 현재 경로 저장
    cwd = os.getcwd()

    # class 폴더가 저장될 상위 폴더 경로 저장
    save_path = os.path.join(cwd, save_folder)


    # ---- 클래스 폴더 생성 구간 ----
    # class 폴더 경로 저장
    # 강제 리셋 인자를 줬다면 체크.
    if force_replace:
        if os.path.isdir(f"{save_path}/{class_name}"):
            # retree 메서드는 지정된 폴더를 포함해 하위 폴더 및 파일을 모두 삭제한다.
            shutil.rmtree(f"{save_path}/{class_name}")
    try:
        # 상위 폴더를 생성했다면 다음으로 class 폴더가 있는지 확인하고,
        # 없다면 생성한다.
        if not os.path.isdir(f"{save_path}/{class_name}"):
            os.makedirs(f"{save_path}/{class_name}")
    except:
        pass
    # ---------------------------

    # ---- 데이터 생성 구간 ----
    scraper = Scraper(class_name, num_imgs, save_folder, limit_time)
    scraper.run()
    # ----------------------

    # ---- train / test set 분리 구간 ----
    if train:
        print(f"[Info] Start Train-Test split ({train_size} : {num_imgs - train_size})")

        # train 폴더 생성
        if not os.path.isdir(f"{save_path}/train"):
            os.makedirs(f"{save_path}/train")

        # train 폴더에 class 폴더 생성(이미 수집했던 class가 있다면 리셋)
        if not os.path.isdir(f"{save_path}/train/{class_name}"):
            os.makedirs(f"{save_path}/train/{class_name}")
        else:
            shutil.rmtree(f"{save_path}/train/{class_name}")
            os.makedirs(f"{save_path}/train/{class_name}")

        # test 폴더 생성
        if not os.path.isdir(f"{save_path}/test"):
            os.makedirs(f"{save_path}/test")

        # test 폴더에 class 폴더 생성(이미 수집했던 class가 있다면 리셋)
        if not os.path.isdir(f"{save_path}/test/{class_name}"):
            os.makedirs(f"{save_path}/test/{class_name}")
        else:
            shutil.rmtree(f"{save_path}/test/{class_name}")
            os.makedirs(f"{save_path}/test/{class_name}")
        
        # 모든 데이터 train, test 폴더로 이동
        size = 0
        for img in os.listdir(f"{save_path}/{class_name}"):
            if size < train_size:
                shutil.move(f"{save_path}/{class_name}/{img}", f"{save_path}/train/{class_name}/{img}")
            else:
                shutil.move(f"{save_path}/{class_name}/{img}", f"{save_path}/test/{class_name}/{img}")
            size += 1
        
        # 빈 폴더 삭제
        shutil.rmtree(f"{save_path}/{class_name}")
        print("[Info] Dataset for modeling created.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--class_name', type=str, nargs='+')
    parser.add_argument('-ni', '--num_imgs', type=int, default=50)
    parser.add_argument('-sf', '--save_folder', type=str, default="datasets")
    parser.add_argument('-l', '--limit_time', type=int, default=10)
    parser.add_argument('-f', '--force_replace', type=bool, default=False)
    parser.add_argument('-t', '--train', type=bool, default=True)
    parser.add_argument('-ts', '--train_size', type=int, default=40)
    
    args = parser.parse_args()
    print(args.class_name)
    for i in range(len(args.class_name)):
        create_dataset(args.class_name[i],
             args.num_imgs,
             args.save_folder,
             args.limit_time,
             args.force_replace,
             args.train,
             args.train_size)
