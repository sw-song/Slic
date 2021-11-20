# operating system (시스템 및 환경 작업)
import os
# shell utility (파일 및 디렉토리 작업)
import shutil
# cli 명령 실행시 인자 전달
import argparse
from scraper import Scraper

def slic(class_name, num_imgs=30, class_dir="datasets", limit_time=10, force_replace_all=False, force_replace=False):

    # 파이썬 파일이 실행되는 현재 경로 저장
    cwd = os.getcwd()

    # ---- 클래스 폴더의 상위 폴더 생성 구간 ----
    # class 폴더가 저장될 상위 폴더 경로 저장
    class_parent_path = os.path.join(cwd, class_dir)

    # 강제 리셋 인자를 줬다면 체크.
    if force_replace_all:
        if os.path.isdir(class_parent_path):
            # rmtree 메서드는 지정된 폴더를 포함해 하위 폴더 및 파일을 모두 삭제한다.
            shutil.rmtree(class_parent_path)
    
    try:
        # class 폴더가 저장될 상위 폴더(폴더명 : class_dir)가 있는지 확인하고,
        # 없다면 생성한다.
        if not os.path.isdir(class_parent_path):
            os.makedirs(class_parent_path)
    except:
        pass
    # -----------------------------------

    # ---- 클래스 폴더 생성 구간 ----
    # class 폴더 경로 저장
    class_path = os.path.join(class_parent_path, class_name)
    
    # 강제 리셋 인자를 줬다면 체크.
    if force_replace:
        if os.path.isdir(class_path):
            # retree 메서드는 지정된 폴더를 포함해 하위 폴더 및 파일을 모두 삭제한다.
            shutil.rmtree(class_path)
    
    try:
        # 상위 폴더를 생성했다면 다음으로 class 폴더가 있는지 확인하고,
        # 없다면 생성한다.
        if not os.path.isdir(class_path):
            os.makedirs(class_path)
    except:
        pass
    # ---------------------------

    scraper = Scraper(class_name, num_imgs, class_dir, limit_time)
    scraper.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--class_name', type=str, nargs='+')
    parser.add_argument('-n', '--num_imgs', type=int, default=30)
    parser.add_argument('-d', '--dir', type=str, default="datasets")
    parser.add_argument('-t', '--limit_time', type=int, default=10)
    parser.add_argument('-a', '--force_all', type=bool, default=False)
    parser.add_argument('-f', '--force', type=bool, default=False)
    
    args = parser.parse_args()
    print(args.class_name)
    for i in range(len(args.class_name)):
        slic(args.class_name[i],
             args.num_imgs,
             args.dir,
             args.limit_time,
             args.force_all,
             args.force)
