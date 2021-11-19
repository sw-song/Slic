from pathlib import Path
import os
# 인자 및 사용자 모듈 제어
import sys
# url 핸들링
import urllib.request
import urllib
# 이미지 유형 인식
import imghdr
# 운영체제별 '호환성' 제공
import posixpath
# regex(정규표현식)
import re

class Scraper:
    def __init__(self, class_name, max_count, class_dir, limit_time):

        # ---- 인자 변수화 ----
        self.class_name = class_name
        ## 예외 처리
        assert type(max_count) == int, "limit must be integer"
        self.max_count = max_count 
        self.class_dir = class_dir 
        ## 예외 처리
        assert type(limit_time) == int, "timeout must be integer"
        self.limit_time = limit_time
        # -----------------

        # 웹 데이터 요청시 전달할 (유저)헤더 정보
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        # 데이터 수집 카운터 (전달된 인자 'max_count' 까지 증가)
        self.count = 0
        # 데이터 수집 모니터링을 위한 변수 - 페이지 넘김 카운터
        self.page_counter = 0

    # image 한 장에 대한 처리(저장) -- call from call_save_img()
    def save_img(self, url, save_path):
        # urllib.request.Request(url, data=None, headers={}, origin_req_host=None, unverifiable=False)
        # data는 post 방식 url 호출시 전달
        req = urllib.request.Request(url, None, self.headers)
        # urlopen은 인자로 url 자체를 받거나 url request Object를 받는다.
        # 웹 데이터 수집시 크롤러 감지를 피하기 위해 header를 같이 넣어줘야 하므로,
        # url request Object를 담은 req 변수를 인자로 전달한다.
        img = urllib.request.urlopen(req).read()
        # urlopen(url)은 Request()처럼 객체를 반환한다. 객체 정보를 읽기 위해 read() 메서드를 사용했다.
        # 이 때 read()로 읽어들인(default option의 경우) 데이터는 바이너리 데이터다. 
        
        # imghdr.what(filename, h=None)
        # h가 명시되면 filename은 건너띈다. h에는 바이너리 데이터를 전달해야한다.
        if not imghdr.what(None, img):
            # 이미지 파일이 아닌 경우
            print(f'[Error] detected invalid image .. {img}')
            raise
        # 이미지 파일이 맞다면 바이너리 데이터를 파일(이미지)로 저장한다.
        with open(save_path, 'wb') as f:
            f.write(img)
        
    def call_save_img(self, url):
        # 데이터 수집 카운터 1 증가(사용자가 입력한 max_count와 비교)
        self.count += 1
        # 일반적으로 모든 환경에서 사용 가능한 이미지 파일 확장자
        default_file_types = ["jpe", "jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]
        
        try:
            # urllib.parse는 url 구성요소를 제어
            # urlsplit()은 각 구성요소를 분할하여 담은 객체 반환
            # 프로토콜(https://) 도메인(www.~~.com/) 다음에 오는 '경로' 추출
            path = urllib.parse.urlsplit(url).path
            # basename()은 url 중 path 문자열에서 마지막 '/' 이후 문자열을 추출
            # 추출한 문자열에 쿼리스트링('?'로 시작)이 포함되어 있다면 그 전까지만 추출
            filename = posixpath.basename(path).split('?')[0]
            # 추출한 filename에서 다시 확장자만 추출
            file_type = filename.split('.')[-1]
            if file_type.lower() not in default_file_types:
                # 예외적인 확장자들은 jpg로 변환
                file_type = "jpg"
            
            # ---- scraping and monitering ----
            print(f"([{self.count}]extracting image from '{url}'...)")
            ## Extract Image from Web(url)
            ## os.getcwd()는 현재 파이썬 실행파일의 디렉토리 위치를 반환한다.
            self.save_img(
                url,
                f"{os.getcwd()}/{self.class_dir}/{self.class_name}/image_{self.count}.{file_type}"
                )
            print("(...extract done!)")
        # (try)예외 처리
        except Exception as e:
            # 이미지로 수집이 불가능한 데이터의 경우 에러 메시지를 띄우고 count를 올리지 않는다.
            # 예외처리를 해주지 않으면 프로그램이 그대로 종료된다. 이를 방지하기 위함.
            self.count -= 1
            print(f"[Error]Issue on url: {url}\n{e}")
             

            
            
            

            

            
            
            




    



