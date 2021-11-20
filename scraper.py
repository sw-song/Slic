from pathlib import Path
import os
# sys : 인자 및 사용자 모듈 제어
import sys
# urllib : url 핸들링
import urllib.request
import urllib
# imghdr : 이미지 유형 인식
import imghdr
# posixpath : 모든 os에서 unix 형태 경로 제어 ('호환성' 제공)
import posixpath
# re : regex(정규표현식)
import re

class Scraper:
    def __init__(self, class_name, num_imgs, class_dir, limit_time):
    #=> class_name : 이미지 분류 클래스 명(수집 대상 쿼리명)
    #=> num_imgs : 수집하려는 이미지 수
    #=> class_dir : 클래스가 저장될 폴더 명
    #=> limit_time : 이미지 수집시 최대 소요 시간 설정

        # ---- 인자 변수화 ----
        self.class_name = class_name
        ## 예외 처리
        assert type(num_imgs) == int, "limit must be integer"
        self.num_imgs = num_imgs 
        self.class_dir = class_dir 
        ## 예외 처리
        assert type(limit_time) == int, "timeout must be integer"
        self.limit_time = limit_time
        # -----------------

        # 웹 데이터 요청시 전달할 (유저)헤더 정보
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        # 데이터 수집 카운터 (전달된 인자 'num_imgs' 까지 증가)
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
            # 객체에서 path는 프로토콜(https://) 도메인(www.~~.com/) 다음에 오는 '경로'를 가짐
            path = urllib.parse.urlsplit(url).path

            # basename()은 url 중 path 문자열에서 마지막 '/' 이후의 문자열을 추출
            # .split('?')[0] -> 추출한 문자열에 쿼리스트링('?'로 시작)이 포함되어 있다면 그 전까지만 추출
            filename = posixpath.basename(path).split('?')[0]
            
            # 추출한 filename에서 다시 확장자만 추출하여 file_type에 저장
            file_type = filename.split('.')[-1]

            # 예외적인 확장자들은 jpg로 변환
            if file_type.lower() not in default_file_types:
                file_type = "jpg"
            
            # ---- scraping and monitering ----
            print(f"([{self.count}]extracting image from '{url}'...)")
            
            ## Extract Image from Web(url)
            ## os.getcwd()는 현재 파이썬 실행파일의 디렉토리 위치를 반환
            self.save_img(
                url,
                f"{os.getcwd()}/{self.class_dir}/{self.class_name}/image_{self.count}.{file_type}"
                )
            print("(...extract done!)")

        # try에 대한 예외 처리
        except Exception as e:
            # 이미지로 수집이 불가능한 데이터의 경우 에러 메시지를 띄우고 count를 올리지 않는다.
            # 예외처리를 해주지 않으면 프로그램이 그대로 종료된다. 이를 방지하기 위함.
            self.count -= 1
            print(f"[Error]Issue on url: {url}\n{e}")
        
    def run(self):
        # 페이지를 순회하며 각 페이지 내 최대 이미지 수만큼 수집한다(최대 이미지 수는 num_imgs)
        # 만약 한 페이지에서 모든 이미지를 수집하지 못한 경우(수집 중 에러) 다음 페이지로 넘어가게 된다.
        while self.count < self.num_imgs:
            print(f"[Info]parsing page {self.page_counter + 1}")

            # q : search name(class)
            # first : number of page
            # count : number of item 
            page_url = f"https://www.bing.com/images/async?" + \
                       f"q={urllib.parse.quote_plus(self.class_name)}" + \
                       f"&first={str(self.page_counter)}" + \
                       f"&count={str(self.num_imgs)}"

            # 한 페이지(page_url) 내 모든 이미지 url 추출 -> links
            req = urllib.request.Request(page_url, None, headers=self.headers)
            resp = urllib.request.urlopen(req)
            html = resp.read().decode('utf8')
            links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
            # --- monitering ---
            print(f"[Info]Indexed {len(links)} images on parsing page {self.page_counter + 1}.")
            print("===================================================")
            # ------------------

            # 각 이미지 순회하며 call_save_img() 호출 -> 내부 save_img() 함수를 통해 이미지 파일 저장
            for link in links:
                if self.count < self.num_imgs:
                    self.call_save_img(link)
                else:
                    # 지정한 최대 이미지 수 만큼만 해당 이미지 링크에 접근하고(이미지저장) loop 통과
                    print("===================================================")
                    break
            
            # --- monitering ---
            print(f"[Info] Done. Saved {self.count} images")
            # ------------------
            
            # 페이지 번호를 올린다. 만약 현재 페이지에서 모든 이미지를 수집하지 못했다면,
            # 다음 페이지에서 남은 이미지를 수집하게 된다.
            self.page_counter += 1