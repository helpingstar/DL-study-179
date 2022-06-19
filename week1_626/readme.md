# ~6.26 스터디

# 데이터 다운로드 방법
URL:
https://www.kaggle.com/datasets/alessiocorrado99/animals10

1. 위 URL에서 animal dataset을 다운로드한다.
2. 폴더 안에 아래 구조 처럼 "raw-img" 폴더를 추가한다.

```
week1_626
├── raw-img
│   ├── cane
│   ├── cavallo
│   └── ...
|
├── class.py
├── split.py
└── translate.py
``` 
3. cv2 설치

``` 
pip install opencv-python
``` 
4. split.py 실행

``` 
python split.py
``` 

5. 최종 폴더 구조

- week1_626 안에 train_img 와 test_img 폴더가 생기면 됩니다.

