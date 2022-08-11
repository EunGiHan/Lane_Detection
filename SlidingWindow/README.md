# Sliding Window 기반 차선 인식

## 프로젝트에 사용된 알고리즘
### Part 1. 영상 전처리
* 차선의 구분을 위한 영상의 이진화가 필요하였음.
* 이진화를 위해 LAB의 L과 HLS의 L의 정보를 따옴. 여기서 정확성을 위해 두 L을 bitwise_and를 하였음.
* 분포의 정규화를 위해 minMaxLoc 함수를 사용해 조금 더 깔끔하게 보이게 하였음.
* 밝기값의 유동적인 변화로 adaptiveThreshold 함수를 사용함.
* median 필터로 소금, 후추값을 제거함.
* 모폴로지 연산 (열기, 닫기, 오픈햇)을 통해서 median 필터의 지워진 차선 부분도 채워넣음.
* warpperceptive를 사용하여 버드아이즈 뷰 상태로 구현하였음.

### Part 2. 차선 인식
* 슬라이딩 윈도우 알고리즘을 채택하여 해당 box에서의 차선인지를 확인, 차선의 중앙을 유도하였음.

## 결과
![image](https://user-images.githubusercontent.com/55529455/184090276-7fbe8e28-c92d-4efb-9ed8-b67cc26d3e08.png)

## reference
https://www.notion.so/d8cada3fee204fecb779943f1cfa8cad
