# Introduction
* Pre-trained Language Model을 기반으로 간단히 Triplet network 학습이 가능한 코드입니다.
* 아직 코드 정리가 안된 단계입니다.(9월 15일 기준)

# Data Processing
* 실제 뉴스 기사를 바탕으로 만든 데이터 입니다.
* 뉴스 기사 내 허위 문장이 태깅된 데이터로, 허위 문장과 아닌 문장으로 구분이 된 데이터를 triplet 데이터로 변환합니다.
  - Path: notebook/data_processing.ipynb
  - Anchor text: 뉴스 기사 (max 512 tokens)
  - Positive text: 뉴스 기사 중 허위 문장이 아닌 텍스트
  - Negative text: 뉴스 기사 중 허위 문장이라고 판단되는 텍스트


# Training 
* Data Processing을 거친 데이터를 바탕으로 모델을 학습시킵니다.
* 학습은 TripletMarginLoss 사용
  - Path: model/main.py

# Visualization
* 학습된 두 종류의 데이터의 분포를 시각화한 것입니다.
* 시각화 알고리즘은 t-SNE를 사용했습니다.
![SJ12AQAAAAASUVORK5CYII=](https://user-images.githubusercontent.com/41908581/133370149-a7b90ff7-958e-4ded-a439-f9def6102e9b.png)
  - Path: notebook/visualization.ipynb
  - 0은 허위 문장
  - 1은 허위 문장이 아닌 문장
