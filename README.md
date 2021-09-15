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
