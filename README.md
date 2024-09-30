# OASIS_team3

OASIS 데이터를 사용하여, 치매 예측 모델 만들기

### 4주차 9/24 ~ 10/1

**📚 해야할 것 📚**

- 각자 두개골 제거 전처리 코드 작성
  coronal, sagittal, transverse
- 기존 코드에서 디벨롭할 만한 부분을 정리하기
- CNN 모델 평가까지 틀 코드 조사해서 정리하기

### 두개골 제거 전처리

기존 전처리 코드에서, 인코딩 전 이미지 파일을 기준으로 진행

제거 효과를 확실히 보기 위해서, transverse 데이터를 사용함

original, masked, stripped 세 가지로 나눠서 시각화

threshold를 조정해서, 두개골 부위 제거하려했으나 효과가 미미

### CNN 모델링 및 학습

기존 전처리 코드에 이어서, CNN 모델링

coronal section 데이터를 사용해서 학습

113개로 데이터 갯수 부족으로 validation accuracy가 낮음

데이터 증강 방식이나 다른 데이터를 추가로 사용할 필요가 있을 듯
