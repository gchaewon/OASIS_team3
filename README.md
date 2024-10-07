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

### 5주차 10/1 ~ 10/8

**📚 해야할 것 📚**

- CNN 모델 설계 및 하이퍼파라미터 조정
- Transverse 두개골 제거 여부에 따른 정확도 비교
- 추가 데이터 확보
  - data 2, 3에서 가져오기
- 데이터 증강(augmentation) 방향 논의


**❓논의 사항**
  
**1. 라벨링 문제**

전체 라벨 223개인데 coronal, sagittal 212개, transverse 213개로 개수가 맞지 않음

<img width="300" alt="Sagittal images 212" src="https://github.com/user-attachments/assets/72a65f45-d98a-4278-9220-6a5015c84c14">

따라서 `preprocessing_1006.ipynb` 파일에 각 섹션별 라벨을 만들고 파일로 저장하는 코드를 추가함

이후 각 섹션별 이미지와 라벨 개수가 일치하는 것을 확인

<img width="300" alt="Sagittal images 212" src="https://github.com/user-attachments/assets/e4ff3e66-d582-46ac-b1fb-0e9136372054">

섹션별 라벨 생성 후 모델 학습하니, 같은 모델임에도 정확도가 잘 나오지 않는 문제 발생


**2. 두개골 제거에 따른 정확도**

또한 공유 받은 두개골 제거 코드를 적용하여 각 섹션 별로 이미지를 출력했을 때

효과가 가장 좋은 sagittal에서 상대적으로 높은 정확도를 보임

transverse의 경우 raw, stripping, masked (oasis에서 가공된 것)에서 

stripping의 정확도가 가장 높았으나, val_loss 값이 지나치게 커서 재확인이 필요함


**3. 데이터 증강**

ImageDataGenerator를 사용해서 데이터 증강

증강 적용하여 학습하였으나, 증강 없이 진행한 버전과 유의미한 차이를 보이지 못함

섹션에 따라 다를 수 있으므로 재확인이 필요함

```
# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=20,              # 회전 
    width_shift_range=0.1,          # 수평 이동 
    height_shift_range=0.1,         # 수직 이동 
    shear_range=0.1,                # 전단 변환
    zoom_range=0.1,                 # 확대/축소
    horizontal_flip=True,           # 수평 반전
    brightness_range=[0.9, 1.1]    # 밝기 조정
)
```

**5. 참고하면 좋을 것 같은 캐글**

[Classification model 모음](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data)

[Validation Accuracy가 높은 모델](https://www.kaggle.com/code/ahnaftahmeed/alzheimer-detection-using-cnn)
