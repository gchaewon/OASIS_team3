import os
import cv2
import pickle
import random
import pandas as pd
import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K


from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix

# 경로 설정
data_dir = "/root/yeoungeun/deeplearning"
pickle_path = os.path.join(data_dir, "dataset.pkl")  # pickle 파일 경로

# Pickle 파일 불러오기
with open(pickle_path, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# 불러온 데이터 확인
print("Pickle 파일에서 데이터 불러오기 완료")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

import gc  # 가비지 컬렉션 모듈 추가

# GPU 비활성화 (CPU만 사용)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 하이퍼파라미터 설정
param_grid = {
    'batch_size': [32, 16, 8],
    'dropout_rate': [0.25, 0.5],
    'filters': [64, 32],
    'learning_rate': [0.001, 0.0001]
}

# EarlyStopping 설정
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=3,
    verbose=0,
    mode="auto",
    restore_best_weights=True
)

# Specificity 정의 (특이도)
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Scorer 정의
specificity_scorer = make_scorer(specificity)
accuracy_scorer = make_scorer(accuracy_score)
precision_scorer = make_scorer(precision_score)
sensitivity_scorer = make_scorer(recall_score)

# CNN 모델 생성 함수
def create_compile_model(dropout_rate=0.25, filters=32, learning_rate=0.001):
    opt = Adam(learning_rate=learning_rate)
    model = Sequential([
        Conv2D(filters, kernel_size=2, padding='Same', activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Dropout(dropout_rate),
        Conv2D(filters * 2, kernel_size=2, padding='Same', activation='relu'),
        BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Dropout(dropout_rate),
        Conv2D(filters * 4, kernel_size=2, padding='Same', activation='relu'),
        BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Dropout(dropout_rate),
        Flatten(), Dense(256, activation='relu'), Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 결과를 저장할 파일 경로
results_file = os.path.join(data_dir, 'grid_search_results.pkl')

# 이전에 완료된 결과가 있다면 로드
if os.path.exists(results_file):
    with open(results_file, 'rb') as f:
        completed_results = pickle.load(f)
else:
    completed_results = []

# ParameterGrid로 모든 파라미터 조합을 생성하여 각 조합에 대해 학습
for i, params in enumerate(ParameterGrid(param_grid)):
    if any(r['params'] == params for r in completed_results):
        print(f"Skipping already completed params: {params}")
        continue

    print(f"Training with params: {params} (Combination {i+1}/{len(ParameterGrid(param_grid))})")
    batch_size = params['batch_size']
    dropout_rate = params['dropout_rate']
    filters = params['filters']
    learning_rate = params['learning_rate']
    

    # EarlyStopping 재설정
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,
        patience=3,
        verbose=0,
        mode="auto",
        restore_best_weights=True
    )


    # 모델 생성
    model = create_compile_model(dropout_rate, filters, learning_rate)
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1  # 에폭 진행 상황 표시
    )

    # 예측 및 평가
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity_score = specificity(y_test, y_pred)

    # 평가 결과 저장
    result = {
        'params': params,
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity_score
    }
    completed_results.append(result)

    # 임시 파일에 저장한 후 덮어쓰기
    temp_file = results_file + '.temp'
    with open(temp_file, 'wb') as f:
        pickle.dump(completed_results, f)
    os.replace(temp_file, results_file)  # 임시 파일을 원본 파일로 교체

    # 결과 출력
    print(f"Params: {params}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity_score}")

    # 메모리 정리
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    del model, history, y_pred  # 추가 메모리 해제
    gc.collect()  # 가비지 컬렉션 실행

# 전체 결과 DataFrame으로 변환
results_df = pd.DataFrame(completed_results)
print("\nBest Parameters and Metrics:")
print(results_df.loc[results_df['accuracy'].idxmax()])

# 결과 데이터프레임 저장
results_df.to_csv(os.path.join(data_dir, 'grid_search_results.csv'), index=False)
