import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
import bottomDNN as bdnn
import glob
def load_trained_models(model_paths):
    models = []
    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        models.append(model)
    return models

def preprocess_input_data(input_data, scaler, feature_names):
    # 입력 데이터를 원-핫 인코딩
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, columns=['gender', 'season', 'age'])
    
    # 누락된 열을 추가하고 0으로 채우기 (학습 데이터의 원-핫 인코딩과 동일한 열을 맞추기 위해)
    missing_cols = set(feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[feature_names]
    
    X = df.values.astype(float)
    X = scaler.transform(X)
    return X


def ensemble_predict(models, X_val):
    predictions = [model.predict(X_val) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

def evaluate_ensemble_model(models, X_val, y_val, clothing_type_encoder):
    y_pred = ensemble_predict(models, X_val)
    accuracy = np.mean(y_pred == y_val)
    print(f'Ensemble Model Accuracy: {accuracy:.2f}')
    
    # 오차 행렬 및 분류 보고서 출력
    print("Confusion Matrix:")
    font_path = 'font/SKYBORI.ttf'  # 여기에는 사용하려는 한글 폰트 파일의 경로를 입력하세요.
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clothing_type_encoder.classes_, yticklabels=clothing_type_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=clothing_type_encoder.classes_))

if __name__ == '__main__':
    dataset_folder = 'Dataset/bottom'
    model_paths = [
        'Models/outer/best_model_13_1.30_0.39.keras',
        'Models/outer/best_model_13_1.26_0.43.keras',
        'Models/outer/best_model_08_1.31_0.38.keras',
        'Models/outer/best_model_08_1.30_0.41.keras',
        'Models/outer/best_model_07_1.29_0.40.keras'
    ]
    
    # 학습된 모델 로드
    models = load_trained_models(model_paths)
    
    if not models:
        raise ValueError("No models were loaded. Please check the model paths.")
    
    # 데이터 로드 및 전처리
    df = pd.read_csv('datasetouter.csv')
    X, y_clothing_type, scaler, clothing_type_encoder, feature_names = bdnn.preprocess_data(df)  # 5개의 값을 받아야 함
    
    # SMOTE 적용
    X_res, y_clothing_type_res = bdnn.apply_smote(X, y_clothing_type)
    X_train, X_val, y_clothing_type_train, y_clothing_type_val = bdnn.train_test_split(
        X_res, y_clothing_type_res, test_size=0.2, random_state=42)
    
    # 앙상블 모델 평가
    evaluate_ensemble_model(models, X_val, y_clothing_type_val, clothing_type_encoder)
    
    # 새로운 입력 데이터로 예측 수행 예시
    input_data = {
        'gender': 'MALE',
        'age': '40대',
        'season': 'winter'
    }
    X_input = preprocess_input_data(input_data, scaler, feature_names)
    y_pred = ensemble_predict(models, X_input)
    predicted_class = clothing_type_encoder.inverse_transform(y_pred)
    print(f'Predicted Clothing Type: {predicted_class[0]}')
