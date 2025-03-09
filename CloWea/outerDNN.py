import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import glob

def load_and_preprocess_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    gender = data['metadata.model']['metadata.model.gender']
    age = data['metadata.model']['metadata.model.age']
    season = data['metadata.clothes']['metadata.clothes.season']
    clothing_type = data['metadata.clothes']['metadata.clothes.type']
    
    return gender, age, season, clothing_type

def create_dataset(dataset_folder):
    data_list = []
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                gender, age, season, clothing_type = load_and_preprocess_data(json_file_path)
                data_list.append([gender, age, season, clothing_type])
    return pd.DataFrame(data_list, columns=['gender', 'age', 'season', 'clothing_type'])

def preprocess_data(df):
    # 모든 범주형 변수를 원-핫 인코딩
    df = pd.get_dummies(df, columns=['gender', 'season', 'age'])
    
    X = df.drop(columns=['clothing_type']).values.astype(float)
    y_clothing_type = df['clothing_type'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clothing_type_encoder = LabelEncoder()
    y_clothing_type = clothing_type_encoder.fit_transform(y_clothing_type)

    return X, y_clothing_type, scaler, clothing_type_encoder, df.columns.drop('clothing_type')


def create_dnn_model(input_shape, output_shape_clothing_type, optimizer):
    inputs = Input(shape=input_shape)
    x = Dense(1024, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    
    clothing_type_output = Dense(output_shape_clothing_type, activation='softmax', name='clothing_type_output')(x)
    
    model = Model(inputs=inputs, outputs=[clothing_type_output])
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_clothing_type_train, X_val, y_clothing_type_val):
    checkpoint = ModelCheckpoint(
        'best_model_{epoch:02d}_{val_loss:.2f}_{val_accuracy:.2f}.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    model.fit(
        X_train, 
        y_clothing_type_train,
        epochs=100, 
        batch_size=16,
        validation_data=(X_val, y_clothing_type_val), 
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

def evaluate_model(model, X_val, y_clothing_type_val, clothing_type_encoder):
    losses = model.evaluate(X_val, y_clothing_type_val, return_dict=True)
    print("Evaluation losses and metrics:", losses)
    clothing_type_loss = losses.get('loss', None)
    clothing_type_accuracy = losses.get('accuracy', None)
    print(f'Best Clothing Type Model Accuracy: {clothing_type_accuracy:.2f}')
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_clothing_type_val, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clothing_type_encoder.classes_, yticklabels=clothing_type_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    print("Classification Report:")
    print(classification_report(y_clothing_type_val, y_pred_classes, target_names=clothing_type_encoder.classes_))

def check_data_imbalance(df):
    # 한글 폰트 설정
    font_path = 'font/SKYBORI.ttf'  # 여기에는 사용하려는 한글 폰트 파일의 경로를 입력하세요.
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    clothing_type_counts = df['clothing_type'].value_counts()
    season_counts = df['season'].value_counts()

    print("Clothing Type Distribution:")
    print(clothing_type_counts)
    print("\nSeason Distribution:")
    print(season_counts)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    clothing_type_counts.plot(kind='bar')
    plt.title('Clothing Type Distribution')
    plt.xlabel('Clothing Type')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    season_counts.plot(kind='bar')
    plt.title('Season Distribution')
    plt.xlabel('Season')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def apply_smote(X, y_clothing_type):
    smote = SMOTE(random_state=42)
    X_res, y_clothing_type_res = smote.fit_resample(X, y_clothing_type)
    return X_res, y_clothing_type_res

def plot_data_distribution(y_clothing_type, title):
    clothing_type_counts = pd.Series(y_clothing_type).value_counts()

    plt.figure(figsize=(6, 6))
    clothing_type_counts.plot(kind='bar')
    plt.title(f'Clothing Type Distribution {title}')
    plt.xlabel('Clothing Type')
    plt.ylabel('Count')
    plt.show()
    
def save_dataframe_to_csv(df, output_path):
    df.to_csv(output_path, index=False)

def k_fold_training(df, n_splits=5):
    X, y_clothing_type, scaler, clothing_type_encoder, feature_names = preprocess_data(df)
    
    # SMOTE 적용 전 데이터 불균형 확인
    plot_data_distribution(y_clothing_type, "Before SMOTE")
    
    X_res, y_clothing_type_res = apply_smote(X, y_clothing_type)
    
    # SMOTE 적용 후 데이터 불균형 확인
    plot_data_distribution(y_clothing_type_res, "After SMOTE")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    
    for train_index, val_index in kfold.split(X_res):
        X_train, X_val = X_res[train_index], X_res[val_index]
        y_clothing_type_train, y_clothing_type_val = y_clothing_type_res[train_index], y_clothing_type_res[val_index]
        
        optimizer = Adam(learning_rate=0.001)  # 혹은 다른 옵티마이저 사용 가능
        model = create_dnn_model((X_train.shape[1],), len(np.unique(y_clothing_type)), optimizer)
        train_model(model, X_train, y_clothing_type_train, X_val, y_clothing_type_val)
        
        models.append(model)
        
    return models

def ensemble_predict(models, X_val):
    predictions = [model.predict(X_val) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

if __name__ == '__main__':
    dataset_folder = 'Dataset/outer'
    df = create_dataset(dataset_folder)
    save_dataframe_to_csv(df, 'datasetouter.csv')
    check_data_imbalance(df)
    
    models = k_fold_training(df, n_splits=5)
    
    # 가장 최근에 저장된 모델 파일 찾기
    latest_model_files = [max(glob.glob(f'best_model_*.keras'), key=os.path.getctime) for _ in range(5)]
    best_models = [tf.keras.models.load_model(file) for file in latest_model_files]
    
    X, y_clothing_type, scaler, clothing_type_encoder, feature_names = preprocess_data(df)  # 5개의 값을 받아야 함
    X_res, y_clothing_type_res = apply_smote(X, y_clothing_type)
    X_train, X_val, y_clothing_type_train, y_clothing_type_val = train_test_split(
        X_res, y_clothing_type_res, test_size=0.2, random_state=42)
    
    y_pred = ensemble_predict(best_models, X_val)
    accuracy = np.mean(y_pred == y_clothing_type_val)
    print(f'Ensemble Model Accuracy: {accuracy:.2f}')
    
    # 오차 행렬 및 분류 보고서 출력
    evaluate_model(best_models[0], X_val, y_clothing_type_val, clothing_type_encoder)
