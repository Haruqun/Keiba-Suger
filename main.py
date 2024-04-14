# 必要なライブラリをインポートする
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os

def main():
    # データを読み込む
    data = load_data()
    # データを前処理する
    preprocessed_data = preprocess_data(data)
    #　データを準備する
    X_train, X_test, y_train, y_test = prepare_data(preprocessed_data)
    # モデルを学習させる
    model = train_model(X_train, y_train)
    # モデルを評価する
    evaluate_model(model, X_test, y_test)

def load_data():
    # CSVファイルからデータを読み込む
    folder_path = 'csv/'  # フォルダのパスを指定
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            print(filename)  # Print the filename
            df = pd.read_csv(os.path.join(folder_path, filename), encoding='shift_jis')
            dataframes.append(df)

    print(dataframes)
    data = pd.concat(dataframes)
    return data

def preprocess_data(data):
    # 目的変数と説明変数を分割
    print(data.columns)
    y = data['Jyuni1']  # 'target'は目的変数の列名を表す
    X = data.drop('target', axis=1)

    # 欠損値の処理
    numerical_data = X.select_dtypes(include=[np.number])
    numerical_data = numerical_data.fillna(numerical_data.mean())
    X[numerical_data.columns] = numerical_data

    # カテゴリ変数のエンコーディング
    categorical_features = ['MakeDate', 'Year']  # 例として、存在する列名に変更
    numerical_features = ['Kaiji', 'Nichiji']  # 例として、存在する列名に変更

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # データのスケーリング
    X = preprocessor.fit_transform(X)

    return X, y

def prepare_data(X, y):
    # データを訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # ランダムフォレストモデルのインスタンスを作成
    model = RandomForestRegressor(random_state=0)
    # 訓練データでモデルをトレーニング
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # テストデータに対する予測を行う
    predictions = model.predict(X_test)
    # 平均絶対誤差（MAE）でモデルを評価
    mae = mean_absolute_error(y_test, predictions)
    print(f"Test MAE: {mae:.2f}")

if __name__ == "__main__":
    main()
