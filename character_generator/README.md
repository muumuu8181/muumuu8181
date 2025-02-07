# Character Generator

画像から基本属性（生命力、攻撃力、守備力）と特技を生成するシステム。

## 機能
- 入力画像からの物体検出と前処理
- CNNを使用した特徴抽出
- 基本属性と特技の生成

## 必要環境
- Python 3.12
- PyTorch
- OpenCV
- NumPy
- scikit-learn

## インストール方法
```bash
pip install -r requirements.txt
```

## プロジェクト構造
```
character_generator/
├── src/
│   ├── models/      # CNNモデルの定義
│   ├── utils/       # ユーティリティ関数
│   ├── data/        # データ処理関連
│   └── __init__.py
├── README.md
└── requirements.txt
```
