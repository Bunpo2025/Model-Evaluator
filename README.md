# Model Evaluator v1.0

セグメンテーションモデルの学習結果を評価するためのGUIツールです。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 機能

### 評価指標（6種類）
| 指標 | 説明 |
|------|------|
| **IoU (Jaccard)** | 予測と正解の重なり具合 |
| **Dice Coefficient** | F1スコアに相当する指標 |
| **Pixel Accuracy** | ピクセル単位の正解率 |
| **Precision** | 精度（誤検出の少なさ） |
| **Recall** | 再現率（見逃しの少なさ） |
| **F1 Score** | PrecisionとRecallの調和平均 |

### 重複領域の除外機能
マスク画像で値が **255** のピクセルを評価から除外できます。
- 学習データの重複部分を無視して正確な評価が可能
- 除外値はGUIで設定可能（デフォルト: 255）

### その他の機能
- 📁 CSV出力機能（評価結果をエクスポート）
- 📈 評価グラフの可視化
- 🖥️ GPU対応（CUDA自動検出）
- 🎨 ダークモードGUI

## 🚀 使い方

### 1. 開発環境で実行
```bash
python evaluate_model.py
```

または
```bash
run_evaluator.bat
```

### 2. ビルド済みexeで実行
```
dist/ModelEvaluator_v1.0/ModelEvaluator_v1.0.exe
```

## 📋 操作手順

1. **モデルファイル**（.pth）を選択
2. **テスト画像フォルダ**を選択
3. **正解マスクフォルダ**を選択
4. （オプション）重複領域の除外設定を確認
5. **「🔍 評価を実行」**ボタンをクリック
6. 結果を確認し、必要に応じてCSV出力

## 🛠️ ビルド方法

PyInstallerでexeファイルを作成：

```bash
pyinstaller --clean ModelEvaluator_v1.0.spec
```

出力先: `dist/ModelEvaluator_v1.0/`

## 📦 必要なライブラリ

```
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=9.0.0
```

## 📁 ファイル構成

```
Model-Evaluator/
├── evaluate_model.py       # メインアプリケーション
├── ModelEvaluator_v1.0.spec # PyInstallerビルド設定
├── run_evaluator.bat       # 起動用バッチファイル
├── icon/                   # アイコンファイル
└── README.md
```

## 📄 ライセンス

MIT License

## 🔗 関連プロジェクト

- [komonjyo_project](https://github.com/Tolgo13/komonjyo_project) - 古文書セグメンテーションプロジェクト

