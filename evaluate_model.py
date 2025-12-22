# -*- coding: utf-8 -*-
"""
モデル評価アプリケーション
学習済みモデルの性能を評価するための専用GUIツール
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import cv2
import numpy as np

# === PyInstaller CUDA DLL パス設定 ===
def setup_cuda_for_exe():
    """PyInstallerでパッケージ化された場合のCUDA環境を設定"""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        current_path = os.environ.get('PATH', '')
        if exe_dir not in current_path:
            os.environ['PATH'] = exe_dir + os.pathsep + current_path
        internal_dir = os.path.join(exe_dir, '_internal')
        if os.path.exists(internal_dir) and internal_dir not in os.environ['PATH']:
            os.environ['PATH'] = internal_dir + os.pathsep + os.environ['PATH']
        if hasattr(sys, '_MEIPASS'):
            meipass = sys._MEIPASS
            if meipass not in os.environ['PATH']:
                os.environ['PATH'] = meipass + os.pathsep + os.environ['PATH']

setup_cuda_for_exe()

import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv
from datetime import datetime

# GUIなし環境でのエラー回避
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')


class MetricsCalculator:
    """セグメンテーション評価指標を計算するクラス
    
    重複部分（ignore_value=255）のピクセルは計算から除外されます。
    """
    
    @staticmethod
    def calculate_iou(pred, target, valid_mask=None):
        """IoU (Intersection over Union) を計算
        
        Args:
            pred: 予測マスク (boolean array)
            target: 正解マスク (boolean array)
            valid_mask: 有効なピクセルを示すマスク（Trueが評価対象）
        """
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    
    @staticmethod
    def calculate_dice(pred, target, valid_mask=None):
        """Dice Coefficient を計算"""
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        intersection = np.logical_and(pred, target).sum()
        total = pred.sum() + target.sum()
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        return 2 * intersection / total
    
    @staticmethod
    def calculate_pixel_accuracy(pred, target, valid_mask=None):
        """Pixel Accuracy を計算"""
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        correct = (pred == target).sum()
        total = pred.size
        if total == 0:
            return 1.0
        return correct / total
    
    @staticmethod
    def calculate_precision(pred, target, valid_mask=None):
        """Precision を計算"""
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        true_positive = np.logical_and(pred, target).sum()
        predicted_positive = pred.sum()
        if predicted_positive == 0:
            return 1.0 if true_positive == 0 else 0.0
        return true_positive / predicted_positive
    
    @staticmethod
    def calculate_recall(pred, target, valid_mask=None):
        """Recall を計算"""
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        true_positive = np.logical_and(pred, target).sum()
        actual_positive = target.sum()
        if actual_positive == 0:
            return 1.0 if true_positive == 0 else 0.0
        return true_positive / actual_positive
    
    @staticmethod
    def calculate_f1(precision, recall):
        """F1 Score を計算"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def calculate_all_metrics(pred, target, valid_mask=None):
        """すべての評価指標を計算
        
        Args:
            pred: 予測マスク (boolean array)
            target: 正解マスク (boolean array)  
            valid_mask: 有効なピクセルを示すマスク（Trueが評価対象、255の領域はFalse）
        """
        iou = MetricsCalculator.calculate_iou(pred, target, valid_mask)
        dice = MetricsCalculator.calculate_dice(pred, target, valid_mask)
        pixel_acc = MetricsCalculator.calculate_pixel_accuracy(pred, target, valid_mask)
        precision = MetricsCalculator.calculate_precision(pred, target, valid_mask)
        recall = MetricsCalculator.calculate_recall(pred, target, valid_mask)
        f1 = MetricsCalculator.calculate_f1(precision, recall)
        
        return {
            'IoU': iou,
            'Dice': dice,
            'Pixel Accuracy': pixel_acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }


class ModelEvaluatorApp:
    """モデル評価GUIアプリケーション"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("モデル評価ツール - Komonjyo Project")
        self.root.geometry("1200x1100")
        self.root.minsize(900, 600)
        
        # 変数初期化
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = tk.StringVar()
        self.image_dir = tk.StringVar()
        self.mask_dir = tk.StringVar()
        self.patch_size_var = tk.IntVar(value=512)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.ignore_value_var = tk.IntVar(value=255)  # 除外領域の値（デフォルト255）
        self.use_ignore_var = tk.BooleanVar(value=True)  # 除外領域を使用するかどうか
        self.results = []
        self.is_evaluating = False
        # New variable to select model architecture
        self.model_type = tk.StringVar(value="Unet (smp)")
        
        # プロジェクトディレクトリ
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # スタイル設定
        self.setup_styles()
        
        # GUI構築
        self.create_widgets()
        
        # GPU情報表示
        self.update_device_info()
    
    def setup_styles(self):
        """スタイルの設定"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # カラーパレット（ダークモード風）
        self.colors = {
            'bg': '#1e1e2e',
            'fg': '#cdd6f4',
            'accent': '#89b4fa',
            'success': '#a6e3a1',
            'warning': '#f9e2af',
            'error': '#f38ba8',
            'surface': '#313244',
            'overlay': '#45475a'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # フォント設定
        self.default_font = tkfont.Font(family="Meiryo UI", size=10)
        self.title_font = tkfont.Font(family="Meiryo UI", size=12, weight="bold")
        self.mono_font = tkfont.Font(family="Consolas", size=9)
        
        # スタイル定義
        self.style.configure('Dark.TFrame', background=self.colors['bg'])
        self.style.configure('Surface.TFrame', background=self.colors['surface'])
        self.style.configure('Dark.TLabel', background=self.colors['bg'], foreground=self.colors['fg'], font=self.default_font)
        self.style.configure('Title.TLabel', background=self.colors['bg'], foreground=self.colors['accent'], font=self.title_font)
        self.style.configure('Surface.TLabel', background=self.colors['surface'], foreground=self.colors['fg'], font=self.default_font)
        self.style.configure('Dark.TButton', font=self.default_font)
        self.style.configure('Accent.TButton', font=self.default_font)
        self.style.configure('Dark.TEntry', font=self.default_font)
        self.style.configure('Dark.TLabelframe', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('Dark.TLabelframe.Label', background=self.colors['bg'], foreground=self.colors['accent'], font=self.title_font)
    
    def create_widgets(self):
        """ウィジェットの作成"""
        # メインコンテナ
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側パネル（設定）
        left_panel = ttk.Frame(main_container, style='Dark.TFrame', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # 右側パネル（結果表示）
        right_panel = ttk.Frame(main_container, style='Dark.TFrame')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # === 左パネルの内容 ===
        self.create_settings_panel(left_panel)
        
        # === 右パネルの内容 ===
        self.create_results_panel(right_panel)
    
    def create_settings_panel(self, parent):
        """設定パネルの作成"""
        # タイトル
        title_label = ttk.Label(parent, text="📊 モデル評価設定", style='Title.TLabel')
        title_label.pack(pady=(0, 15))
        
        # デバイス情報
        self.device_label = ttk.Label(parent, text="", style='Dark.TLabel')
        self.device_label.pack(pady=(0, 10))
        
        # モデル選択
        model_frame = ttk.LabelFrame(parent, text="モデルファイル", style='Dark.TLabelframe')
        model_frame.pack(fill=tk.X, pady=5)
        
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, font=self.default_font, width=40)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        model_btn = ttk.Button(model_frame, text="参照", command=self.browse_model, style='Dark.TButton')
        model_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # Model type selection (Unet variants)
        type_frame = ttk.LabelFrame(parent, text="モデルタイプ", style='Dark.TLabelframe')
        type_frame.pack(fill=tk.X, pady=5)
        type_options = ["Unet (smp)", "Unet + scSE"]
        type_combo = ttk.Combobox(type_frame, textvariable=self.model_type, values=type_options, state='readonly', width=30)
        type_combo.pack(fill=tk.X, padx=5, pady=5)
        
        # テスト画像フォルダ
        image_frame = ttk.LabelFrame(parent, text="テスト画像フォルダ", style='Dark.TLabelframe')
        image_frame.pack(fill=tk.X, pady=5)
        
        image_entry = ttk.Entry(image_frame, textvariable=self.image_dir, font=self.default_font, width=40)
        image_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        image_btn = ttk.Button(image_frame, text="参照", command=self.browse_image_dir, style='Dark.TButton')
        image_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # テストマスクフォルダ
        mask_frame = ttk.LabelFrame(parent, text="正解マスクフォルダ", style='Dark.TLabelframe')
        mask_frame.pack(fill=tk.X, pady=5)
        
        mask_entry = ttk.Entry(mask_frame, textvariable=self.mask_dir, font=self.default_font, width=40)
        mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        mask_btn = ttk.Button(mask_frame, text="参照", command=self.browse_mask_dir, style='Dark.TButton')
        mask_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # パラメータ設定
        param_frame = ttk.LabelFrame(parent, text="パラメータ", style='Dark.TLabelframe')
        param_frame.pack(fill=tk.X, pady=5)
        
        # パッチサイズ
        patch_frame = ttk.Frame(param_frame, style='Dark.TFrame')
        patch_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(patch_frame, text="パッチサイズ:", style='Dark.TLabel').pack(side=tk.LEFT)
        patch_combo = ttk.Combobox(patch_frame, textvariable=self.patch_size_var, values=[256, 512, 1024], width=10, state='readonly')
        patch_combo.pack(side=tk.RIGHT)
        
        # 閾値
        threshold_frame = ttk.Frame(param_frame, style='Dark.TFrame')
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(threshold_frame, text="二値化閾値:", style='Dark.TLabel').pack(side=tk.LEFT)
        threshold_spin = ttk.Spinbox(threshold_frame, textvariable=self.threshold_var, from_=0.1, to=0.9, increment=0.1, width=10)
        threshold_spin.pack(side=tk.RIGHT)
        
        # 除外領域設定
        ignore_frame = ttk.LabelFrame(parent, text="重複領域の除外設定", style='Dark.TLabelframe')
        ignore_frame.pack(fill=tk.X, pady=5)
        
        # 除外機能のオン/オフ
        ignore_check_frame = ttk.Frame(ignore_frame, style='Dark.TFrame')
        ignore_check_frame.pack(fill=tk.X, padx=5, pady=5)
        ignore_check = ttk.Checkbutton(ignore_check_frame, text="重複領域を評価から除外する", 
                                       variable=self.use_ignore_var, style='Dark.TCheckbutton')
        ignore_check.pack(side=tk.LEFT)
        
        # 除外値の設定
        ignore_value_frame = ttk.Frame(ignore_frame, style='Dark.TFrame')
        ignore_value_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(ignore_value_frame, text="除外するピクセル値:", style='Dark.TLabel').pack(side=tk.LEFT)
        ignore_spin = ttk.Spinbox(ignore_value_frame, textvariable=self.ignore_value_var, 
                                  from_=0, to=255, increment=1, width=10)
        ignore_spin.pack(side=tk.RIGHT)
        
        # 説明ラベル
        ignore_info = ttk.Label(ignore_frame, text="※マスク画像で値が255の領域は計算から除外されます", 
                               style='Dark.TLabel', foreground='#a6adc8')
        ignore_info.pack(padx=5, pady=(0, 5))
        
        # 評価実行ボタン
        self.eval_button = ttk.Button(parent, text="🔍 評価を実行", command=self.start_evaluation, style='Accent.TButton')
        self.eval_button.pack(fill=tk.X, pady=15)
        
        # プログレスバー
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(parent, text="", style='Dark.TLabel')
        self.progress_label.pack(pady=5)
        
        # CSV出力ボタン
        self.export_button = ttk.Button(parent, text="📁 結果をCSV出力", command=self.export_to_csv, state='disabled', style='Dark.TButton')
        self.export_button.pack(fill=tk.X, pady=5)
        
        # ログ表示
        log_frame = ttk.LabelFrame(parent, text="ログ", style='Dark.TLabelframe')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, bg=self.colors['surface'], fg=self.colors['fg'], 
                                font=self.mono_font, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
    
    def create_results_panel(self, parent):
        """結果表示パネルの作成"""
        # タイトル
        title_label = ttk.Label(parent, text="📈 評価結果", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # 全体サマリー
        summary_frame = ttk.LabelFrame(parent, text="全体サマリー", style='Dark.TLabelframe')
        summary_frame.pack(fill=tk.X, pady=5)
        
        self.summary_labels = {}
        metrics = ['IoU', 'Dice', 'Pixel Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        summary_grid = ttk.Frame(summary_frame, style='Dark.TFrame')
        summary_grid.pack(fill=tk.X, padx=10, pady=10)
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            metric_frame = ttk.Frame(summary_grid, style='Surface.TFrame')
            metric_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            ttk.Label(metric_frame, text=metric, style='Surface.TLabel').pack(pady=(5, 0))
            value_label = ttk.Label(metric_frame, text="--", style='Surface.TLabel', font=self.title_font)
            value_label.pack(pady=(0, 5))
            self.summary_labels[metric] = value_label
        
        for i in range(3):
            summary_grid.columnconfigure(i, weight=1)
        
        # グラフ表示エリア
        graph_frame = ttk.LabelFrame(parent, text="評価グラフ", style='Dark.TLabelframe')
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig = Figure(figsize=(8, 4), facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['surface'])
        self.ax.tick_params(colors=self.colors['fg'])
        for spine in self.ax.spines.values():
            spine.set_color(self.colors['overlay'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 詳細結果テーブル
        table_frame = ttk.LabelFrame(parent, text="画像ごとの評価結果", style='Dark.TLabelframe')
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview
        columns = ('filename', 'iou', 'dice', 'pixel_acc', 'precision', 'recall', 'f1')
        self.result_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        self.result_tree.heading('filename', text='ファイル名')
        self.result_tree.heading('iou', text='IoU')
        self.result_tree.heading('dice', text='Dice')
        self.result_tree.heading('pixel_acc', text='Pixel Acc')
        self.result_tree.heading('precision', text='Precision')
        self.result_tree.heading('recall', text='Recall')
        self.result_tree.heading('f1', text='F1')
        
        self.result_tree.column('filename', width=200)
        for col in columns[1:]:
            self.result_tree.column(col, width=80, anchor='center')
        
        tree_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def update_device_info(self):
        """デバイス情報を更新"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.device_label.config(text=f"🖥️ GPU: {gpu_name}", foreground=self.colors['success'])
        else:
            self.device_label.config(text="🖥️ CPU モード", foreground=self.colors['warning'])
    
    def log(self, message):
        """ログメッセージを追加"""
        self.log_text.config(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
    
    def browse_model(self):
        """モデルファイルを選択"""
        default_dir = os.path.join(self.project_dir, "output", "models")
        if not os.path.exists(default_dir):
            default_dir = self.project_dir
        
        path = filedialog.askopenfilename(
            title="モデルファイルを選択",
            initialdir=default_dir,
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            self.log(f"モデル選択: {os.path.basename(path)}")
    
    def browse_image_dir(self):
        """テスト画像フォルダを選択"""
        default_dir = os.path.join(self.project_dir, "data", "images")
        if not os.path.exists(default_dir):
            default_dir = self.project_dir
        
        path = filedialog.askdirectory(title="テスト画像フォルダを選択", initialdir=default_dir)
        if path:
            self.image_dir.set(path)
            self.log(f"画像フォルダ選択: {path}")
    
    def browse_mask_dir(self):
        """正解マスクフォルダを選択"""
        default_dir = os.path.join(self.project_dir, "data", "masks")
        if not os.path.exists(default_dir):
            default_dir = self.project_dir
        
        path = filedialog.askdirectory(title="正解マスクフォルダを選択", initialdir=default_dir)
        if path:
            self.mask_dir.set(path)
            self.log(f"マスクフォルダ選択: {path}")
    
    def load_model(self):
        """モデルを読み込む"""
        model_path = self.model_path.get()
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError("モデルファイルが見つかりません")
        
        # Instantiate model based on selected type
        selected_type = self.model_type.get()
        if selected_type == "Unet (smp)":
            self.model = smp.Unet("resnet34", in_channels=3, classes=1).to(self.device)
        elif selected_type == "Unet + scSE":
            self.model = smp.Unet("resnet34", in_channels=3, classes=1, decoder_attention_type="scse").to(self.device)
        else:
            # Fallback to plain Unet
            self.model = smp.Unet("resnet34", in_channels=3, classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.log(f"モデル読み込み完了 ({selected_type}): {os.path.basename(model_path)}")
    
    def safe_cv2_imread(self, image_path):
        """Unicode対応の画像読み込み"""
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if data.size == 0:
                return None
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            return image
        except Exception:
            return None
    
    def get_image_mask_pairs(self):
        """画像とマスクのペアを取得"""
        image_dir = self.image_dir.get()
        mask_dir = self.mask_dir.get()
        
        if not os.path.isdir(image_dir):
            raise FileNotFoundError("画像フォルダが見つかりません")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError("マスクフォルダが見つかりません")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        pairs = []
        
        for img_file in os.listdir(image_dir):
            name, ext = os.path.splitext(img_file)
            if ext.lower() not in image_extensions:
                continue
            
            # 対応するマスクを検索
            mask_file = None
            for mask_ext in image_extensions:
                candidate = os.path.join(mask_dir, name + mask_ext)
                if os.path.exists(candidate):
                    mask_file = candidate
                    break
            
            if mask_file:
                pairs.append((os.path.join(image_dir, img_file), mask_file, img_file))
        
        return pairs
    
    def predict_image(self, image_path):
        """画像の予測を実行"""
        patch_size = self.patch_size_var.get()
        threshold = self.threshold_var.get()
        
        # 画像読み込み
        image = self.safe_cv2_imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # RGB変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # リサイズ
        image_resized = cv2.resize(image, (patch_size, patch_size))
        
        # テンソル変換
        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255.0
        
        # 推論
        with torch.no_grad():
            pred = self.model(image_tensor)
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float()
        
        # 元サイズに戻す
        pred_np = pred.squeeze().cpu().numpy()
        pred_full = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pred_full.astype(bool)
    
    def load_mask(self, mask_path):
        """マスク画像を読み込む
        
        Returns:
            tuple: (target_mask, valid_mask)
                - target_mask: 正解マスク (boolean array, 欠損部分がTrue)
                - valid_mask: 有効領域マスク (boolean array, 評価対象がTrue、除外領域がFalse)
        """
        mask = self.safe_cv2_imread(mask_path)
        if mask is None:
            return None, None
        
        # グレースケール変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 除外領域の検出（ignore_valueと一致するピクセル）
        ignore_value = self.ignore_value_var.get()
        use_ignore = self.use_ignore_var.get()
        
        if use_ignore:
            # 除外領域を検出（ignore_valueのピクセルはFalse = 評価から除外）
            valid_mask = (mask != ignore_value)
        else:
            # 除外しない場合はすべてTrue
            valid_mask = np.ones(mask.shape, dtype=bool)
        
        # 二値化（除外領域以外の部分で、閾値127以上を欠損として検出）
        # まず、除外領域以外で二値化
        target_mask = np.zeros(mask.shape, dtype=bool)
        # 除外値でないピクセルのうち、値が127より大きいものを欠損とする
        target_mask[(mask > 127) & (mask != ignore_value)] = True
        
        return target_mask, valid_mask
    
    def start_evaluation(self):
        """評価を開始"""
        if self.is_evaluating:
            return
        
        # 入力チェック
        if not self.model_path.get():
            messagebox.showerror("エラー", "モデルファイルを選択してください")
            return
        if not self.image_dir.get():
            messagebox.showerror("エラー", "テスト画像フォルダを選択してください")
            return
        if not self.mask_dir.get():
            messagebox.showerror("エラー", "正解マスクフォルダを選択してください")
            return
        
        self.is_evaluating = True
        self.eval_button.config(state='disabled')
        self.export_button.config(state='disabled')
        
        # 結果クリア
        self.results = []
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        threading.Thread(target=self.run_evaluation, daemon=True).start()
    
    def run_evaluation(self):
        """評価を実行（バックグラウンド）"""
        try:
            # モデル読み込み
            self.log("モデルを読み込み中...")
            self.load_model()
            
            # 画像ペア取得
            self.log("画像とマスクのペアを検索中...")
            pairs = self.get_image_mask_pairs()
            
            if not pairs:
                raise ValueError("評価対象の画像ペアが見つかりません")
            
            self.log(f"評価対象: {len(pairs)} 件")
            
            # プログレスバー設定
            self.root.after(0, lambda: self.progress.config(maximum=len(pairs), value=0))
            
            # 除外設定のログ出力
            if self.use_ignore_var.get():
                self.log(f"重複領域除外: 有効 (除外値={self.ignore_value_var.get()})")
            else:
                self.log("重複領域除外: 無効")
            
            # 各画像を評価
            for i, (img_path, mask_path, filename) in enumerate(pairs):
                self.root.after(0, lambda f=filename: self.progress_label.config(text=f"処理中: {f}"))
                
                # 予測
                pred = self.predict_image(img_path)
                if pred is None:
                    self.log(f"警告: {filename} の読み込みに失敗")
                    continue
                
                # マスク読み込み（target_maskとvalid_maskの両方を取得）
                target, valid_mask = self.load_mask(mask_path)
                if target is None:
                    self.log(f"警告: {filename} のマスク読み込みに失敗")
                    continue
                
                # サイズ確認（異なる場合はリサイズ）
                if pred.shape != target.shape:
                    target = cv2.resize(target.astype(np.uint8), (pred.shape[1], pred.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    valid_mask = cv2.resize(valid_mask.astype(np.uint8), (pred.shape[1], pred.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # 除外領域の統計情報をログ出力（最初の画像のみ）
                if i == 0 and self.use_ignore_var.get():
                    total_pixels = valid_mask.size
                    valid_pixels = valid_mask.sum()
                    ignored_pixels = total_pixels - valid_pixels
                    ignored_ratio = (ignored_pixels / total_pixels) * 100
                    self.log(f"除外領域: {ignored_pixels:,} ピクセル ({ignored_ratio:.1f}%)")
                
                # 評価指標計算（valid_maskを渡して除外領域を無視）
                metrics = MetricsCalculator.calculate_all_metrics(pred, target, valid_mask)
                metrics['filename'] = filename
                self.results.append(metrics)
                
                # テーブルに追加
                self.root.after(0, lambda m=metrics: self.add_result_to_table(m))
                
                # プログレス更新
                self.root.after(0, lambda v=i+1: self.progress.config(value=v))
            
            # サマリー計算
            self.root.after(0, self.update_summary)
            self.root.after(0, self.update_graph)
            
            self.log(f"評価完了: {len(self.results)} 件処理")
            self.root.after(0, lambda: self.progress_label.config(text="評価完了"))
            self.root.after(0, lambda: messagebox.showinfo("完了", f"評価が完了しました\n処理件数: {len(self.results)} 件"))
            
        except Exception as e:
            self.log(f"エラー: {e}")
            self.root.after(0, lambda: messagebox.showerror("エラー", str(e)))
        finally:
            self.is_evaluating = False
            self.root.after(0, lambda: self.eval_button.config(state='normal'))
            if self.results:
                self.root.after(0, lambda: self.export_button.config(state='normal'))
    
    def add_result_to_table(self, metrics):
        """結果をテーブルに追加"""
        self.result_tree.insert('', 'end', values=(
            metrics['filename'],
            f"{metrics['IoU']:.4f}",
            f"{metrics['Dice']:.4f}",
            f"{metrics['Pixel Accuracy']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['F1 Score']:.4f}"
        ))
    
    def update_summary(self):
        """サマリーを更新"""
        if not self.results:
            return
        
        metrics_keys = ['IoU', 'Dice', 'Pixel Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for key in metrics_keys:
            values = [r[key] for r in self.results]
            avg = np.mean(values)
            self.summary_labels[key].config(text=f"{avg:.4f}")
    
    def update_graph(self):
        """グラフを更新"""
        if not self.results:
            return
        
        self.ax.clear()
        self.ax.set_facecolor(self.colors['surface'])
        
        metrics_keys = ['IoU', 'Dice', 'Pixel Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#f5c2e7', '#cba6f7', '#fab387']
        
        x = np.arange(len(self.results))
        width = 0.12
        
        for i, (key, color) in enumerate(zip(metrics_keys, colors)):
            values = [r[key] for r in self.results]
            offset = (i - len(metrics_keys)/2 + 0.5) * width
            self.ax.bar(x + offset, values, width, label=key, color=color, alpha=0.8)
        
        self.ax.set_xlabel('画像番号', color=self.colors['fg'])
        self.ax.set_ylabel('スコア', color=self.colors['fg'])
        self.ax.set_title('画像ごとの評価スコア', color=self.colors['fg'])
        self.ax.legend(loc='upper right', facecolor=self.colors['surface'], labelcolor=self.colors['fg'])
        self.ax.set_ylim(0, 1.1)
        self.ax.tick_params(colors=self.colors['fg'])
        
        for spine in self.ax.spines.values():
            spine.set_color(self.colors['overlay'])
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def export_to_csv(self):
        """結果をCSV出力"""
        if not self.results:
            messagebox.showwarning("警告", "出力するデータがありません")
            return
        
        # 保存先選択
        default_name = f"evaluation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            title="CSVファイルの保存先を選択",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not path:
            return
        
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # ヘッダー
                writer.writerow(['ファイル名', 'IoU', 'Dice', 'Pixel Accuracy', 'Precision', 'Recall', 'F1 Score'])
                
                # データ
                for r in self.results:
                    writer.writerow([
                        r['filename'],
                        f"{r['IoU']:.6f}",
                        f"{r['Dice']:.6f}",
                        f"{r['Pixel Accuracy']:.6f}",
                        f"{r['Precision']:.6f}",
                        f"{r['Recall']:.6f}",
                        f"{r['F1 Score']:.6f}"
                    ])
                
                # サマリー行
                writer.writerow([])
                writer.writerow(['=== 平均スコア ==='])
                metrics_keys = ['IoU', 'Dice', 'Pixel Accuracy', 'Precision', 'Recall', 'F1 Score']
                avg_values = ['平均'] + [f"{np.mean([r[k] for r in self.results]):.6f}" for k in metrics_keys]
                writer.writerow(avg_values)
            
            self.log(f"CSV出力完了: {path}")
            messagebox.showinfo("完了", f"CSVファイルを出力しました\n{path}")
            
            # ファイルを開くか確認
            if messagebox.askyesno("確認", "出力したCSVファイルを開きますか？"):
                os.startfile(path)
                
        except Exception as e:
            self.log(f"CSV出力エラー: {e}")
            messagebox.showerror("エラー", f"CSV出力に失敗しました:\n{e}")


def main():
    root = tk.Tk()
    app = ModelEvaluatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


