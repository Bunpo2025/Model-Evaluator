# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
import os
import sys
import glob

# PyTorch関連のサブモジュールを自動収集
torch_submodules = collect_submodules('torch')
torchvision_submodules = collect_submodules('torchvision')
timm_submodules = collect_submodules('timm')

# PyTorchのCUDA関連DLLを収集
def collect_torch_cuda_libs():
    """PyTorchのCUDA関連DLLを収集する"""
    binaries = []
    try:
        import torch
        torch_path = os.path.dirname(torch.__file__)
        lib_path = os.path.join(torch_path, 'lib')
        
        if os.path.exists(lib_path):
            # PyTorchのlib/ディレクトリから全てのDLLを収集
            for dll in glob.glob(os.path.join(lib_path, '*.dll')):
                binaries.append((dll, '.'))
            # .pydファイルも収集
            for pyd in glob.glob(os.path.join(lib_path, '*.pyd')):
                binaries.append((pyd, '.'))
        
        # bin/ディレクトリがある場合も収集
        bin_path = os.path.join(torch_path, 'bin')
        if os.path.exists(bin_path):
            for dll in glob.glob(os.path.join(bin_path, '*.dll')):
                binaries.append((dll, '.'))
        
        # CUDA関連DLLの名前パターン
        cuda_dll_patterns = [
            'cudart*.dll', 'cublas*.dll', 'cublasLt*.dll', 'cudnn*.dll',
            'cufft*.dll', 'curand*.dll', 'cusolver*.dll', 'cusparse*.dll',
            'nvrtc*.dll', 'nvToolsExt*.dll', 'c10_cuda.dll', 'torch_cuda*.dll',
            'caffe2_nvrtc.dll', 'nvfuser*.dll'
        ]
        
        # torch/_Cディレクトリも確認
        c_path = os.path.join(torch_path, '_C')
        if os.path.exists(c_path):
            for dll in glob.glob(os.path.join(c_path, '*.dll')):
                binaries.append((dll, '_C'))
                
    except ImportError:
        print("Warning: torch not found, CUDA DLLs will not be collected")
    except Exception as e:
        print(f"Warning: Error collecting torch CUDA libs: {e}")
    
    return binaries

# CUDA DLLを収集
cuda_binaries = collect_torch_cuda_libs()
print(f"Collected {len(cuda_binaries)} CUDA-related binaries")

# PyTorchの動的ライブラリも収集
torch_libs = collect_dynamic_libs('torch')
torchvision_libs = collect_dynamic_libs('torchvision')

a = Analysis(
    ['app/evaluate_model.py'],
    pathex=[],
    binaries=cuda_binaries + torch_libs + torchvision_libs,
    datas=[
        ('icon/Defect_segmenter.ico', 'icon'),
        ('icon/Defect_segmenter.png', 'icon'),
    ],
    hiddenimports=[
        'scipy._lib.messagestream',
        'scipy.special._ufuncs',
        'scipy.special._ufuncs_cxx',
        'scipy.special.cython_special',
        'segmentation_models_pytorch',
        'torch._numpy',
        'torch._numpy._ndarray',
        'torch._numpy._ufuncs',
        'torch._numpy._dtypes',
        'torch._numpy._dtypes_impl',
        'torch._numpy._funcs',
        'torch._numpy._util',
        'torch._dynamo',
        'torch._dynamo.utils',
        'torch._dynamo.package',
        'torch._dynamo.exc',
        'torch._dynamo.aot_compile',
        'torchvision.ops.roi_align',
        'torchvision.ops.poolers',
        'torchvision.models.convnext',
        'timm',
        'timm.layers',
        'timm.layers._fx',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
    ] + torch_submodules + torchvision_submodules + timm_submodules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,  # PyTorchの動的インポート問題を回避するためアーカイブを展開
    optimize=0,
)

pyz = PYZ(a.pure)

# onedir形式に変更（CUDA DLLの互換性向上のため）
exe = EXE(
    pyz,
    a.scripts,
    [],  # onefileではなくonedirなのでここは空
    exclude_binaries=True,  # バイナリはCOLLECTで収集
    name='ModelEvaluator_v1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # CUDA DLLの圧縮を避ける
    console=False,  # GUIアプリなのでコンソール非表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon/Defect_segmenter.ico',
)

# onedir形式：全てのファイルをフォルダに収集
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,  # CUDA DLLは圧縮しない方が安全
    upx_exclude=[],
    name='ModelEvaluator_v1.0',
)


