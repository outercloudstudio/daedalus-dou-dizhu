## If `carg run` is failing due to missing libtorch
Run 
```
$env:LIBTORCH_USE_PYTORCH = "1"
$torch_path = python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$env:PATH = "$torch_path;$env:PATH"
```

`pip3 install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`