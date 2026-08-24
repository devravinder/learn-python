# 03 — Solutions: Environment & Python Setup

## 1–2. Create environment and install deps

```bash
cd learn-ai-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Verify install

```bash
python -c "
import sys, numpy, pandas, matplotlib, sklearn, torch
print('Python  :', sys.version.split()[0])
print('numpy   :', numpy.__version__)
print('pandas  :', pandas.__version__)
print('mpl     :', matplotlib.__version__)
print('sklearn :', sklearn.__version__)
print('torch   :', torch.__version__)
"
```

## 4. Jupyter scratch notebook

`lessons/001_environment-and-python-setup/scratch.ipynb`, first cell:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.sum())   # 15
print(arr.mean())  # 3.0
```

## 5. `check_env.py`

```python
import sys
import numpy
import pandas
import matplotlib
import sklearn
import torch

print("Python  :", sys.version.split()[0])
print("numpy   :", numpy.__version__)
print("pandas  :", pandas.__version__)
print("mpl     :", matplotlib.__version__)
print("sklearn :", sklearn.__version__)
print("torch   :", torch.__version__)
```

Run with:

```bash
python lessons/001_environment-and-python-setup/check_env.py
```
