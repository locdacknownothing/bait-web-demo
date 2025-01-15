# Train POI Vietnam project using YOLO

## Prerequisites
- Prepare datasets (train, val, test)
- Config the paths in `config_data.yaml` file

## Train steps

### 1. Create virtual environment
```bash
python3 -m venv .venv
```

### 2. Activate venv
```bash
source .venv/bin/activate
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

### 4. Start training
```bash
python3 train.py
```
