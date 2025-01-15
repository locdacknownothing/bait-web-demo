# Train POI Thai project using YOLO

## Prerequisites
- Prepare datasets (train, val, test)
- Config data paths in `config_data.yaml` file

## Train steps

### 1. Create virtual environment
```bash
python3 -m venv .venv
```

### 2. Activate venv
```bash
source .venv/bin/activate
```

### 2. Install packages
```bash
pip install -r requirements.txt
```

### 3. Start training
```bash
python3 train.py
```
