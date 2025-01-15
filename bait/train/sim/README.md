# Train Sim Model project using CNN and Triplet loss

## Prerequisites
- Prepare train and validation datasets
- Config the variables in `config.py` file (dataset paths, backbone name, etc.)

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
