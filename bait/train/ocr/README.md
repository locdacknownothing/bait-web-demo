# Train OCR project using YOLO and VietOCR

## Prerequisites
- Prepare dataset as decribed in `imgs/image.png`
- Pretrained model with .pt or .pth extensions (optional)

### Note for dataset
Dataset OCR is stored with the format:

<img src = \bait\train\ocr\imgs\image.png >


In which, each text file has the format:
```
./Data_OCR/img/1.jpg    WORD1 
./Data_OCR/img/2.jpg    WORD2
```

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
python3 vietocr_train.py
```
