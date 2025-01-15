# Text detection

## Mô tả 
Model text detection giúp nhận diện văn bản có trong hình ảnh dưới dạng các bounding box. 

## Input, output
- Input: được hỗ trợ nhiều kiểu dữ liệu
    - đường dẫn đến 1 thư mục/ảnh 
    - numpy array (`dtype=np.unit8`) 
    - danh sách các phần tử có một trong các kiểu trên 
- Output: danh sách các object kết quả. Ví dụ:

```python
[
    ultralytics.engine.results.Results object with attributes:

    boxes: ultralytics.engine.results.Boxes object
    keypoints: None
    masks: None
    names: {0: 'text'}
    obb: None
    orig_img: array([...], dtype=uint8)
    orig_shape: (height, width)
    path: '/path/to/raw/image.jpg'
    probs: None
    save_dir: '/path/to/save/dir'
    speed: {...}
] 
``` 

## Config và kết quả model 
Các tham số config chạy model và kết quả train model có thể được xem trong các file tương ứng *results/args.yaml* và *results/results.json*

## Cài đặt môi trường 
Có thể cài đặt cho môi trường Python bất kỳ. Sau đây là các bước cài đặt **venv** cho Linux.

1. Khởi tạo môi trường ảo 
```bash
python3 -m venv .venv 
```

2. Kích hoạt môi trường
```bash
source .venv/bin/activate
``` 

3. Cài đặt thư viện 
```bash 
pip install -r requirements.txt
```

## Các bước chạy model 

Các bước chạy model có thể được xem mẫu trong file *detect.py*. Chi tiết mỗi bước:

### 1. Tải weights từ remote server 

- File  weights có thể được tải tự động hoặc *rsync* thủ công 
- Đường dẫn: *username@ip_machine:path/to/weights* 
    - weight_detect: `ts0107@192.168.1.41:/mnt/data/text_detection/v3.0.0/yolo11x_1612.pt` 
- Cách tải tự động:

```python
from utils_td.file import get_weights

weight_detect = get_weights(
    src_path="/mnt/data/text_detection/v3.0.0/yolo11x_1612.pt",
    src_server="ts0107@192.168.1.41",
    dst_path="./weights",
)
```
### 2. Khởi tạo instance Detector 
```python
from detect import Detector 

detector = Detector(weight_detect) 
```

### 3. Gọi hàm detect 
```python
results = detector.detect(input_source)
```


