# OCR 


## Mô tả 

Model **OCR (Optical Character Recognition)** giúp nhận biết (recognize) văn bản trong hình ảnh (cụ thể là POI). Model OCR chỉ hỗ trợ đầu vào là kết quả detection của model **Text detection**.


## Input, output 

- Input: danh sách kết quả detection 
- Output: từ điển (dict) có key là đường dẫn đến ảnh, value là danh sách object kết quả recognition 

**Ví dụ:** 

- Input: (xem chi tiết hơn trong project *Text detection*)

    ```python
    [
        ultralytics.engine.results.Results object with attributes: ...
    ]
    ```

- Output: 

    ```python 
    {
        '/path/to/image': [ 
            RegRes(xyxy=[94.99051666259766, 86.29183959960938, 123.20413970947266, 104.73748779296875], text='CẮT', conf=0.8803002238273621), 
            RegRes(xyxy=[125.35323333740234, 89.83067321777344, 156.6786346435547, 107.00862884521484], text='TÓC', conf=0.8760930299758911), 
            RegRes(xyxy=[159.12368774414062, 95.05512237548828, 194.96900939941406, 109.67574310302734], text='NAM', conf=0.875045120716095), 
            RegRes(xyxy=[214.4693145751953, 96.94599914550781, 236.06797790527344, 111.98326110839844], text='NỮ', conf=0.8622713685035706), 
            RegRes(xyxy=[87.27597045898438, 1.1808027029037476, 150.2356719970703, 28.023189544677734], text='Salon', conf=0.8016341924667358)
        ]
    }
    ```

## Cách chạy 

### Cài đặt môi trường 
Có thể cài đặt môi trường Python bất kỳ. Sau đây là các bước cài đặt *venv* cho Linux.

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

### Tải weights từ remote server 

- File  weights có thể được tải tự động hoặc *rsync* thủ công 
- Đường dẫn: *username@ip_machine:/path/to/weights* 
    - weight_reg: `ts0107@192.168.1.41:/mnt/data/ocr/v3.0.0/transformerocr_2911.pth` 
    - config_reg: `ts0107@192.168.1.41:/mnt/data/ocr/v3.0.0/config_transformer.yml`
- Cách tải tự động:

```python
from utils_td.file import get_weights

weights_path = get_weights(
    src_path="/path/to/weights",
    src_server="username@ip_machine",
    dst_path="./weights",
)
```

### Khởi tạo instance OCR 

```python 
from ocr_run import OCR 

ocr = OCR(
    weight_reg="/path/to/weight_reg",
    config_reg="/path/to/config_reg",
)
```

### Gọi hàm recognize 

```python
recognition_results = ocr.recognize(detection_results) 
```

### (Optional) Extract text 

Ngoài ra, tùy vào yêu cầu của mỗi project, kết quả có thể được biến đổi bằng các function trong *utils/extractor.py*. Ví dụ:

```python
>>> from utils.extractor import extract_string_dict 
>>> string_dict = extract_string_dict(recognition_results) 
>>> print(string_dict) 
{'/path/to/image': 'cat toc nam nu salon'}

```


## Thay đổi so với phiên bản trước 

### v3.0.0 

- Tách riêng model Text detection và OCR  
- Cập nhật weights mới 
