# Ner


## Mô tả 

Model **NER (Named Entity Recognition)** giúp bóc tách các thuộc tính tên riêng (name), từ khóa (keyword) và địa chỉ (address) cho văn bản bất kỳ (ví dụ như output của model OCR).

## Input, output

- Input: hỗ trợ nhiều kiểu dữ liệu đầu vào
    1. chuỗi văn bản (string)
    2. danh sách các chuỗi văn bản (list of strings)
    3. từ điển map ảnh:chuỗi (dictionary of images' strings), ảnh có thể là path hoặc url

- Output: tùy vào kiểu dữ liệu đầu vào mà đầu ra khác nhau
    - Với kiểu 1 và 2: danh sách các NERResult instances. Trong đó, mỗi instance có nhiều nhất 3 thuộc tính `address, keyword, name` với giá trị là danh sách các strings.
    - Với kiểu 3: từ điển có key cũng là key của đầu vào, value là NERResult instance.

- Ví dụ: 

**NERResult instance:** 
```python
NERResult(
    "address": ["177 Phạm Như Xương TP ĐN"],
    "keyword": ["nail"],
    "name": ["Thủy Mộc"]
)
```

**Kiểu 1:**
- Input: 
```python
"Thủy Mộc nail 0334 908 256 177 Phạm Như Xương TP. ĐN" 
```
- Output: 
```python
[NERResult(
    'address' = ['177 Phạm Như Xương TP ĐN'],
    'keyword' = ['nail'],
    'name' = ['Thủy Mộc']
)]
``` 

**Kiểu 2:** 
- Input: 
```python
[
    'Thủy Mộc nail 0334 908 256 177 Phạm Như Xương TP. ĐN', 
    'Honda Thành Chuyên Sửa Chữa Bảo Dưỡng Xe Máy'
]
```
- Output: 
```python
[
    NERResult(
        'address' = ['177 Phạm Như Xương TP ĐN'],
        'keyword' = ['nail'],
        'name' = ['Thủy Mộc']
    ), 
    NERResult(
        'address' = None
        'keyword' = ['Honda', 'Sửa Chữa Bảo Dưỡng Xe Máy'], 
        'name' = ['Thành']
    )
]
```

**Kiêu 3:**
- Input: 
```python
{
    '/path/to/image/1.jpg': 'Thủy Mộc nail 0334 908 256 177 Phạm Như Xương TP. ĐN', 
    '/url/to/image/2.png': 'Honda Thành Chuyên Sửa Chữa Bảo Dưỡng Xe Máy'
}
```
- Output: 
```python
{
    '/path/to/image/1.jpg': NERResult(
        'address' = ['177 Phạm Như Xương TP ĐN'],
        'keyword' = ['nail'],
        'name' = ['Thủy Mộc']
    ), 
    '/url/to/image/2.png': NERResult(
        'address' = None
        'keyword' = ['Honda', 'Sửa Chữa Bảo Dưỡng Xe Máy'], 
        'name' = ['Thành']
    )
}
```

## Cách chạy

### Cài đặt môi trường

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

### Các bước chạy model 

Các bước chạy model có thể được xem và chạy thử trong file *ner.py*. Chi tiết mỗi bước:

#### 1. Tải model từ remote server
- Folder chứa các file cần thiết (bao gồm weights) cho model có thể được tải tự động. 
- Đường dẫn: *username@ip_machine:path/to/folder* 
    - model_path: `ts0107@192.168.1.41:/mnt/data/ner/v1.0.0/model_2511` 
- Cách dùng:

```python
from file import get_model

model_path = get_model(
    src_path="/mnt/data/ner/v1.0.0/model_2511",
    src_server="ts0107@192.168.1.41",
    dst_path="./models",
)
```

#### 2. Khởi tạo instance NER 
```python
from ner import NER 

ner = NER(model_path)  # model is downloaded from previous step
```

#### 3. Gọi hàm predict 
```python 
source = "Thủy Mộc nail 0334 908 256 177 Phạm Như Xương TP. ĐN"  # replace with any source in 1 of 3 formats defined above  
results = ner(source) 
```
