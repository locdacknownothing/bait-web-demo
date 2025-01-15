# OD_POI_PA

## Configs
Các tham số chạy model được set trong `./weights/default.yaml`.

Mặc định model sẽ trả về trong thư mục output:

+ Ảnh crop theo label (`save_crop=True`)

+ File txt chứa bounding box (`save_txt=True`)

+ Ảnh visualize bounding box (`save=True`)

## Run

### Khởi tạo môi trường ảo
```bash
python3 -m venv .venv
```

### Kích hoạt môi trường
```bash
source .venv/bin/activate
```

### Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Validate input 

- Trước khi detect, input là đường dẫn của một file/thư mục có thể được validate có MIME type hợp lệ và kích thước (size) lớn hơn *0 byte* và nhỏ hơn *10 MiB* 
- Output: danh sách các đường dẫn file hợp lệ (nếu input là 1 file thì vẫn trả về danh sách). Ví dụ: `["/image/path/1", "/image/path/2"]`
- Cách dùng: 

```python 
from utils import validate_source 

image_files = validate_source("/path/to/input") 
```

#### Khởi tạo model
```bash
od = OD(
    "/path/to/output/dir", 
    DEAULT_CONFIG_PATH, 
    DEFAULT_WEIGHTS, 
    DEFAULT_WEIGHT_POI_CLS
)
```

**Lưu ý**: 

- Nếu không lưu ảnh, có thể truyền vào thư mục output giá trị `None`. 
- Các file weight có thể được sync về từ server. Đường dẫn đến các file weights: _username@ip_machine:path/to/weight_
    - DEFAULT_WEIGHTS: `ts0107@192.168.1.41:/mnt/data/od/v2.0.0/yolov9_od_2507.pt` 
    - DEFAULT_WEIGHT_POI_CLS: `ts0107@192.168.1.41:/mnt/data/od/v2.0.0/poi_cls_1510_64.pt` 


#### Gọi hàm detect
```bash
od.detect(đường dẫn đến thư mục ảnh đầu vào/ảnh đầu vào/numpy array)

od.detect("./in")
```

#### Output 
Kết quả trả về là một list các object kết quả. Ví dụ:

```python
[
    ultralytics.engine.results.Results object with attributes:

    boxes: ultralytics.engine.results.Boxes object
    keypoints: None
    masks: None
    names: {0: 'PA', 1: 'POI', 2: 'SN'}
    obb: None
    orig_img: array(...)
    orig_shape: (height, width)
    path: 'path/to/raw/image'
    probs: None
    save_dir: 'path/to/save/dir'
    speed: {...}
]
```

#### Chạy model
```bash
python3 main.py -i path/to/source [-o path/to/save/dir] 
```

## Thay đổi so với phiên bản trước 

### v2.0.0
1. **Tính năng mới:** Lọc label POI theo diện tích. Tính năng này được đặt làm tham số mặc định cho hàm `detect` (`filter=True`) nên cách gọi hàm này vẫn có thể giữ như cũ.
1. **Model mới:** Model phân loại POI (`POI_CLS`) với file weights mới được tích hợp để hỗ trợ lọc POI.

### v1.1.0
1. Validate đầu vào là đường dẫn đến 1 file ảnh hoặc 1 thư mục nhiều file ảnh 
1. Thay đổi cách khởi tạo class `OD`: thêm hai tham số `config_path`, `weight_path`
## Thay đổi so với phiên bản trước 

