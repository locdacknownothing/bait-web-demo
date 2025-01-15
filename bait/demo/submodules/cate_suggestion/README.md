# Cate Suggestion

## Configs
Các tham số chạy model được set trong `configs/default.yaml`. Nếu cần thay đổi các thông số để phù hợp với output, hãy vào file này xem hướng dẫn và chỉnh sửa

Model Cate version DT_2.0.0 hỗ trợ:
+ Apple: 1106 cate [Details](./configs/list_support_apple)
+ Here: 198 cate (đã tính sub cate) [Details](./configs/list_support_here)

## Run

### Khởi tạo môi trường ảo

Lưu ý: cài đặt python version < 3.11
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

### Khởi tạo model
```bash
from predict import Model_Cate
model = Model_Cate("weight_model","label_list","map_name_cate","config")

# model = Model_Cate(model_file="/home/dgm_bait_02/short-text-classification/output/bert-base-uncased-finetuned-dgm_apple_0512/checkpoint-382592_0512",label_list="/home/dgm_bait_02/short-text-classification/dgm_label.json",map_name_cate="/home/dgm_bait_02/dms/cate_suggestion/configs/list_cate.txt",config="/home/dgm_bait_02/dms/cate_suggestion/configs/default.yaml")

config: default trong ./configs/default.yaml

Đường dẫn đến các file weights: _username@ip_machine:path/to/weight_
- weight_model: `ts0107@192.168.1.41:/mnt/data/cate_suggestion/DT_2.0.0/checkpoint-1254600-231224` 
- label_list: `ts0107@192.168.1.41:/mnt/data/cate_suggestion/DT_2.0.0/dgm_label.json`
- map_name_cate: `ts0107@192.168.1.41:/mnt/data/cate_suggestion/DT_2.0.0/list_cate.txt` 
```


Model hỗ trợ các dạng input sau:

+ text: str

+ xlsx: vui lòng kiểm tra format trong input/example.xlsx

+ json: vui lòng kiểm tra format trong input/example.json (đang improve)


### Với text, chạy hàm sau

```bash
model.predict(text)

# model.predict("Nhà May Duy Ngọc)

Output model có dạng list: [Cate x, Cate y,...]
```

### Với 2 dạng input còn lại (đã hỗ trợ xlsx)
```bash
model.multi_input("path_input")

# model.multi_input("/home/dgm_bait_02/short-text-classification/convert_ocr.json")
```