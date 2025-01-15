from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import os



class OCR:
    def __init__(self, weight, path):
        self.weight = weight
        self.path = path
    
    def train(self,batch,iters):
        config = Cfg.load_config_from_name("vgg_transformer")

        dataset_params = {
            "name": "hw1",
            "data_root": self.path,
            "train_annotation": os.path.join(self.path,"txt","ocr.txt"),
            "valid_annotation": os.path.join(self.path,"txt","ocr_test.txt"),
            "image_height": 32,
        }

        params = {
            "print_every": 1000,
            "valid_every": 10000,
            "iters": iters,
            "checkpoint": "./checkpoint/transformerocr_checkpoint.pth",
            "export": "./weights/transformerocr.pth",
            "metrics": 10000,
            "batch_size": batch,  # 64
        }

        config["vocab"] = (
            "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        )
        
        config["weights"] = self.weight
        config["cnn"]["ss"] = [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]]

        config["trainer"].update(params)
        config["dataset"].update(dataset_params)

        trainer = Trainer(config, pretrained=True)

        trainer.train()
        trainer.config.save('/home/dgm_bait_02/ocr/weight_vietocr/config_transformer_0507.yml')
        trainer.save_weights('/home/dgm_bait_02/ocr/weight_vietocr/weight_transformer_0507.pth')

if __name__ == "__main__":
    ocr = OCR(None,"/home/dgm_bait_02/ocr/Data_OCR")
    ocr.train(batch=1,iters = 3000000)
