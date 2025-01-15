# Train Text Detection module for OCR using EAST

## Description
This is a PyTorch Re-Implementation of [EAST: An Efficient and Accurate Scene Text Detector](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf).

* Only RBOX part is implemented.
* Using dice loss instead of class-balanced cross-entropy loss. Some codes refer to [argman/EAST](https://github.com/argman/EAST) and [songdejia/EAST](https://github.com/songdejia/EAST)

## Prerequisites
- Prepare datasets with the structure:
```
.
└── data
    ├── test_gt
    │   └── gt_1.txt
    ├── test_img
    │   └── im_1.jpg
    ├── train_gt
    │   └── gt_2.txt
    └── train_img
        └── im_2.jpg
```

gt_*.txt: `x1,y1,x2,y2,x3,y3,x4,y4,label`
```
94,10,117,10,117,41,93,41,0
118,15,147,15,148,46,118,46,1
149,9,165,9,165,43,150,43,1
167,9,180,9,179,43,167,42,0
```

## Train
Modify the configuration parameters in ```config.py``` and run:
```
python3 train.py
```
## Detect
Modify the parameters in ```detect.py``` and run:
```
python3 detect.py
```
## Evaluate
* The evaluation scripts are from [ICDAR Offline evaluation](http://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) and have been modified.
* Modify the parameters in ```eval.py``` and run:
```
python3 eval.py
```
