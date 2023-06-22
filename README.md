# yolov7 for customization
To custom, this is base model of yolov7

# How to use

Refer to <code>Makefile</code>

### train
```bash
# yolov7
python3 train.py --workers 1 --batch-size 4 --data data/origin.yaml --img 640 640 --mtype yolov7 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
# aux
python3 train_aux.py --workers 0 --batch-size 4 --data data/origin.yaml --img 640 640 --mtype yolov7_aux --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

### test
```python
# yolov7
python3 test.py --data data/origin.yaml --img 640 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weights <wight_path> --name yolov7_640_val
```

### detect
```
# yolov7
python3 detect.py --weights <wight_path> --conf 0.25 --img-size 640 --source <image_path>
```

# ONNX Export

```
python3 export.py --weights <wight_path> --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

# plus
### shinkhorn yolov7
```
git clone -b shinkhorn <shinkhorn repogitory_url>
```

### knowleadge
<b>ReOrg layer in yolov7-aux</b>

<img src="https://github.com/madara-tribe/custom-yolov7/assets/48679574/ed5bcb44-03cc-4ae6-8442-451b2c5614af" width="800" height="500"/>
