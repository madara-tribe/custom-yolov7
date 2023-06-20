#/bin/sh
python3 detect.py \
    --weights runs/train/yolov7_traffic/weights/best.pt \
    --conf 0.25 --img-size 640 \
    --source dataset/images/val \
    --name yolov7_traffics
