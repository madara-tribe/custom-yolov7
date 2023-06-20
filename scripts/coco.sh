#/bin/sh
python3 train.py --workers 2 --device 0 \
 --batch-size 8 \
 --data data/coco.yaml \
 --img 416 416 \
 --cfg cfg/training/yolov7.yaml \
 --weights '' \
 --name coco \
 --hyp data/hyp.scratch.p5.yaml
