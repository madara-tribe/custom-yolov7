#/bin/sh
python3 train.py --workers 2 --device 0 \
 --batch-size 8 \
 --data data/origin.yaml \
 --img 640 640 \
 --cfg cfg/training/yolov7.yaml \
 --weights '' \
 --name yolov7_traffic \
 --hyp data/hyp.scratch.p5.yaml
