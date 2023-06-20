#/bin/sh
python3 train_aux.py --workers 1 --device 0 \
   --batch-size 2 \
   --data data/origin.yaml \
   --img 1280 1280 \
   --cfg cfg/training/yolov7-w6.yaml \
   --weights 'runs/train/yolov7-w6/weights/best.pt' \
   --name yolov7-w6 \
   --hyp data/hyp.scratch.p6.yaml
