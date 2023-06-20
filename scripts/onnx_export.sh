#/bin/sh
WEIGHT_PATH="./best.pt"
python3 export.py --weights $WEIGHT_PATH --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
