#/bin/sh
WEIGHT_PATH="./best.pt"
python3 export.py --weights $WEIGHT_PATH \
                  --img-size 640 640 \
                  --include-nms
