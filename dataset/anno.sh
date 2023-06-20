#/bin/sh
unzip archive.zip
mv archive/images .
mkdir -p labels/train images/train labels/val images/val
mv images/road2*.png images/val/
mv images/*.png images/train/
mv archive/annotations/road2*.xml labels/val/
mv archive/annotations/*.xml labels/train/
rm -r archive
python3 annotate.py labels/train
python3 annotate.py labels/val
