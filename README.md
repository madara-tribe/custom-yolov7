# YoloV7 with customization

```
# version
- python3.7
- pytorch 1.8.1+cu101
- torchvison 0.9.1+cu101
- torchaudio '0.8.1'
```


custmize yolov7 such as Sinkhorn loss optimize and so on. and base yolov7 for various purpose.


# version to convert as onnx

```sh
# version for onnx 
- onnx '1.12.0'
- onnxruntime '1.13.1'
- onnxsim(simplify) '0.4.8'
# coreml tool version
- coremltools '6.0'
```

# convert to CoreML 

<b>to include "class labels" to model</b>

<img src="https://user-images.githubusercontent.com/48679574/200152880-9e9d5557-b2d6-4418-8774-63e96d02dd45.png" width="800" height="300"/>

```python
COREML_CLASS_LABELS = ["trafficlight","stop", "speedlimit","crosswalk"]
# add "classifier_config" argument to model
classifier_config = ct.ClassifierConfig(COREML_CLASS_LABELS)
ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])], 
           classifier_config=classifier_config)
```

# Sinkhorn use for OTA loss optimized
optimize loss matrix(as cost-matrix) by Sinkhorn as follows:

![Shinhorn](https://user-images.githubusercontent.com/48679574/200572062-b75718c7-11dd-41d0-88e5-11ee40c7bcb7.png)

## Result of Confusion Matrix
Use shinkhorn to only <b>"class loss"</b> and had results of almost equal or little better optimized. If Sinkhorn is used to all <b>"losses (bbox, object, class)"</b>, result may become better.

<b>yolov7 / yolov7+shinkhorn</b>

<img src="https://user-images.githubusercontent.com/48679574/202828143-861f9c40-0072-4646-b627-4dc2a1f22593.png" width="450" height="400"/><img src="https://user-images.githubusercontent.com/48679574/202828151-734c3d50-1a35-4f12-91a0-7b438c4ebe57.png" width="450" height="400"/>


# References
- [CoreML API References of Classifiers](https://coremltools.readme.io/docs/classifiers)
- [Pre-release yolov7](https://github.com/WongKinYiu/yolov7)
