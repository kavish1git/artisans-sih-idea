# Indian Handicrafts Dataset Structure (SIH26090)

This directory provides the standard dataset layout for collecting artisan training images and fine-tuning object detection and classification models.

## Directory Structure

```text
dataset/
├── images/           # Raw smartphone handicraft photographs (.jpg, .png)
├── annotations/      # Annotation files in YOLO/COCO format (.txt or .json)
├── classes.txt       # 24 Indian handicraft categories
└── README.md         # Documentation and training guidelines
```

## Supported Categories (classes.txt)

1. **Textiles**: Phulkari, Dupatta, Saree, Shawl, Handloom textile, Embroidery
2. **Pottery & Ceramics**: Pottery, Ceramic craft
3. **Baskets & Cane**: Basket, Bamboo craft
4. **Woodcraft**: Wooden craft, Wood carving
5. **Metal & Bell Metal**: Metal craft, Brass craft
6. **Jewelry**: Jewelry, Kundan, Meenakari, Beaded crafts
7. **Bags & Leather**: Handmade bag, Leather craft
8. **Art & Decor**: Decorative craft, Painting, Sculpture, Toy, Candle, Home decor
9. **Fallback**: Other handicraft

## Annotation Format

For bounding box detection (YOLO format):
Each image `dataset/images/craft_001.jpg` has a corresponding label file `dataset/annotations/craft_001.txt`:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Coordinates are normalized between `0.0` and `1.0`.

## Fine-Tuning Guide

1. Populate `dataset/images/` with real photographs of Indian artisan crafts.
2. Annotate bounding boxes and categories using tools like LabelImg, CVAT, or Roboflow.
3. Train or fine-tune a YOLO/MobileNet backbone using PyTorch or Ultralytics:
   ```bash
   yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```
4. Export the resulting model to ONNX:
   ```bash
   yolo export model=runs/detect/train/weights/best.pt format=onnx
   ```
5. Place the exported `.onnx` model inside the `models/` directory.
