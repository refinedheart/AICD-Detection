#!/bin/bash
# python train.py --data BCCD.yaml --weights '' --cfg yolov5l.yaml  --img 640 --noplots --batch-size 16 --epoch 300
python train.py --data BCCD.yaml --weights '' --cfg yolov5s.yaml  --img 640 --noplots --batch-size 16 --epoch 300
# python train.py --data BCCD.yaml --weights '' --cfg yolov5s.yaml --distill yolov5l-BCCD.pt --img 640 --noplots --batch-size 8 --distill_mode=C --epochs 300