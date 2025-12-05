#!/bin/bash
/home/bai/python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --distill yolov5l.pt --img 640 --noplots --batch-size 8