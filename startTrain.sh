#!/bin/bash
# /home/bai/anaconda3/envs/jittor-yolov5/bin/python train.py --data BCCD.yaml --weights '' --cfg yolov5l.yaml  --img 416 --noplots --batch-size 8
# /home/bai/anaconda3/envs/jittor-yolov5/bin/python train.py --data BCCD.yaml --weights '' --cfg yolov5s.yaml  --img 416 --noplots --batch-size 8
/home/bai/anaconda3/envs/jittor-yolov5/bin/python train.py --data BCCD.yaml --weights '' --cfg yolov5s.yaml --distill yolov5l-BCCD.pt --img 416 --noplots --batch-size 8 --distill_mode=C --epochs 100