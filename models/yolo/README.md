# YOLO detector weights

`best_v41s.pt` — YOLOv8s drone detector (v4.1s), the frozen perception model used by all
three configs (C1/C2/C3). mAP50 = 0.991; trained with ~1200 background negatives.
Input 640×480. Loaded by `drone_tracking/scripts/yolo_detection_node.py`.
Deploy path on the machine: `~/drone_detection/models/best.pt`.
