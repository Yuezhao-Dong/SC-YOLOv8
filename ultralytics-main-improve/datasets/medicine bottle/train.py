from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-CBAM.yaml")
    # model = YOLO(r"E:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-improve.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-C3STR.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-SPD.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-SEA.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-C3STR-CBAM-layer.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-SPD-CBAM.yaml")
    model = YOLO(r"E:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-SPD-CBAM-layer.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-layer.yaml")
    # model = YOLO(r"D:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-layer-SPD_Conv.yaml")
    # model = YOLO(r"E:\ultralytics-main-improve\ultralytics\cfg\models\v8\yolov8s-seg-SPD-CBAM-layer1.yaml")

    # data = YOLO(r"D:\ultralytics-main\yolov8s-seg.pt")
    data = YOLO(r"E:\ultralytics-main-improve\yolov8s-seg.pt")

    model.train(data=r'E:\ultralytics-main-improve\datasets\medicine bottle\medicine bottle.yaml', epochs=100,batch=16,imgsz=640)