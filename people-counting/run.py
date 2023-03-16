import argparse
import csv
import datetime
import json
import os
import sys
import time
from itertools import zip_longest

import cv2
import dlib
import imutils
import numpy as np
import schedule
from imutils.video import VideoStream

from centroidtracker import CentroidTracker
from trackableobject import TrackableObject

t0 = time.time()

# cwd = '/home/pi/people-counting/people-counting'
cwd = 'D:/Dev/repo/people-counting/people-counting'

def run():

    ## ค่าคอนฟิคที่รับเป็น argument ตอนรันไฟล์
    #สำหรับไฟล์ vdo
    #python run.py --input videos/example_03.mp4 -s 30 -c 0.2 --no-display
    #python run.py --input videos/example_03.mp4 -s 30 -c 0.2

    #สำหรับกล้อง
    #python run.py --no-display
    #python run.py
    #              -------------------------------------------------------
    
    ap = argparse.ArgumentParser()

    # source ของวิดีโอที่จะใช้แสดง เป็นตัวเลือกเพิ่มเติม หากไม่ใส่จะเลือกเป็น camera
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    # ตัวเลือกว่าจะ save video ออกมามั้ย
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    # ค่าความหน้าจะเป็นที่ใช้ในการตัดสินใจว่าวัตถุนั้น ๆ เป็นไปวัตถุที่เราต้องการหรือไม่ เช่น detect เจอมอไซต์ จะมีความน่าจะเป็นออกมา หลาย ๆ ไม่ว่าจะเป็น มอไซต์ 70% หรือ 0.7 จักรยาน 36% ตุ๊ก ๆ 5 % เป็นต้น
    ap.add_argument("-c", "--confidence", type=float, default=0.8,
                    help="minimum probability to filter weak detections")
    #เมื่อ detect เจอวัตถุจะให้ข้ามกี่ frames ค่าคอนฟิคนี้มีผลกับความเร็วการประมวลผล และกำตรวจจับการเคลื่อนไหมของวัตถุ
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    #จะให้แสดงผลวิดีโอหรือไม่
    ap.add_argument("-d", "--display", default=True, action='store_true')
    ap.add_argument("-n-d", "--no-display", dest='display', action='store_false',
                    help="not show real-time monitoring window")
    args = vars(ap.parse_args())

    # ค่าเริ่มต้นที่จะใช้จำแนกวัตถุที่ detect ได้
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # โหลด Model ที่จะใช้จำแนกวัตถุ --> โมเอามาใช้เป็น model สำเร็จรูปที่ผ่านการทำ Machine learning มาแล้ว
    net = cv2.dnn.readNetFromCaffe('{}/mobilenet_ssd/MobileNetSSD_deploy.prototxt'.format(cwd), '{}/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'.format(cwd))

    # ตรวจสอบว่ามี argument input ป่าว, ถ้าไม่มีให้ใช้กล้อง raspberry pi
    if not args.get("input", False):
        vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)

    # ถ้ามีก็ใช้ไฟล์วิดีโอ
    else:
        vs = cv2.VideoCapture(args["input"])

    # สมมุติว่าต้องการจะให้เขียนไฟล์
    writer = None

    # ประกาศตัวแปรความกว้างความสูงของเฟรม
    W = None
    H = None

    # กำหนดกึ่งกลางของวัตถุที่จะแท็ก
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    # ประกาศตัวแปรเพื่อเก็บ data ของวัตถุที่จะแท็ก
    trackers = []
    trackableObjects = {}

    # ประกาศตัวแปรเก็บจะนวณเฟรม คนเดินไปทางซ้าย คนเดินไปทางขวา
    totalFrames = 0
    totalLeft = 0
    totalRight = 0
    x = []
    empty = []
    empty1 = []

    # ลูป
    while True:
        # เก็บ frame จากวิดีโอหรือกล้อง
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        # เซ็คว่ามีข้อมูล
        if args["input"] is not None and frame is None:
            break

        # ปรับขนาน frame ให้กว้างสุด 500 และเปลี่ยนรูปแบบสีจาก BGR เป็น RGB
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # เซ็ตขนานของ frame
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # เขียนไฟล์ถ้าต้องการวิดีโอ
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (W, H), True)

        # สถานะตอนนี้
        status = "Waiting"
        rects = []

        # ถ้าเป็น frame ที่ต้องประมวลผล
        if totalFrames % args["skip_frames"] == 0:
            # เซ็ตสถานะ
            status = "Detecting"
            trackers = []

            # เปลี่ยนรูปเป็น blob type เพื่อที่จะเอาไปเข้า ssd เพื่อทำนายว่าในรูปมีอะไรในนั้นบ้าง
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # ตรวจสอบทุก opject ที่เจอ
            for i in np.arange(0, detections.shape[2]):
                # detections จะมีข้อมูลที่เกี่ยวข้ออยู่เยอะแต่เอาความหน้าจะเป็นออกมา
                confidence = detections[0, 0, i, 2]

                # เลือกดูแค่ค่าที่มากกว่าความน่าจะเป็นที่ตั้งไว้จาก argument
                if confidence > args["confidence"]:
                    # เอาลำดับออกมา
                    idx = int(detections[0, 0, i, 1])

                    # เซ็คว่าลำดับนั้นคือคนหรือไม่
                    if CLASSES[idx] != "person":
                        continue

                    # ถ้าเป็นคนให้มาตีกรอบให้มัน
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    if(args["display"]):
                        # วาดกรอบลงไปใน frame
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)

                    # เอากรอบนี้เก็บไว้เพื่อใช้แท็กการเคลื่อนที่
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

        # ถ้าไม่ใช่ให้มาประมวลผลการเคลื่อนที่
        else:
            for tracker in trackers:
                # เซ็ตสถานะ
                status = "Tracking"

                # update the tracker
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        # วาดเส้นตรงกลาง
        if(args["display"]):
            cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 0), 3)

        # อัพเดตการติดตามตำแหน่ง
        objects = ct.update(rects)

        # เซ็คทุก object ที่มีการติดตาม
        for (objectID, centroid) in objects.items():
            # เรียกข้อมูลของ opject ออกมา
            to = trackableObjects.get(objectID, None)

            # ถ้าไม่มีให้คิดว่าวัตถุนั้นเป็นชิ้นใหม่ให้สร้าง id ให้ซะ
            if to is None:
                to = TrackableObject(objectID, centroid)

            # ถ้ามีให้มาเซ็คการเคลื่อนที่
            else:
                # เดาทิศทางการเคลื่อนที่
                y = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(y)
                to.centroids.append(centroid)

                # เซ็คว่าได้นับยัง
                if not to.counted:
                    # ถ้าค่า direction ไปในทางลบ ก็แสดงว่าวัตถุนั้นกำลังไปทางซ้าย
                    if direction > 0 and centroid[0] > W // 2:
                    # if direction < 0 and centroid[0] < W // 2:
                        totalLeft += 1
                        empty.append(totalLeft)
                        to.counted = True

                    # ถ้าไม่ก็ทางขวา
                    elif direction < 0 and centroid[0] < W // 2:
                    # elif direction > 0 and centroid[0] > W // 2:
                        totalRight += 1
                        empty1.append(totalRight)
                        to.counted = True

                    x = []
                    # นับว่าเข้าไปกี่คนแล้ว
                    x.append(len(empty)-len(empty1))

            # เก็บข้อมูล object  ไว้ใช้ครั้งต่อไป
            trackableObjects[objectID] = to

            # แสดง id ที่รูป แล้วก็จุดตรงกลาง
            if args["display"]:
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(
                    frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # ข้อมูลสถานะ
        info = [
            ("Exit", totalRight),
            ("Enter", totalLeft),
            ("Status", status),
        ]

        info2 = [
            ("Total people inside", x),
        ]

        if args["display"]:
            # แสดงข้อมูลสถานะในรูป
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for (i, (k, v)) in enumerate(info2):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (265, H - ((i * 20) + 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # แสดงข้อมูลสถานะใน command line
            # print("[INFO] Total people exit: {}, Total people enter: {}, Total people inside: {}".format(
            #     totalLeft, totalRight, x), end="\r", flush=True)

            # ส่งออกเป็น JSON
            print(json.dumps({"exit": totalRight, "enter": totalLeft, "inside": x}))

        # ต้องบันทึกวิดีโอมั้ย
        if writer is not None:
            writer.write(frame)

        # show the output frame
        if args["display"]:
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        
        # รอรับว่ามีการพิมพ์อะไรมามั้ย
        key = cv2.waitKey(1) & 0xFF

        # ถ้าพิมพ์ q มาให้ออกจากโปรแกรม
        if key == ord("q"):
            break

        totalFrames += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()