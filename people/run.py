import argparse
import cv2
import numpy as np
import os
import time


def preprocess(frame):
    frame = frame[50:-50, 200:-200]  # crop
    (h, w) = frame.shape[:2]
    frame = cv2.resize(frame, (500, int(500. * h / w)))  # resize
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='people counting')
    parser.add_argument('--yolodir', default='/Users/alberto/projects/noize/yolo-coco')
    parser.add_argument('--video', default='/Users/alberto/projects/noize/video.mp4', help='video file')
    args = parser.parse_args()

    # load yolo stuff
    net = cv2.dnn.readNetFromDarknet(os.path.join(args.yolodir, 'yolov3.cfg'), os.path.join(args.yolodir, 'yolov3.weights'))
    label_names = open(os.path.join(args.yolodir, 'coco.names')).read().strip().split("\n")

    # load video
    vs = cv2.VideoCapture(args.video)

    while True:
        # time.sleep(0.1)
        ret, frame = vs.read()
        if not ret:
            break
        frame = preprocess(frame)

        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

    vs.release()
    cv2.destroyAllWindows()
