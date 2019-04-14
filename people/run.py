import argparse
import cv2
import numpy as np
import os
import time

W, H = 544, 416

def preprocess(frame):
    frame = frame[50:-50, 150:-150]  # crop
    (h, w) = frame.shape[:2]
    # frame = cv2.resize(frame, (416, int(416. * h / w)))  # resize
    frame = cv2.resize(frame, (W, H))  # resize
    return frame


def yolo_detect(net, frame, layer_names):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1. / 255, (416, 416),
                swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(layer_names)


def process_yolo_outputs(outputs, args):
    centers = []
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args.min_confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                if classID == 0: # only keep the class PERSON
                    centers.append([centerX, centerY])
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    return centers, boxes, confidences


def show_boxes(frame, idxs, boxes, confidences=None):
        for i in idxs:  # range(len(boxes)):
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(label_names[0], confidences[i])
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5, color, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='people counting')
    parser.add_argument('--yolodir',
                        default='/Users/alberto/projects/noize/noize/people/yolo-coco',
                        help='directory for yolo models')
    parser.add_argument('--video',
                        default='/Users/alberto/projects/noize/noize/people/data/video.mp4',
                        help='video file')
    parser.add_argument('--frame_delta', type=int, default=5)
    parser.add_argument('--min_confidence', type=float, default=0.3)
    parser.add_argument('--nms_threshold', type=float, default=0.2)
    args = parser.parse_args()

    # load yolo stuff
    net = cv2.dnn.readNetFromDarknet(os.path.join(args.yolodir, 'yolov3.cfg'),
                                     os.path.join(args.yolodir, 'yolov3.weights'))
    label_names = open(os.path.join(args.yolodir, 'coco.names')).read().strip().split("\n")
    lnames = net.getLayerNames()
    lnames = [lnames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # load video
    vs = cv2.VideoCapture(args.video)

    f = 0
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        f += 1
        if f % args.frame_delta != 0: # skip some framew to go faster
            continue

        frame = preprocess(frame)

        # detection
        outputs = yolo_detect(net, frame, lnames)
        centers, boxes, confidences = process_yolo_outputs(outputs, args)
        # print(boxes, confidences)

        # non-max suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.min_confidence, args.nms_threshold)
        if not isinstance(idxs, tuple):
            idxs = idxs.flatten()
        centers = [centers[idx] for idx in idxs]
        print(centers)

        # show things
        show_boxes(frame, idxs, boxes, confidences)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()
