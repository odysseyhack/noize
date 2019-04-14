import argparse
import cv2
import numpy as np
import os
import time

W, H = 544, 416  # output frame size

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
                    centers.append((centerX, centerY))
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


def _distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class BasicTracker:
    def __init__(self, on_move_fn=None, max_match_dist=10):
        '''
        Basic tracker: only matches when the detection moves within a radius `max_match_dist`.
            When there is a movement, the callback on_move_fn is called.
            TODO: make a better tracker w/ filtering and that doesn't lose track of people
            when a detection goes missing...

        Args:
        on_move_fn: called with (old_pos, new_pos) when a movement is found
        max_match_dist: maximum movement distance allowed for match in consecutive frames
        '''
        self.people = []
        self.on_move_fn = on_move_fn
        self.max_match_dist = max_match_dist

    def update(self, centers):
        new_people = []
        for c in centers:
            if not self.people:
                # add new tracker
                new_people.append(c)
            else:
                # find closest person being tracked
                dists = [_distance(c, p) for p in self.people]
                idx = np.argmin(dists)
                d = dists[idx]
                if d < self.max_match_dist:
                    # found match, move person
                    if self.on_move_fn is not None:
                        self.on_move_fn(self.people[idx], c)
                    new_people.append(c)
                    del self.people[idx]
        self.people = new_people
        print('people:', self.people)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='people counting')
    parser.add_argument('--yolodir',
                        default='/Users/alberto/projects/noize/noize/people/yolo-coco',
                        help='directory for yolo models')
    parser.add_argument('--video',
                        default='/Users/alberto/projects/noize/noize/people/data/video.mp4',
                        help='video file')
    parser.add_argument('--out_video',
                        default='people.mp4',
                        help='output video file')
    parser.add_argument('--frame_delta', type=int, default=5)
    parser.add_argument('--min_confidence', type=float, default=0.3)
    parser.add_argument('--nms_threshold', type=float, default=0.2)
    parser.add_argument('--max_match_dist', type=float, default=50.0)
    parser.add_argument('--initial_num_people', type=int, default=3,
                        help='initial number of people at the beginning of the video')
    args = parser.parse_args()

    # load yolo stuff
    net = cv2.dnn.readNetFromDarknet(os.path.join(args.yolodir, 'yolov3.cfg'),
                                     os.path.join(args.yolodir, 'yolov3.weights'))
    label_names = open(os.path.join(args.yolodir, 'coco.names')).read().strip().split("\n")
    lnames = net.getLayerNames()
    lnames = [lnames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # load video
    vs = cv2.VideoCapture(args.video)

    # output video setup
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(args.out_video, fourcc, 5.0, (W, H))

    # setup crossing people counter
    num_people_inside = args.initial_num_people
    frame = None

    line_start, line_end = (10, 220), (500, 10)
    line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    def _cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def on_move(old_pos, new_pos):
        global num_people_inside, frame
        if frame is not None:
            cv2.line(frame, old_pos, new_pos, (0, 255, 0), 2)
        old_vec = (old_pos[0] - line_start[0], old_pos[1] - line_start[1])
        new_vec = (new_pos[0] - line_start[0], new_pos[1] - line_start[1])
        co, cn = _cross(line_vec, old_vec), _cross(line_vec, new_vec)
        print('someone moved!', co, cn)

        if co <= 0 and cn > 0:
            num_people_inside += 1
            print('one person entered. total =', num_people_inside)
        if co > 0 and cn <= 0:
            num_people_inside -= 1
            print('one person left. total =', num_people_inside)

    # init tracker
    tracker = BasicTracker(on_move_fn=on_move, max_match_dist=args.max_match_dist)

    f = 0  # frame counter
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        f += 1
        if f % args.frame_delta != 0:  # skip some framew to go faster
            continue

        # basic preprocessing
        frame = preprocess(frame)

        # detection with yolo
        outputs = yolo_detect(net, frame, lnames)
        centers, boxes, confidences = process_yolo_outputs(outputs, args)

        # run non-max suppression (remove redundant detections)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.min_confidence, args.nms_threshold)
        if not isinstance(idxs, tuple):
            idxs = idxs.flatten()
        centers = [centers[idx] for idx in idxs]
        print('detection positions:', centers)

        # update tracker with the new detections
        tracker.update(centers)

        # show things
        show_boxes(frame, idxs, boxes, confidences)
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
        text = "people inside: {}".format(num_people_inside)
        cv2.putText(frame, text, (W // 2 - 20, H - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 2)

        if args.out_video:
            writer.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if args.out_video:
        writer.release()
    vs.release()
    cv2.destroyAllWindows()
