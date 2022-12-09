#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import time
from math import ceil

import cv2
import numpy as np
from cv2 import dnn

from picamera2 import Picamera2
from libcamera import Transform

class UltraFaceDetector:
    image_mean = np.array([127, 127, 127])
    image_std = 128.0
    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2
    min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
    strides = [8.0, 16.0, 32.0, 64.0]    

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--onnx_path', default="version-RFB-320_simplified.onnx", type=str, help='onnx version')
    #parser.add_argument('--input_size', default="320,240", type=str, help='define network input size,format: width,height')
    #parser.add_argument('--input_size', default="640,320", type=str, help='define network input size,format: width,height')
    #parser.add_argument('--threshold', default=0.65, type=float, help='score threshold')
    #parser.add_argument('--imgs_path', default="imgs", type=str, help='imgs dir')
    #parser.add_argument('--results_path', default="results", type=str, help='results dir')
    #args = parser.parse_args()

    def __init__(self, camera, onnx_path="version-RFB-320_simplified.onnx", input_size="320, 240", threshold=0.65):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.threshold = threshold
        self.camera = camera

    def define_img_size(self, image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self.generate_priors(feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes)
        return priors

    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
        print("priors nums:{}".format(len(priors)))
        return np.clip(priors, 0.0, 1.0)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]


    def area_of(self, left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]


    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)


    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs,
                                iou_threshold=iou_threshold,
                                top_k=top_k,
                                )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


    def convert_locations_to_boxes(self, locations, priors, center_variance,
                                size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)


    def center_form_to_corner_form(self, locations):
        return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                            locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)

    def detect_faces(self):
        net = dnn.readNetFromONNX(self.onnx_path)
        input_size = [int(v.strip()) for v in self.input_size.split(",")]
        width = input_size[0]
        height = input_size[1]
        priors = self.define_img_size(input_size)      
        while True:
            im = self.camera.capture_array()          
            #cv2.imshow("Camera", im)
            rect = cv2.resize(im, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(rect, 1 / self.image_std, (width, height), 127))
            time_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            #print("inference time: {} s".format(round(time.time() - time_time, 4)))
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = self.convert_locations_to_boxes(boxes, priors, self.center_variance, self.size_variance)
            boxes = self.center_form_to_corner_form(boxes)
            boxes, labels, probs = self.predict(im.shape[1], im.shape[0], scores, boxes, self.threshold)
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            fps = int(1/(time.time() - time_time))
            cv2.putText(im, f"FPS:{fps}", (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 0, 255), 3, cv2.LINE_AA)
            ret, frame = cv2.imencode(".jpg", im)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')        

 