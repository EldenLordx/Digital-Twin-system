import numpy as np
from feature.face.faceExtractor import FaceExtractor
from mot.deepsort.nn_matching import NearestNeighborDistanceMetric
from mot.deepsort.preprocessing import non_max_suppression
from mot.deepsort.detection import Detection
from mot.deepsort.tracker import Tracker

class FaceDeepSort(object):
    def __init__(self, gpu_id):
        self.min_confidence = 0.2
        self.nms_max_overlap = 1.0
        self.extractor = FaceExtractor()
        self.extractor.init(gpu_id)

        max_cosine_distance = 0.4
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        #self.width, self.height = ori_img.size
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)

        # save features and ori

        detections = [Detection(bbox_xywh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs
        
    def convert_to_square(self,bbox,img_size):
        margin = 20
        square_bbox = bbox.copy()
        h = bbox[3] - bbox[1] + 1
        w = bbox[2] - bbox[0] + 1
        max_side = np.maximum(h, w)
        square_bbox[0] = bbox[0] + w * 0.5 - max_side * 0.5
        square_bbox[1] = bbox[1] + h * 0.5 - max_side * 0.5
        square_bbox[2] = square_bbox[0] + max_side - 1
        square_bbox[3] = square_bbox[1] + max_side - 1
        
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(square_bbox[0]-margin/2, 0)
        bb[1] = np.maximum(square_bbox[1]-margin/2, 0)
        bb[2] = np.minimum(square_bbox[2]+margin/2, img_size[1]-1)
        bb[3] = np.minimum(square_bbox[3]+margin/2, img_size[0]-1)
        return bb
    
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _get_features(self, bbox_xywh, ori_img):
        features = []
        #img_size = ori_img.size
        #ori_img = np.asarray(ori_img)
        img_size = np.asarray(ori_img.shape)[0:2]
        #img_size = ori_img.size
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            x1,y1,x2,y2 = self.convert_to_square([x1,y1,x2,y2], img_size)
            im = ori_img[y1:y2, x1:x2]

            #feature = self.extractor.extract(im)[0]
            feature = self.extractor.embedding(im).cpu().detach().numpy()
            #print(feature.size)
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features


