import numpy as np
import cv2 as cv
import sys

class ImageProcessing():
    def __init__(self):
        self.bounding_boxes = []
        self.class_ids = []
        self.confidences = []
        self.detected_objects = []

        self.net = cv.dnn.readNet("ai_components/yolov4-tiny.weights", "ai_components/yolov4-tiny.cfg")

        with open("ai_components/coco.names", "r") as file:
            self.classes = file.read().strip().split("\n")

        self.open_video()

    def open_video(self):
        self.video = cv.VideoCapture(0)

        while True:
            self.ret, self.frame = self.video.read()

            (self.width, self.height) = (self.frame.shape[1], self.frame.shape[0])

            self.blobbing()

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

            if self.ret == False:
                break

            if self.frame is None:
                break
        
    def close_all(self):
        self.video.release()
        cv.destroyAllWindows()
        sys.exit()

    def blobbing(self):
        self.blob = cv.dnn.blobFromImage(self.frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(self.blob)

        self.layers = self.net.getLayerNames()

        self.output_layers = [self.layers[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.outputs = self.net.forward(self.output_layers)

        self.detect()
    
    def detect(self):
        for self.output in self.outputs:
            for self.detection in self.output:
                self.scores = self.detection[5:]

                self.class_id = np.argmax(self.scores)

                self.confidence = self.scores[self.class_id]

                if self.confidence > 0.75:
                    self.bounding_box = self.detection[0:4] * np.array([self.width, self.height, self.width, self.height])

                    (self.center_x, self.center_y, self.width_bb, self.height_bb) = self.bounding_box.astype("int")

                    self.x, self.y = int(self.center_x - (self.width_bb / 2)), int(self.center_y - (self.height_bb / 2))

                    self.bounding_boxes.append([self.x, self.y, int(self.width_bb), int(self.height_bb)])
                    self.confidences.append(float(self.confidence))
                    self.class_ids.append(self.class_id)

                    self.draw()

    def draw(self):
        self.indices = cv.dnn.NMSBoxes(self.bounding_boxes, self.confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(self.indices) > 0:
            for self.i in self.indices.flatten():
                self.coordinate_x, self.coordinate_y, self.width_bounding_box, self.height_bounding_box= self.bounding_boxes[self.i]

                self.detected_objects.append({
                    'confidence' : self.confidence,
                    'class_id' : self.class_id,
                    'class_name' : self.classes[self.class_id],
                    'bounding_box_datas' : [self.coordinate_x, self.coordinate_y, self.width_bounding_box, self.height_bounding_box]
                })

        for self.object in self.detected_objects:
            self.x_bb, self.y_bb, self.w_bb, self.h_bb = self.object['bounding_box_datas']

            self.text = f"{self.object['class_name']} ::: {self.object['confidence']:.2f}"

            cv.rectangle(self.frame, (self.x_bb, self.y_bb), (self.x_bb + self.w_bb, self.y_bb + self.h_bb), (0, 0, 255), 2)
            cv.putText(self.frame, self.text, (self.x_bb, self. y_bb - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        self.show()

    def show(self):
        cv.imshow("video", self.frame)

        self.detected_objects = []
        self.bounding_boxes = []
        self.class_ids = []
        self.confidences = []

ImageProcessing()