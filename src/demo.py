import cv2
import os
import threading
import time
from darcyai import DarcyAI, DarcyAIConfig
from flask import Flask, request, Response


VIDEO_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video0")


class Demo:
    def __init__(self):
        self.__seen_people = []

        # Init our Flask App
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.__flask_app = Flask(__name__, static_url_path=script_dir)
        self.__flask_app.add_url_rule("/", "root", self.__root)
        self.__flask_app.add_url_rule("/mode", "mode", self.__change_mode, methods = ["POST"])

        # Configure how the AI will work
        darcy_ai_config = DarcyAIConfig(
            face_rectangle_yfactor=0.8,
            pose_minimum_face_threshold=0.5,
            object_tracking_color_sample_pixels=16)

        # Init the Darcy AI SDK Library and call back
        self.__ai = DarcyAI(
            config=darcy_ai_config,  # This is the config
            data_processor=self.__analyze, # This is the call back for any detected objects
            frame_processor=self.__frame_processor, # This will annotate a frame with person id
            flask_app=self.__flask_app,
            arch="armv7l",
            use_pi_camera=False,
            video_device=VIDEO_DEVICE)

        self.__mode = "people_counting"


    def StartAPIServer(self):
        self.__flask_app.run(
            host="0.0.0.0",
            port=3456,
            debug=False)


    def StartPeopleCounting(self):
        self.__mode = "people_counting"

        self.__ai.StartPeoplePerception()


    def StartObjectDetection(self):
        self.__mode = "object_detection"

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.__ai.StartObjectDetection(
            detect_perception_model="%s/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" % script_dir,
            detect_perception_threshold=0.7,
            detect_perception_labels_file="%s/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu_labels.txt" % script_dir)


    def StartImageClassification(self):
        self.__mode = "image_classification"

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.__ai.StartObjectClassification(
            classify_perception_model="%s/mobilenet_v2_1.0_224_quant.tflite" % script_dir,
            classify_perception_top_k=1,
            classify_perception_threshold=0,
            classify_perception_labels_file="%s/mobilenet_v2_1.0_224_quant_labels.txt" % script_dir)


    def Stop(self):
        self.__ai.Stop()


    def __change_mode(self):
        body = request.json

        if body["mode"] == "people_counting" and self.__mode != "people_counting":
            self.Stop()
            threading.Thread(target=demo.StartPeopleCounting).start()
        elif body["mode"] == "object_detection" and self.__mode != "object_detection":
            self.Stop()
            threading.Thread(target=demo.StartObjectDetection).start()
        elif body["mode"] == "image_classification" and self.__mode != "image_classification":
            self.Stop()
            threading.Thread(target=demo.StartImageClassification).start()
        
        return Response(status=200)

    def __analyze(self, frame_number, objects, labels=None):
        if self.__mode != "people_counting":
            return

        for object in objects:
            if object.uuid is None:
                continue

            # Check to see if we have already seen this person, otherwise add to the dictionary
            if object.uuid not in self.__seen_people:
                self.__seen_people.append(object.uuid)

    def __draw_object_rectangle_on_frame(self, frame, object, label=None):
        if self.__mode == "people_counting":
            box = object.bounding_box
            cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 1)
            cv2.putText(frame, "{}: {}".format(object.uuid, object.body["face_position"]), (box[0][0] + 2, box[0][1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.__mode == "object_detection":
            box = object.bbox
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            cv2.putText(frame, "%s: %.1f%%" % (label, object.score * 100), (box[0] + 2, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame


    def __frame_processor(self, frame_number, frame, detected_objects, labels=None):
        if len(detected_objects) == 0:
            return frame

        if self.__mode == "image_classification":
            # print("%f: %s" % (detected_objects[0].score, labels[0]))
            return cv2.putText(frame, "{}".format(labels[0]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_clone = frame.copy()
        for idx, object in enumerate(detected_objects):
            label = None
            if labels is not None:
                label = labels[idx]

            frame_clone = self.__draw_object_rectangle_on_frame(frame_clone, object, label)

        if self.__mode == "people_counting":
            cv2.putText(frame_clone, "Count: {}".format(len(self.__seen_people)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
        return frame_clone


    def __root(self):
        return self.__flask_app.send_static_file("index.html")


if __name__ == "__main__":
    demo = Demo()
    threading.Thread(target=demo.StartAPIServer).start()
    
    threading.Thread(target=demo.StartPeopleCounting).start()

    while True:
        time.sleep(1)
