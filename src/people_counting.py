import cv2
import os
import threading
from darcyai import DarcyAI, DarcyAIConfig
from flask import Flask, request, Response


VIDEO_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video0")


class PeopleCounting:
    def __init__(self):
        self.__seen_people = []

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.__flask_app = Flask(__name__, static_url_path=script_dir)
        self.__flask_app.add_url_rule("/", "root", self.__root)

        darcy_ai_config = DarcyAIConfig(
            face_rectangle_yfactor=0.8,
            pose_minimum_face_threshold=0.5,
            object_tracking_color_sample_pixels=16)
        
        self.__ai = DarcyAI(
            config=darcy_ai_config,
            data_processor=self.__analyze,
            frame_processor=self.__frame_processor,
            flask_app=self.__flask_app,
            arch="armv7l",
            use_pi_camera=False,
            video_device=VIDEO_DEVICE)


    def Start(self):
        threading.Thread(target=self.__ai.Start).start()

        self.__flask_app.run(
            host="0.0.0.0",
            port=3456,
            debug=False)


    def __analyze(self, frame_number, objects):
        for object in objects:
            if object.uuid is None:
                continue

            if object.uuid not in self.__seen_people:
                self.__seen_people.append(object.uuid)


    def __draw_object_rectangle_on_frame(self, frame, object):
        box = object.bounding_box
        cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 1)
        cv2.putText(frame, "{}: {}".format(object.uuid, object.body["face_position"]), (box[0][0] + 2, box[0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


    def __frame_processor(self, frame_number, frame, detected_objects):
        frame_clone = frame.copy()
        for object in detected_objects:
            frame_clone = self.__draw_object_rectangle_on_frame(frame_clone, object)

        cv2.putText(frame_clone, "Count: {}".format(len(self.__seen_people)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame_clone


    def __root(self):
        return self.__flask_app.send_static_file('index.html')


if __name__ == "__main__":
    people_counting = PeopleCounting()
    people_counting.Start()
