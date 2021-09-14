import cv2
import os
import threading
from darcyai import DarcyAI, DarcyAIConfig
from flask import Flask, request


MIN_FACE_HEIGHT = int(os.getenv("MIN_FACE_HEIGHT", 40))
MIN_BODY_HEIGHT = int(os.getenv("MIN_BODY_HEIGHT", 240))
VIDEO_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video0")


def analyze(frame_number, objects):
    for object in objects:
        if object.body["has_face"]:
            print("{}: {}".format(object.object_id, object.body["face_position"]))
        else:
            print("{}: No face".format(object.object_id))


def draw_object_rectangle_on_frame(frame, object):
    box = object.bounding_box
    cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 1)
    cv2.putText(frame, "{}: {}".format(object.uuid, object.body["face_position"]), (box[0][0] + 2, box[0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame


def frame_processor(frame_number, frame, detected_objects):
    frame_clone = frame.copy()
    for object in detected_objects:
        frame_clone = draw_object_rectangle_on_frame(frame_clone, object)

    return frame_clone


def config_handler(config):
    def config_helper():
        body = request.json
        try:
            for key in body:
                config.Set(key, body[key])

            return "", 200
        except Exception as e:
            return str(e), 400

    return config_helper


if __name__ == "__main__":
    config = DarcyAIConfig(
        pose_minimum_face_height=MIN_FACE_HEIGHT,
        pose_minimum_body_height=MIN_BODY_HEIGHT)

    flask_app = Flask(__name__)
    flask_app.add_url_rule("/config", "config", config_handler(config), methods=["PUT"])

    ai = DarcyAI(
        data_processor=analyze,
        frame_processor=frame_processor,
        flask_app=flask_app,
        use_pi_camera=False,
        video_device=VIDEO_DEVICE,
        config=config)
    threading.Thread(target=ai.StartPeoplePerception).start()

    flask_app.run(
        host="0.0.0.0",
        port=3456,
        debug=False)
