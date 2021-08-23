import cv2
import os
import threading
from darcyai import DarcyAI, DarcyAIConfig
from flask import Flask, request, Response


VIDEO_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video0")
EMBEDDINGS_FILENAME = os.getenv("EMBEDDINGS_FILENAME", "embeddings.mapping")

seen_people = []

def analyze(frame_number, objects):
    for object in objects:
        if object.uuid is None:
            continue

        if object.uuid not in seen_people:
            seen_people.append(object.uuid)


def draw_object_rectangle_on_frame(frame, object):
    box = object.bounding_box
    bbox = object.body["body_rectangle"]
    cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 1)
    cv2.rectangle(frame, bbox[0], bbox[1], (0, 0, 255), 1)
    cv2.putText(frame, "{}: {}".format(object.uuid, object.body["face_position"]), (box[0][0] + 2, box[0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def frame_processor(frame_number, frame, detected_objects):
    frame_clone = frame.copy()
    for object in detected_objects:
        frame_clone = draw_object_rectangle_on_frame(frame_clone, object)

    cv2.putText(frame_clone, "Count: {}".format(len(seen_people)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame_clone


def root():
    return flask_app.send_static_file('index.html')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    flask_app = Flask(__name__, static_url_path=script_dir)
    flask_app.add_url_rule("/", "root", root)

    darcy_ai_config = DarcyAIConfig(
        embeddings_filename=EMBEDDINGS_FILENAME,
        face_rectangle_yfactor=0.8,
        object_tracking_color_sample_pixels=16)
    
    ai = DarcyAI(
        config=darcy_ai_config,
        data_processor=analyze,
        frame_processor=frame_processor,
        flask_app=flask_app,
        arch="armv7l",
        use_pi_camera=False,
        video_device=VIDEO_DEVICE)
    threading.Thread(target=ai.Start).start()

    flask_app.run(
        host="0.0.0.0",
        port=3456,
        debug=False)
