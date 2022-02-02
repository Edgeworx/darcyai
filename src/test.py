import cv2
import os
import pathlib

from darcyai.perceptor.tflite.image_classification_perceptor import ImageClassificationPerceptor
from darcyai.input.camera_stream import CameraStream
from darcyai.output.live_feed_stream import LiveFeedStream
from darcyai.pipeline import Pipeline


def classifier_input_callback(input_data, pom, config):
    return input_data.data.copy()


def live_feed_callback(pom, input_data):
    frame = input_data.data.copy()

    if len(pom.classifier[1]) > 0:
        label = pom.classifier[1][0]
    else:
        label = "---"

    color = (0, 255, 0)
    cv2.putText(frame, str(label), (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


camera = CameraStream(video_device="/dev/video0", fps=20)

pipeline = Pipeline(input_stream=camera)

live_feed = LiveFeedStream(path="/", port=3456, host="0.0.0.0")
pipeline.add_output_stream("output", live_feed_callback, live_feed)

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, "mobilenet_v1_1.0_224_quant.tflite")
labels_file = os.path.join(script_dir, "labels_mobilenet_quant_v1_224.txt")
image_classification = ImageClassificationPerceptor(model_path=model_file,
                                                    num_threads=2,
                                                    threshold=0.9,
                                                    top_k=1,
                                                    labels_file=labels_file)

pipeline.add_perceptor("classifier", image_classification,
                       input_callback=classifier_input_callback)

pipeline.run()
