import os
import threading
import time
from darcyai import DarcyAI


VIDEO_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video0")


if __name__ == "__main__":
    ai = DarcyAI(
        do_perception=False,
        use_pi_camera=False,
        video_device=VIDEO_DEVICE)

    threading.Thread(target=ai.StartVideoStream).start()

    ai.LoadCustomModel('ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')

    while True:
        time.sleep(1)

        _, latest_frame = ai.GetLatestFrame()
        latency, result = ai.RunCustomModel(latest_frame)
