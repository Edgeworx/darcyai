import cv2
import falconn
import math
import numpy as np
import os
import pickle
import threading
import time
import traceback
import uuid
import pathlib
from collections import OrderedDict
from pkg_resources import parse_version

from pycoral import __version__ as pycoral_version
assert parse_version(pycoral_version) >= parse_version('1.0.1'), \
        'darcyai requires PyCoral version >= 1.0.1'
from pycoral.utils import edgetpu, dataset
from pycoral.adapters import common
from pycoral.adapters import detect, classify

from flask import Flask, request, Response
from imutils.video import VideoStream
from darcyai.pose_engine import PoseEngine, KeypointType
from darcyai.config import DarcyAIConfig
from darcyai.object import DetectedObject


class DarcyAI:
    def __init__(
                 self,
                 data_processor=None,
                 frame_processor=None,
                 do_perception=True,
                 use_pi_camera=True,
                 video_device=None,
                 detect_perception_model=None,
                 detect_perception_threshold=None,
                 detect_perception_labels_file=None,
                 classify_perception_model=None,
                 classify_perception_mean=None,
                 classify_perception_std=None,
                 classify_perception_top_k=None,
                 classify_perception_threshold=None,
                 classify_perception_labels_file=None,
                 flask_app=None,
                 video_file=None,
                 video_width=640,
                 video_height=480,
                 config=DarcyAIConfig(),
                 arch=os.uname().machine):
        """
        Initializes DarcyAI Module

        :param data_processor: Callback method to call with detected objects
        :param frame_processor: Callback method to call with detected objects to process frame
        :param do_perception: Whether to do object detection
        :param use_pi_camera: Whether to use PiCamera
        :param video_device: Video device to use
        :param detect_perception_model: Detection model to use
        :param detect_perception_threshold: Detection threshold to use
        :param detect_perception_labels_file: Detection labels file to use
        :param classify_perception_model: Classification model to use
        :param classify_perception_mean: Classification mean to use
        :param classify_perception_std: Classification std to use
        :param classify_perception_top_k: Classification top k to use
        :param classify_perception_threshold: Classification threshold to use
        :param classify_perception_labels_file: Classification labels file to use
        :param flask_app: Flask app to use
        :param video_file: Video file to use
        :param video_width: Video width
        :param video_height: Video height
        :param config: Instance of DarcyAIConfig
        :param arch: Architecture of machine
        """

        if do_perception and data_processor is None:
            raise Exception("data_processor callback is required")

        self.__use_pi_camera = use_pi_camera

        if not use_pi_camera and video_file is None and video_device is None:
            raise Exception("video_device is required")
        self.__video_device = video_device

        self.__data_processor = data_processor
        self.__frame_processor = frame_processor
        self.__classify_perception_model = classify_perception_model
        self.__detect_perception_model = detect_perception_model
        self.__config = config
        self.__do_perception = do_perception

        if do_perception:
            if not classify_perception_model is None:
                if classify_perception_mean is None:
                    raise Exception("classify_perception_mean must be set")

                if classify_perception_std is None:
                    raise Exception("classify_perception_std must be set")

                if classify_perception_top_k is None:
                    raise Exception("classify_perception_top_k must be set")

                if classify_perception_threshold is None:
                    raise Exception("classify_perception_threshold must be set")

                self.__classify_perception_mean = classify_perception_mean
                self.__classify_perception_std  = classify_perception_std
                self.__classify_perception_engine = edgetpu.make_interpreter(classify_perception_model)
                self.__classify_perception_engine.allocate_tensors()
                input_shape = self.__classify_perception_engine.get_input_details()[0]['shape']
                self.__classify_model_inference_shape = (input_shape[2], input_shape[1])

                if not classify_perception_labels_file is None:
                    self.__classify_perception_labels = dataset.read_label_file(classify_perception_labels_file)
                else:
                    self.__classify_perception_labels = None
            elif not detect_perception_model is None:
                if detect_perception_threshold is None:
                    raise Exception("detect_perception_threshold must be set")

                self.__detect_perception_threshold = detect_perception_threshold
                self.__detect_perception_engine = edgetpu.make_interpreter(detect_perception_model)
                self.__detect_perception_engine.allocate_tensors()
                input_shape = self.__detect_perception_engine.get_input_details()[0]['shape']
                self.__detect_model_inference_shape = (input_shape[2], input_shape[1])

                if not detect_perception_labels_file is None:
                    self.__detect_perception_labels = dataset.read_label_file(detect_perception_labels_file)
                else:
                    self.__detect_perception_labels = None
            else:
                script_dir = pathlib.Path(__file__).parent.absolute()
                pose_model_path = os.path.join(script_dir, 'models', 'posenet.tflite')
                self.__pose_engine = self.__get_engine(pose_model_path, arch)

                self.__recent_colors_table_lock = threading.Lock()

                self.__recent_colors_dataset = []
                self.__trained_object_ids = [None]

        self.__custom_engine = None
        self.__custom_engine_inference_shape = None
        self.__custom_engine_output_offsets = [0]

        self.__persons_history = OrderedDict()

        # initialize video camera
        self.__video_file = video_file
        if self.__video_file is None:
            self.__frame_width = video_width
            self.__frame_height = video_height
            self.__vs = self.__initialize_video_camera_stream()
        else:
            self.__vs = self.__initialize_video_file_stream(video_file)

        self.__frame_history = OrderedDict()
        self.__frame_number = 0
        self.__latest_frame = None

        self.__object_number = 0
        self.__recent_objects = {}
        self.__objects_history = {}
        
        self.__flask_app = flask_app
    

    def __get_lsh_params(self, size):
        number_of_tables = 4
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = size
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = number_of_tables
        params_cp.num_rotations = 1
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(8, params_cp)

        return params_cp


    def __setup_recent_colors_table(self):
        if len(self.__recent_colors_dataset) < 2:
            return

        self.__recent_colors_table_lock.acquire()

        try:
            size = (self.__config.GetObjectTrackingColorSamplePixels() ** 2) * 3
            params_cp = self.__get_lsh_params(size)

            falcon_table = falconn.LSHIndex(params_cp)
            falcon_table.setup(np.array(self.__recent_colors_dataset, dtype=np.float64))
    
            self.__recent_colors_query_object = falcon_table.construct_query_object()
        finally:
            self.__recent_colors_table_lock.release()


    def __get_engine(self, path, arch):
        """Initializes object detection engine.

        path: Path to TFLite model
        """

        engine = PoseEngine(model_path=path, arch=arch)
        input_shape = engine.get_input_tensor_shape()
        inference_size = (input_shape[2], input_shape[1])

        return engine


    def __initialize_video_file_stream(self, video_file):
        """Initialize and return video file stream
        """

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise Exception("Cannot open {}".format(video_file))

        self.__frame_width = int(cap.get(3))
        self.__frame_height = int(cap.get(4))

        return cap


    def __initialize_video_camera_stream(self):
        """Initialize and return VideoStream
        """

        if not self.__use_pi_camera:
            vs = VideoStream(src=self.__video_device, resolution=(self.__frame_width, self.__frame_height), framerate=20).start()
        else:
            vs = VideoStream(usePiCamera=True, resolution=(self.__frame_width, self.__frame_height), framerate=20).start()
        test_frame = vs.read()
        counter = 0
        while test_frame is None:
            if counter == 10:
                os._exit(1)

            #Give the camera unit a couple of seconds to start
            print("Waiting for camera unit to start...")
            counter += 1
            time.sleep(1)
            test_frame = vs.read()

        return vs


    def __add_current_frame_to_rolling_history(self, frame):
        """Adds current frame to frame histort.

        frame: Current frame
        """
        milltime = int(round(time.time() * 1000))
        current_frame = {"timestamp" : milltime, "frame" : frame.copy()}
        self.__frame_history[self.__frame_number] = current_frame

        # TODO: make max frames buffer size configurable
        if len(self.__frame_history) > 100:
            self.__frame_history.popitem(last=False)


    def __add_new_object_to_tracker(self):
        """Adds an object to tracker.
        """
        self.__object_number += 1

        self.__object_missing[self.__object_number] = 0
        self.__object_history[self.__object_number] = OrderedDict()
        self.__object_seen[self.__object_number] = OrderedDict()

        return self.__object_number


    def __record_object_seen_value_for_current_frame(self, frame_number, object_id, seen):
        """Sets the object seen/unseen in current frame.

        frame_number: Current frame number  
        object_id: Id of the object to set the flag for  
        seen: True/False
        """
        self.__object_seen[object_id][frame_number] = seen

        if len(self.__object_seen[object_id]) > self.__config.GetPersonTrackingCreationM():
            self.__object_seen[object_id].popitem(last=False)


    def __determine_if_object_seen_enough_to_create_person(self, frame_number, object_id):
        """Sets object as seen when it exists in the frame for a while.

        frame_number: current frame number  
        object_id: id of the object
        """
        ready_for_creation = False

        # We need to see if the key is present for safety otherwise we could throw a "key error"
        if object_id in self.__object_seen:
            # Shortcut processing by looking if we can possibly have enough "seen" occurrences yet
            if len(self.__object_seen[object_id]) >= self.__config.GetPersonTrackingCreationN():
                yes_count = 0

                for frame_number, was_seen in self.__object_seen[object_id].items():
                    if was_seen:
                       yes_count += 1

                if yes_count >= self.__config.GetPersonTrackingCreationN():
                    ready_for_creation = True

        return ready_for_creation


    def __record_object_info_in_tracker_for_id(self, frame_number, object_id, object_info):
        self.__object_missing[object_id] = 0
        self.__object_history[object_id][frame_number] = object_info
        self.__record_object_seen_value_for_current_frame(frame_number, object_id, True)

        if len(self.__object_history[object_id]) > self.__config.GetObjectTrackingInfoHistoryCount():
            self.__object_history[object_id].popitem(last=False)


    def __process_cleanup_of_missing_objects(self):
        for object_id in list(self.__object_missing.keys()):
            if self.__object_missing[object_id] > self.__config.GetObjectTrackingRemovalCount() and object_id in self.__object_data:
                object_data = self.__object_data[object_id]

                del self.__object_missing[object_id]
                del self.__object_history[object_id]
                del self.__object_seen[object_id]
                del self.__object_data[object_id]


    def __mark_unmatched_object_ids_as_missing(self, current_frame_number):
        # Loop through the object history and see if any do not have matches for the current frame
        for object_id in list(self.__object_history.keys()):
            if current_frame_number in self.__object_history[object_id]:
                continue
            else:
                self.__object_missing[object_id] += 1
                self.__record_object_seen_value_for_current_frame(current_frame_number, object_id, False)


    def __check_if_object_data_record_exists_for_object_id(self, object_id):
        if object_id in self.__object_data:
            return True
        else:
            return False


    def __create_new_object_data_record_with_object_id(self, object_id, object_uuid):
        # Create standard dictionary for storing person data with default values
        # The detection history fields are ordered dictionary objects because we will need to pop old readings off the list
        data = {}
        data["uuid"] = object_uuid

        self.__object_data[object_id] = data


    def __upsert_colors(self, colors, object_id):
        index = self.__recent_objects[object_id]["index"]

        if index == len(self.__recent_colors_dataset):
            if len(self.__recent_colors_dataset) == 0:
                self.__recent_colors_dataset = [colors]
            else:
                self.__recent_colors_dataset.append(colors)
        else:
            self.__recent_colors_dataset[index] = np.array([self.__recent_colors_dataset[index], colors]).mean(axis=0)

        if len(self.__recent_colors_dataset) > 1:
            self.__setup_recent_colors_table()


    def __process_new_object(self, object, current_frame_number):
        self.__object_number += 1

        object_id = self.__object_number
        object.object_id = object_id

        object_uuid = str(uuid.uuid4())[0:8]
        object.uuid = object_uuid

        self.__recent_objects[object_id] = {
            "index": len(self.__recent_colors_dataset),
            "uuid": object_uuid,
            "last_seen": current_frame_number,
        }
        self.__recent_objects[object_id]["history"] = OrderedDict()
        self.__recent_objects[object_id]["history"][current_frame_number] = object.tracking_info

        self.__upsert_colors(object.tracking_info["color_sample"], object_id)


    def __experimental_uuid_assignement(self, current_frame_number, objects):
        if len(self.__recent_colors_dataset) == 0:
            [self.__process_new_object(object, current_frame_number) for object in objects]
            
            return

        assigned_object_ids = []
        for idx, object in enumerate(objects):
            possible_matches = []
            tracking_info = object.tracking_info
            face_score_list = OrderedDict()
            body_score_list = OrderedDict()
            face_lowest_score = 1000
            body_lowest_score = 1000
            
            for (object_id, recent_object) in zip(self.__recent_objects.keys(), self.__recent_objects.values()):
                history_num = 0

                vector_face_x = 0
                vector_face_y = 0
                prior_face_x = 0
                prior_face_y = 0
                cum_face_centroid_distance = 0
                cum_face_color_distance = 0
                cum_face_size_distance = 0
                cum_face_centroid_with_vector_distance = 0

                vector_body_x = 0
                vector_body_y = 0
                prior_body_x = 0
                prior_body_y = 0
                cum_body_centroid_distance = 0
                cum_body_color_distance = 0
                cum_body_size_distance = 0
                cum_body_centroid_with_vector_distance = 0

                for history_frame_number in list(recent_object["history"].keys()):
                    history_num += 1

                    # First add to the vector if we are not on the first history frame
                    if history_num > 1:
                        vector_face_x += (recent_object["history"][history_frame_number]["face_centroid"][0] - prior_face_x)
                        vector_face_y += (recent_object["history"][history_frame_number]["face_centroid"][1] - prior_face_y)

                        vector_body_x += (recent_object["history"][history_frame_number]["body_centroid"][0] - prior_body_x)
                        vector_body_y += (recent_object["history"][history_frame_number]["body_centroid"][1] - prior_body_y)

                    prior_face_x = recent_object["history"][history_frame_number]["face_centroid"][0]
                    prior_face_y = recent_object["history"][history_frame_number]["face_centroid"][1]

                    prior_body_x = recent_object["history"][history_frame_number]["body_centroid"][0]
                    prior_body_y = recent_object["history"][history_frame_number]["body_centroid"][1]

                    cum_face_centroid_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["face_centroid"], tracking_info["face_centroid"], 2)
                    cum_face_color_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["face_color"], tracking_info["face_color"], 3)
                    cum_face_size_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["face_size"], tracking_info["face_size"], 2)

                    cum_body_centroid_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["body_centroid"], tracking_info["body_centroid"], 2)
                    cum_body_color_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["body_color"], tracking_info["body_color"], 3)
                    cum_body_size_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["body_size"], tracking_info["body_size"], 2)

                if history_num > 1:
                    vector_face_x = vector_face_x / (history_num - 1)
                    vector_face_y = vector_face_y / (history_num - 1)

                    vector_body_x = vector_body_x / (history_num - 1)
                    vector_body_y = vector_body_y / (history_num - 1)

                projected_face_x = prior_face_x + vector_face_x
                projected_face_y = prior_face_y + vector_face_y

                projected_body_x = prior_body_x + vector_body_x
                projected_body_y = prior_body_y + vector_body_y

                cum_face_centroid_with_vector_distance = self.__euclidean_distance((projected_face_x, projected_face_y), tracking_info["face_centroid"], 2)
                cum_body_centroid_with_vector_distance = self.__euclidean_distance((projected_body_x, projected_body_y), tracking_info["body_centroid"], 2)

                cum_face_centroid_distance = cum_face_centroid_distance / history_num
                cum_face_color_distance = cum_face_color_distance / history_num
                cum_face_size_distance = cum_face_size_distance / history_num

                cum_body_centroid_distance = cum_body_centroid_distance / history_num
                cum_body_color_distance = cum_body_color_distance / history_num
                cum_body_size_distance = cum_body_size_distance / history_num

                # total_current_score_face = (self.__config.GetObjectTrackingCentroidWeight() * cum_face_centroid_distance) + (self.__config.GetObjectTrackingColorWeight() * cum_face_color_distance) + (self.__config.GetObjectTrackingVectorWeight() * cum_face_centroid_with_vector_distance) + (self.__config.GetObjectTrackingSizeWeight() * cum_face_size_distance)
                total_current_score_face = (self.__config.GetObjectTrackingCentroidWeight() * cum_face_centroid_distance) + (self.__config.GetObjectTrackingVectorWeight() * cum_face_centroid_with_vector_distance)
                # total_current_score_face = (self.__config.GetObjectTrackingCentroidWeight() * cum_face_centroid_distance)

                # total_current_score_body = (self.__config.GetObjectTrackingCentroidWeight() * cum_body_centroid_distance) + (self.__config.GetObjectTrackingColorWeight() * cum_body_color_distance) + (self.__config.GetObjectTrackingVectorWeight() * cum_body_centroid_with_vector_distance) + (self.__config.GetObjectTrackingSizeWeight() * cum_body_size_distance)
                total_current_score_body = (self.__config.GetObjectTrackingCentroidWeight() * cum_body_centroid_distance) + (self.__config.GetObjectTrackingVectorWeight() * cum_body_centroid_with_vector_distance)
                # total_current_score_body = (self.__config.GetObjectTrackingCentroidWeight() * cum_body_centroid_distance)

                # print(idx, object_id, total_current_score_body, total_current_score_body)

                face_score_list[object_id] = total_current_score_face
                body_score_list[object_id] = total_current_score_body

                if total_current_score_face < face_lowest_score:
                    face_lowest_score = total_current_score_face

                if total_current_score_body < body_lowest_score:
                    body_lowest_score = total_current_score_body

                if total_current_score_face < 100 and total_current_score_body < 100:
                    possible_matches.append((object_id, self.__recent_objects[object_id]["index"], total_current_score_face, total_current_score_body))

            object.face_score_list = face_score_list
            object.body_score_list = body_score_list

            object.face_lowest_score = face_lowest_score
            object.body_lowest_score = body_lowest_score

            possible_matches = sorted(possible_matches, key=lambda x: x[2])
            std = np.std([x[2] for x in possible_matches])
            filtered_possible_matches = [x for x in possible_matches if x[2] <= face_lowest_score * 1.2 and x[3] <= body_lowest_score * 1.2]

            if len(filtered_possible_matches) > 1:
                # color_matches = self.__recent_colors_query_object.find_k_nearest_neighbors(
                #     object.tracking_info["color_sample"],
                #     k=len(filtered_possible_matches))
                color_matches = self.__recent_colors_query_object.find_near_neighbors(
                    object.tracking_info["color_sample"],
                    threshold=9)
            elif len(filtered_possible_matches) == 1:
                color_matches = [filtered_possible_matches[0][1]]
            elif len(self.__recent_colors_dataset) > 1:
                color_matches = self.__recent_colors_query_object.find_near_neighbors(
                    object.tracking_info["color_sample"],
                    threshold=9)
            else:
                color_matches = []

            for match in color_matches:
                possible_match = next((x for x in filtered_possible_matches if x[1] == match), None)
                if possible_match is not None and possible_match[0] not in assigned_object_ids:
                    object_id = possible_match[0]
                    assigned_object_ids.append(object_id)
                    self.__recent_objects[object_id]["last_seen"] = current_frame_number
                    self.__recent_objects[object_id]["tracking_info"] = object.tracking_info
                    self.__recent_objects[object_id]["history"][current_frame_number] = object.tracking_info

                    object.object_id = object_id
                    object.uuid = self.__recent_objects[object_id]["uuid"]
    
                    break


        for object in objects:
            if object.object_id == 0:
                self.__process_new_object(object, current_frame_number)


    def __apply_best_object_matches_to_object_tracking_info(self, current_frame_number, objects):
        for object in objects:
            tracking_info = object.tracking_info
            face_score_list = OrderedDict()
            face_lowest_score = 1000

            # Loop through existing object ID histories and compute matching
            for existing_object_id in list(self.__object_history.keys()):
                #Loop through the history - keys will be frame numbers - and throw out history entries that are too old
                #List will be oldest entries first
                history_num = 0
                vector_face_x = 0
                vector_face_y = 0
                prior_face_x = 0
                prior_face_y = 0
                cum_face_centroid_distance = 0
                cum_face_color_distance = 0
                cum_face_size_distance = 0
                cum_face_centroid_with_vector_distance = 0

                for history_frame_number in list(recent_object["history"].keys()):
                    # This history entry is valid - proceed
                    history_num += 1

                    # First add to the vector if we are not on the first history frame
                    if history_num > 1:
                        vector_face_x += (recent_object["history"][history_frame_number]["face_centroid"][0] - prior_face_x)
                        vector_face_y += (recent_object["history"][history_frame_number]["face_centroid"][1] - prior_face_y)

                    prior_face_x = recent_object["history"][history_frame_number]["face_centroid"][0]
                    prior_face_y = recent_object["history"][history_frame_number]["face_centroid"][1]

                    cum_face_centroid_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["face_centroid"], tracking_info["face_centroid"], 2)
                    cum_face_color_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["face_color"], tracking_info["face_color"], 3)
                    cum_face_size_distance += self.__euclidean_distance(recent_object["history"][history_frame_number]["face_size"], tracking_info["face_size"], 2)

                if history_num > 1:
                    vector_face_x = vector_face_x / (history_num - 1)
                    vector_face_y = vector_face_y / (history_num - 1)

                projected_face_x = prior_face_x + vector_face_x
                projected_face_y = prior_face_y + vector_face_y

                cum_face_centroid_with_vector_distance = self.__euclidean_distance((projected_face_x, projected_face_y), tracking_info["face_centroid"], 2)

                cum_face_centroid_distance = cum_face_centroid_distance / history_num
                cum_face_color_distance = cum_face_color_distance / history_num
                cum_face_size_distance = cum_face_size_distance / history_num

                total_current_score_face = (self.__config.GetObjectTrackingCentroidWeight() * cum_face_centroid_distance) + (self.__config.GetObjectTrackingColorWeight() * cum_face_color_distance) + (self.__config.GetObjectTrackingVectorWeight() * cum_face_centroid_with_vector_distance) + (self.__config.GetObjectTrackingSizeWeight() * cum_face_size_distance)
                face_score_list[existing_object_id] = total_current_score_face
                if total_current_score_face < face_lowest_score:
                    face_lowest_score = total_current_score_face

            object.face_score_list = face_score_list
            object.face_lowest_score = face_lowest_score

        # Loop through the objects and find the lowest score that is also the lowest score for that object
        for existing_object_id in list(self.__object_history.keys()):
            obj_face_lowest_score = 1000
            obj_best_object_iterator = -1
            object_iter = -1

            for object in objects:
                object_iter += 1
                this_obj_score = object.face_score_list[existing_object_id]

                if this_obj_score < obj_face_lowest_score and object.face_lowest_score == this_obj_score:
                    obj_face_lowest_score = this_obj_score
                    obj_best_object_iterator = object_iter

            if obj_best_object_iterator > -1:
                self.__record_object_info_in_tracker_for_id(current_frame_number, existing_object_id, objects[obj_best_object_iterator].tracking_info)
                objects[obj_best_object_iterator].object_id = existing_object_id
                #Check if we have a person data record yet
                has_object = self.__check_if_object_data_record_exists_for_object_id(existing_object_id)
                if has_object:
                    #Add the person ID to the object
                    objects[obj_best_object_iterator].object_id = existing_object_id
                    objects[obj_best_object_iterator].uuid = self.__object_data[existing_object_id]["uuid"]
                else:
                    #See if we can create one based on our appearance history
                    object_ready = self.__determine_if_object_seen_enough_to_create_person(current_frame_number, existing_object_id)
                    object_uuid = str(uuid.uuid4())[0:8]
                    self.__create_new_object_data_record_with_object_id(existing_object_id, object_uuid)
                    objects[obj_best_object_iterator].object_id = existing_object_id
                    objects[obj_best_object_iterator].uuid = object_uuid

        #Loop through objects one last time to see if any do not have an object ID yet
        for object in objects:
            if object.object_id == 0:
                new_object_id = self.__add_new_object_to_tracker()
                object.object_id = new_object_id
                self.__object_history[new_object_id][current_frame_number] = object.tracking_info
                self.__object_missing[new_object_id] = 0


    def __euclidean_distance(self, point1, point2, coordinate_count):
        tmp_sum = 0

        for i in range(coordinate_count):
            tmp_sum += (point1[i] - point2[i]) ** 2
        
        distance = tmp_sum ** 0.5

        return distance
    

    def __normalize_embeddings(self, embeddings):
        return (embeddings / np.sqrt((embeddings ** 2).sum())).astype(np.float64)


    def __get_centroid(self, frame, bbox):
        centroid = (0, 0)
        size = (0, 0)
        average_color = (0, 0, 0)

        start_x = bbox[0][0]
        start_y = bbox[0][1]
        end_x = bbox[1][0]
        end_y = bbox[1][1]

        size_w = int(end_x - start_x)
        size_h = int(end_y - start_y)
        centroid_x = int((start_x + end_x) / 2)
        centroid_y = int((start_y + end_y) / 2)

        size = (size_w, size_h)
        centroid = (centroid_x, centroid_y)

        centroid_color_sample_start_x = max(centroid_x - int(self.__config.GetObjectTrackingColorSamplePixels() / 2), 0)
        centroid_color_sample_end_x = min(centroid_x + int(self.__config.GetObjectTrackingColorSamplePixels() / 2), self.__frame_width)
        centroid_color_sample_start_y = max(centroid_y - int(self.__config.GetObjectTrackingColorSamplePixels() / 2), 0)
        centroid_color_sample_end_y = min(centroid_y + int(self.__config.GetObjectTrackingColorSamplePixels() / 2), self.__frame_height)

        color_sample_chunk = frame[centroid_color_sample_start_y : centroid_color_sample_end_y, centroid_color_sample_start_x : centroid_color_sample_end_x]

        #Find the average pixel color values for BGR
        avg_blue = 0
        avg_green = 0
        avg_red = 0

        avg_color_per_row = np.average(color_sample_chunk, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        avg_blue = int(avg_color[0])
        avg_green = int(avg_color[1])
        avg_red = int(avg_color[2])

        average_color = (avg_blue, avg_green, avg_red)

        return centroid, size, average_color, color_sample_chunk


    def __generate_tracking_info_for_object(self, frame, body):
        tracking_info = {}

        face_centroid, face_size, face_average_color, _ = self.__get_centroid(frame, body["face_rectangle"])
        body_centroid, body_size, body_average_color, color_sample_chunk = self.__get_centroid(frame, body["body_rectangle"])

        tracking_info["face_size"] = face_average_color
        tracking_info["face_centroid"] = face_centroid
        tracking_info["face_color"] = face_average_color

        tracking_info["body_size"] = body_average_color
        tracking_info["body_centroid"] = body_centroid
        tracking_info["body_color"] = body_average_color
        tracking_info["color_sample"] = self.__normalize_embeddings(color_sample_chunk.flatten())

        # tracking_info["embeddings"] = self.__get_embeddings(frame, body)
        tracking_info["face_position"] = body["face_position"]

        return tracking_info


    def __encode_jpeg(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        img_encode = cv2.imencode(".jpg", frame, encode_param)[1]
        data_encode = np.array(img_encode)
        return data_encode.tobytes()


    def __get_object_bounding_box(self, body):
        start_x = 0
        start_y = 0
        end_x = 0
        end_y = 0

        if body["has_face"]:
            start_x = body["face_rectangle"][0][0]
            start_y = body["face_rectangle"][0][1]
            end_x = body["face_rectangle"][1][0]
            end_y = body["face_rectangle"][1][1]
        else:
            start_x = body["body_rectangle"][0][0]
            start_y = body["body_rectangle"][0][1]
            end_x = body["body_rectangle"][1][0]
            end_y = body["body_rectangle"][1][1]

        return (start_x, start_y), (end_x, end_y)


    def __generate_stream(self):
        while True:
            try:
                while self.__latest_frame is None:
                    time.sleep(0.10)
                    continue

                time.sleep(1.0 / 10)
                frame = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + self.__encode_jpeg(self.__latest_frame)
                yield(frame)
            except Exception as e:
                tb = traceback.format_exc()
                print("Error at generating stream {}".format(tb))
                pass


    def __live_feed(self): 
        response = Response(self.__generate_stream(), 
                            mimetype="multipart/x-mixed-replace; boundary=frame") 
        
        return response


    def __start_api_server(self):
        if self.__flask_app is None:
            self.__flask_app = Flask(__name__)
            ssl_context = None
            self.__flask_app.add_url_rule("/live-feed", "__live_feed", self.__live_feed)
            self.__flask_app.run(
                host="0.0.0.0",
                port=self.__config.GetLiveStreamPort(),
                ssl_context=ssl_context,
                debug=False)
        else:
            self.__flask_app.add_url_rule("/live-feed", "__live_feed", self.__live_feed)


    def __get_qualified_body_detections(self, poses, with_face=False, with_body=False):
        #Loop through all raw detected poses and return an array of qualified body dictionaries
        bodies = []

        for pose in poses:
            curBody = {}
            meetsConfidence = False
            meetsSize = False
            curBody["person_id"] = 0
            curBody["has_face"] = False
            curBody["has_body"] = False
            curBody["has_forehead"] = False
            curBody["face_score"] = 0.0
            curBody["body_score"] = 0.0
            curBody["body_rectangle"] = ((0,0),(0,0))
            curBody["face_rectangle"] = ((0,0),(0,0))
            curBody["body_id"] = 0

            #Compute the confidence scores for face and body separately
            faceScore = (pose.keypoints[KeypointType.NOSE].score + pose.keypoints[KeypointType.LEFT_EYE].score + pose.keypoints[KeypointType.RIGHT_EYE].score + pose.keypoints[KeypointType.LEFT_EAR].score + pose.keypoints[KeypointType.RIGHT_EAR].score) / 5
            bodyScore = (pose.keypoints[KeypointType.LEFT_SHOULDER].score + pose.keypoints[KeypointType.RIGHT_SHOULDER].score + pose.keypoints[KeypointType.LEFT_ELBOW].score + pose.keypoints[KeypointType.RIGHT_ELBOW].score + pose.keypoints[KeypointType.LEFT_WRIST].score + pose.keypoints[KeypointType.RIGHT_WRIST].score + pose.keypoints[KeypointType.LEFT_HIP].score + pose.keypoints[KeypointType.RIGHT_HIP].score + pose.keypoints[KeypointType.LEFT_KNEE].score + pose.keypoints[KeypointType.RIGHT_KNEE].score + pose.keypoints[KeypointType.LEFT_ANKLE].score + pose.keypoints[KeypointType.RIGHT_ANKLE].score) / 12

            #Evaluate confidence levels first
            if faceScore >= self.__config.GetPoseMinimumFaceThreshold():
                meetsConfidence = True
                curBody["has_face"] = True
                curBody["face_score"] = faceScore
            
            if bodyScore >= self.__config.GetPoseMinimumBodyThreshold():
                meetsConfidence = True
                curBody["has_body"] = True
                curBody["body_score"] = bodyScore

            #Now check for size requirements
            if meetsConfidence:
                #Get face and body sizes from keypoint coordinates in rough estimate style
                faceHeight = ((pose.keypoints[KeypointType.LEFT_SHOULDER].point[1] + pose.keypoints[KeypointType.RIGHT_SHOULDER].point[1]) / 2) - ((pose.keypoints[KeypointType.LEFT_EYE].point[1] + pose.keypoints[KeypointType.RIGHT_EYE].point[1]) / 2)
                bodyHeight = ((pose.keypoints[KeypointType.LEFT_ANKLE].point[1] + pose.keypoints[KeypointType.RIGHT_ANKLE].point[1]) / 2) - pose.keypoints[KeypointType.NOSE].point[1]

                if faceHeight >= self.__config.GetPoseMinimumFaceHeight():
                    meetsSize = True
                    curBody["simple_face_height"] = faceHeight

                if bodyHeight >= self.__config.GetPoseMinimumBodyHeight():
                    meetsSize = True
                    curBody["simple_body_height"] = bodyHeight

                #If we meet both size and confidence requirements, put this body in the output set
                if meetsSize and (not with_face or curBody["has_face"]) and (not with_body or curBody["has_body"]):
                    curBody["pose"] = pose
                    bodies.append(curBody)

        return bodies


    def __determine_face_position(self, body):
        #Calculate combo scores of different sets
        pose = body["pose"]
        leftEyeLeftEarScore = (pose.keypoints[KeypointType.LEFT_EYE].score + pose.keypoints[KeypointType.LEFT_EAR].score) / 2
        rightEyeRightEarScore = (pose.keypoints[KeypointType.RIGHT_EYE].score + pose.keypoints[KeypointType.RIGHT_EAR].score) / 2
        eyeAndNoseScore = (pose.keypoints[KeypointType.NOSE].score + pose.keypoints[KeypointType.RIGHT_EYE].score + pose.keypoints[KeypointType.LEFT_EYE].score) / 3
        rightness = leftEyeLeftEarScore - rightEyeRightEarScore
        leftness = rightEyeRightEarScore - leftEyeLeftEarScore

        facePosition = 'Unknown'

        if body["has_face"]:
            if rightness > (-1 * self.__config.GetFacePositionLeftRightThreshold()) and rightness < self.__config.GetFacePositionLeftRightThreshold() and leftness > (-1 * self.__config.GetFacePositionLeftRightThreshold()) and leftness < self.__config.GetFacePositionLeftRightThreshold() and eyeAndNoseScore >= self.__config.GetFacePositionStraightThreshold():
                facePosition = "Straight"
            elif rightness >= self.__config.GetFacePositionLeftRightThreshold() and leftness <= (-1 * self.__config.GetFacePositionLeftRightThreshold()):
                facePosition = "Right"
            elif leftness >= self.__config.GetFacePositionLeftRightThreshold() and rightness <= (-1 * self.__config.GetFacePositionLeftRightThreshold()):
                facePosition = "Left"
            else:
                facePosition = "Away"
        else:
            facePosition = "Away"

        return facePosition


    def __determine_if_forehead_visible(self, body):
        hasForehead = False

        if body["has_face"] and body["face_position"] != "Away":
            hasForehead = True

        return hasForehead


    def __determine_forehead_center(self, body):
        foreheadCenter = (0,0)

        if body["has_forehead"]:
            #Get eye locations
            leftEyeX = int(body["pose"].keypoints[KeypointType.LEFT_EYE].point[0])
            leftEyeY = int(body["pose"].keypoints[KeypointType.LEFT_EYE].point[1])
            rightEyeX = int(body["pose"].keypoints[KeypointType.RIGHT_EYE].point[0])
            rightEyeY = int(body["pose"].keypoints[KeypointType.RIGHT_EYE].point[1])
            
            #Adjust the eye vertical position by 1 pixel if they are exactly equal so we don't have 0 slope
            dY = rightEyeY - leftEyeY
            if dY == 0:
                dY = 1

            dX = rightEyeX - leftEyeX
            if dX == 0:
                dX = 1    

            #Calculate distance between eyes and set distance to forehead
            eyeSlope = dY / dX
            eyeDistance = math.sqrt((dX * dX) + (dY * dY))
            distanceToForehead = eyeDistance * 0.5
            inverseEyeSlope = -1 / eyeSlope
            midX = (leftEyeX + rightEyeX) / 2
            midY = (leftEyeY + rightEyeY) / 2
            endConstant = distanceToForehead / math.sqrt(1 + (inverseEyeSlope * inverseEyeSlope))
            foreheadX = 0
            foreheadY = 0

            if eyeSlope < 0:
                foreheadX = int(midX - endConstant)
                foreheadY = int(midY - (endConstant * inverseEyeSlope))
            else:
                foreheadX = int(midX + endConstant)
                foreheadY = int(midY + (endConstant * inverseEyeSlope))

            foreheadCenter = (foreheadX, foreheadY)

        return foreheadCenter


    def __find_body_rectangle(self, body):
        bodyRectangle = ((0, 0),(0, 0))

        # if body["has_body"]:
        #     #We have a high enough confidence that the body points are real so let's use them to make a rectangle
        #     #Find the lowest and highest X and Y and then add a few pixels of padding
        #     lowestY = self.__frame_height
        #     lowestX = self.__frame_width
        #     highestY = 0
        #     highestX = 0

        #     for label, keypoint in body["pose"].keypoints.items():
        #         faceLabels = {KeypointType.NOSE, KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE, KeypointType.LEFT_EAR, KeypointType.RIGHT_EAR}
        #         if label in faceLabels:
        #             continue
        #         else:
        #             if keypoint.point[0] < lowestX: lowestX = keypoint.point[0]
        #             if keypoint.point[1] < lowestY: lowestY = keypoint.point[1]
        #             if keypoint.point[0] > highestX: highestX = keypoint.point[0]
        #             if keypoint.point[1] > highestY: highestY = keypoint.point[1]

        #     bodyRectangle = ((int(lowestX - 2), int(lowestY - 2)),(int(highestX + 2), int(highestY + 2)))
        # elif body["has_face"]:
        start_x = body["face_rectangle"][0][0]
        start_y = body["face_rectangle"][0][1]
        end_x = body["face_rectangle"][1][0]
        end_y = body["face_rectangle"][1][1]

        face_width = end_x - start_x
        face_height = end_y - start_y

        bodyRectangle = (
            (max(int(start_x - (face_width * 0.5)), 0), min(int(end_y), self.__frame_height)), 
            (min(int(end_x + (face_width * 0.5)), self.__frame_width), min(int(end_y + (face_height * 2)), self.__frame_height)))

        return bodyRectangle


    def __find_face_rectangle(self, body):
        faceRectangle = ((0,0),(0,0))

        if not body["has_face"]:
            return faceRectangle

        #We have a high enough confidence that the face points are real so let's use them to make a rectangle
        #Find the lowest and highest X and then add a few pixels of padding
        lowestY = self.__frame_height
        lowestX = self.__frame_width
        highestY = 0
        highestX = 0

        for label, keypoint in body["pose"].keypoints.items():
            faceLabels = {KeypointType.NOSE, KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE, KeypointType.LEFT_EAR, KeypointType.RIGHT_EAR}
            if label in faceLabels:
                lowestX = min(keypoint.point[0], lowestX)
                lowestY = min(keypoint.point[1], lowestY)
                highestX = max(keypoint.point[0], highestX)
                highestY = max(keypoint.point[1], highestY)
            else:
                continue

        xSpread = highestX - lowestX
        ySpread = highestY - lowestY
        if ySpread == 0:
            ySpread = 1

        lowestX -= 8
        highestX += 8
        xFactor = xSpread / ySpread * 0.8

        yExpand = ySpread * xFactor * self.__config.GetFaceRectangleYFactor()
        lowestY -= yExpand
        highestY += yExpand

        lowestX = max(lowestX, 0)
        lowestY = max(lowestY, 0)
        highestX = min(highestX, self.__frame_width)
        highestY = min(highestY, self.__frame_height)

        faceRectangle = ((int(lowestX), int(lowestY)), (int(highestX), int(highestY)))

        return faceRectangle

    
    def __record_person_history(self, detected_objects):
        for object in list(detected_objects):
            object_id = object.object_id
            if object_id in self.__object_history:
                if not object_id in self.__persons_history:
                    self.__persons_history[object_id] = OrderedDict()

                self.__persons_history[object_id][self.__frame_number] = object


    def __people_perception(self, frame):
        poses, _ = self.__pose_engine.DetectPosesInImage(frame)
        bodies = self.__get_qualified_body_detections(poses, with_face=True, with_body=False)

        detected_objects = []
        for body in bodies:
            body["face_position"] = self.__determine_face_position(body)
            body["has_forehead"] = self.__determine_if_forehead_visible(body)
            body["forehead_center"] = self.__determine_forehead_center(body)
            body["face_rectangle"] = self.__find_face_rectangle(body)
            body["body_rectangle"] = self.__find_body_rectangle(body)

            tracking_info = self.__generate_tracking_info_for_object(frame, body)
            # cv2.circle(frame, tracking_info["face_centroid"], 5, (0, 255, 0), -1)
            # cv2.circle(frame, tracking_info["body_centroid"], 5, (0, 0, 255), -1)
            detected_object = DetectedObject()
            detected_object.bounding_box = self.__get_object_bounding_box(body)
            detected_object.tracking_info = tracking_info
            detected_object.body = body
            detected_objects.append(detected_object)

        self.__experimental_uuid_assignement(self.__frame_number, detected_objects)

        return detected_objects


    def Start(self):
        threading.Thread(target=self.__start_api_server).start()

        while True:
            try:
                if self.__video_file is None:
                    frame = self.__vs.read()

                    counter = 0
                    while frame is None:
                        if counter == 10:
                            os._exit(1)

                        counter += 1
                        time.sleep(1)
                        frame = self.__vs.read()
                else:
                    if not self.__vs.isOpened():
                        break

                    success, frame = self.__vs.read()
                    if not success:
                        break

                if self.__config.GetFlipVideoFrame():
                    frame = cv2.flip(frame, 1)

                self.__frame_number += 1

                self.__add_current_frame_to_rolling_history(frame)

                if self.__do_perception:
                    labels = None
                    if not self.__detect_perception_model is None:
                        labels = []
                        _, scale = common.set_resized_input(
                            self.__detect_perception_engine,
                            (frame.shape[1], frame.shape[0]),
                            lambda size: cv2.resize(frame, size))
                        start = time.perf_counter()
                        self.__detect_perception_engine.invoke()
                        inference_time = time.perf_counter() - start
                        detected_objects = detect.get_objects(self.__detect_perception_engine, self.__detect_perception_threshold, scale)
                        if not self.__detect_perception_labels is None:
                            for object in detected_objects:
                                labels.append(self.__detect_perception_labels.get(object.id, object.id))
                    elif not self.__classify_perception_model is None:
                        labels = []
                        params = common.input_details(self.__classify_perception_engine, 'quantization_parameters')
                        scale = params['scales']
                        zero_point = params['zero_points']
                        mean = self.__classify_perception_mean
                        std = self.__classify_perception_std
                        if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
                            # Input data does not require preprocessing.
                            common.set_input(self.__classify_perception_engine, frame)
                        else:
                            # Input data requires preprocessing
                            normalized_input = (np.asarray(frame) - mean) / (std * scale) + zero_point
                            np.clip(normalized_input, 0, 255, out=normalized_input)
                            common.set_input(self.__classify_perception_engine, normalized_input.astype(np.uint8))

                        start = time.perf_counter()
                        self.__classify_perception_engine.invoke()
                        inference_time = time.perf_counter() - start
                        detected_objects = classify.get_classes(self.__classify_perception_engine, 0, 0)
                        if not self.__classify_perception_labels is None:
                            for object in detected_objects:
                                labels.append(self.__classify_perception_labels.get(object.id, object.id))
                    else:
                        detected_objects = self.__people_perception(frame)

                    if labels is None:
                        self.__data_processor(self.__frame_number, detected_objects)
                    else:
                        self.__data_processor(self.__frame_number, detected_objects, labels)
                else:
                    detected_objects = None

                if not self.__frame_processor is None:
                    if labels is None:
                        self.__latest_frame = self.__frame_processor(self.__frame_number, frame, detected_objects)
                    else:
                        self.__latest_frame = self.__frame_processor(self.__frame_number, frame, detected_objects, labels)
                else:
                    self.__latest_frame = frame
            except Exception as e:
                tb = traceback.format_exc()
                print("Error at generating stream {}".format(tb))
                pass


    def LoadCustomModel(self, model_path):
        if not self.__custom_engine is None:
            raise Exception("A custom model is already loaded")

        self.__custom_engine = edgetpu.make_interpreter(model_path)
        self.__custom_engine.allocate_tensors()
        input_shape = self.__custom_engine.get_input_details()[0]['shape']
        self.__custom_engine_inference_shape = (input_shape[2], input_shape[1])

    
    def RunCustomModel(self, frame):
        if self.__custom_engine is None:
            raise Exception("No custom model is loaded")

        for_custom_engine = cv2.resize(frame, self.__custom_engine_inference_shape)
        common.set_input(self.__custom_engine, for_custom_engine)

        start = time.perf_counter()
        self.__custom_engine.invoke()
        inference_time = time.perf_counter() - start
        outputs = common.output_tensor(self.__custom_engine, 0)

        return inference_time * 1000, outputs


    def GetPersonHistory(self, person_id):
        if not self.__do_perception or not self.__classify_perception_model is None:
            raise Exception("People perception is disabled")
            
        if not person_id in self.__persons_history:
            return None

        return self.__persons_history[person_id]


    def SetConfig(config):
        self.__config = config


    def GetLatestFrame(self):
        return self.__frame_number, self.__latest_frame


if __name__ == "__main__":
    pass