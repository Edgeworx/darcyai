import collections
import cv2
import numpy as np
import operator
import re
from importlib import import_module
from typing import Any, List

from darcyai.config_registry import ConfigRegistry
from darcyai.utils import validate_not_none, validate_type, validate

from .dataset import read_label_file
from .tflite_perceptor_base import TFLitePerceptorBase


Class = collections.namedtuple("Class", ["id", "score"])

class ImageClassificationPerceptor(TFLitePerceptorBase):
    """
    ImageClassificationPerceptor is a class that implements the Perceptor interface for
    image classification.

    Arguments:
        threshold (float): The threshold for object detection.
        top_k (int): The number of top predictions to return.
        labels_file (str): The path to the labels file.
        labels (dict): A dictionary of labels.
        mean (float): The mean of the image.
        std (float): The standard deviation of the image.
        **kwargs: Keyword arguments to pass to Perceptor.
    """


    def __init__(self,
                 threshold:float,
                 top_k:int=None,
                 labels_file:str=None,
                 labels:dict=None,
                 mean:float=128.0,
                 std:float=128.0,
                 **kwargs):
        super().__init__(**kwargs)

        validate_not_none(threshold, "threshold is required")
        validate_type(threshold, (float, int), "threshold must be a number")
        validate(0 <= threshold <= 1, "threshold must be between 0 and 1")

        if top_k is not None:
            validate_type(top_k, int, "top_k must be an integer")
            validate(top_k > 0, "top_k must be greater than 0")

        if labels is not None:
            validate_type(labels, dict, "labels must be a dictionary")
            self.__labels = labels
        elif labels_file is not None:
            validate_type(labels_file, str, "labels_file must be a string")
            self.__labels = read_label_file(labels_file)
        else:
            self.__labels = None

        validate_type(mean, (int, float), "mean must be a number")
        validate_type(std, (int, float), "std must be a number")

        self.__threshold = threshold
        self.__top_k = top_k
        self.__mean = mean
        self.__std = std


    def run(self, input_data:Any, config:ConfigRegistry=None) -> (List[Any], List[str]):
        """
        Runs the image classification model.

        Arguments:
            input_data (Any): The input data to run the model on.
            config (ConfigRegistry): The configuration for the perceptor.

        Returns:
            (list[Any], list(str)): A tuple containing the detected classes and the labels.
        """
        labels = []
        resized_frame = cv2.resize(input_data, self.input_shape)

        if "quantization_parameters" in self.input_details[0]:
            params = self.input_details[0]["quantization_parameters"]
            scales = params["scales"]
            zero_points = params["zero_points"]

            if abs(scales * self.__std - 1) < 1e-5 and abs(self.__mean - zero_points) < 1e-5:
                input_tensor = np.array(np.expand_dims(resized_frame, 0))
                self.interpreter.set_tensor(self.input_index, input_tensor)
            else:
                normalized_frame = (np.asarray(resized_frame) - self.__mean) / (self.__std * scales) \
                    + zero_points
                np.clip(normalized_frame, 0, 255, out=normalized_frame)
                input_tensor = np.array(np.expand_dims(normalized_frame, 0))
                self.interpreter.set_tensor(self.input_index, input_tensor)
        else:
            input_tensor = np.array(np.expand_dims(resized_frame, 0))
            self.interpreter.set_tensor(self.input_index, input_tensor)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index).flatten()

        if self.__top_k is not None:
            top_k = min(self.__top_k, len(output_data))
        else:
            top_k = len(output_data)

        classes = [
            Class(i, output_data[i])
            for i in np.argpartition(output_data, -top_k)[-top_k:]
            if output_data[i] >= (self.__threshold * 100)
        ]
        detected_classes = sorted(classes, key=operator.itemgetter(1), reverse=True)

        if not self.__labels is None:
            for detected_object in detected_classes:
                labels.append(self.__labels.get(detected_object.id, detected_object.id))

        return detected_classes, labels
