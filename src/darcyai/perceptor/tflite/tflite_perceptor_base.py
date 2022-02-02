from importlib import import_module
from typing import Union

from darcyai.perceptor.perceptor import Perceptor
from darcyai.utils import validate_type, validate

class TFLitePerceptorBase(Perceptor):
    """
    Base class for all TFLite Perceptors.

    Arguments:
        num_threads (int): The number of CPU threads to be used.
            Defaults to 1.
    """
    def __init__(self, num_threads:int=1, **kwargs):
        super().__init__(**kwargs)

        validate_type(num_threads, int, "num_threads must be an integer")
        validate(num_threads > 0, "num_threads must be greater than 0")

        self.__num_threads = num_threads

        self.interpreter = None
        self.input_details = None
        self.input_shape = None
        self.input_index = None
        self.output_index = None

    def load(self, accelerator_idx: Union[int, None] = None) -> None:
        """
        Loads the perceptor.

        # Arguments
        accelerator_idx (int, None): Not used.
        """
        tflite_runtime = load_delegate = import_module("tflite_runtime.interpreter")
        Interpreter = tflite_runtime.Interpreter

        self.interpreter = Interpreter(model_path=self.model_path, num_threads=self.__num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()

        input_shape = self.input_details[0]["shape"]
        self.input_shape = (input_shape[2], input_shape[1])

        self.input_index = self.input_details[0]["index"]

        output_details = self.interpreter.get_output_details()
        self.output_index = output_details[0]["index"]
