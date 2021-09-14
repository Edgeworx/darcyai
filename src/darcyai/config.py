class ConfigProperty:
  def __init__(self, setter, getter):
    self.__setter = setter
    self.__getter = getter


  def Get(self):
    return self.__getter()


  def Set(self, value):
    return self.__setter(value)


class DarcyAIConfig:
  def __init__(
               self,
               flip_video_frame=False,
               person_tracking_creation_m=20,
               person_tracking_creation_n=16,
               object_tracking_info_history_count=3,
               object_tracking_removal_count=50,
               object_tracking_centroid_weight=0.25,
               object_tracking_color_weight=0.25,
               object_tracking_vector_weight=0.25,
               object_tracking_size_weight=0.25,
               object_tracking_color_sample_pixels=4,
               pose_minimum_face_threshold=0.4,
               pose_minimum_body_threshold=0.2,
               pose_minimum_face_height=40,
               pose_minimum_body_height=240,
               face_position_left_right_threshold=0.3,
               face_position_straight_threshold=0.7,
               face_rectangle_yfactor=1.0,
               live_stream_port=3456,
               embeddings_filename=None):

    self.__properties = {
      "flip_video_frame": ConfigProperty(self.SetFlipVideoFrame, self.GetFlipVideoFrame),
      "person_tracking_creation_m": ConfigProperty(self.SetPersonTrackingCreationM, self.GetPersonTrackingCreationM),
      "person_tracking_creation_n": ConfigProperty(self.SetPersonTrackingCreationN, self.GetPersonTrackingCreationN),
      "object_tracking_info_history_count": ConfigProperty(self.SetObjectTrackingInfoHistoryCount, self.GetObjectTrackingInfoHistoryCount),
      "object_tracking_removal_count": ConfigProperty(self.SetObjectTrackingRemovalCount, self.GetObjectTrackingRemovalCount),
      "object_tracking_centroid_weight": ConfigProperty(self.SetObjectTrackingCentroidWeight, self.GetObjectTrackingCentroidWeight),
      "object_tracking_color_weight": ConfigProperty(self.SetObjectTrackingColorWeight, self.GetObjectTrackingColorWeight),
      "object_tracking_vector_weight": ConfigProperty(self.SetObjectTrackingVectorWeight, self.GetObjectTrackingVectorWeight),
      "object_tracking_size_weight": ConfigProperty(self.SetObjectTrackingSizeWeight, self.GetObjectTrackingSizeWeight),
      "object_tracking_color_sample_pixels": ConfigProperty(self.SetObjectTrackingColorSamplePixels, self.GetObjectTrackingColorSamplePixels),
      "pose_minimum_face_threshold": ConfigProperty(self.SetPoseMinimumFaceThreshold, self.GetPoseMinimumFaceThreshold),
      "pose_minimum_body_threshold": ConfigProperty(self.SetPoseMinimumBodyThreshold, self.GetPoseMinimumBodyThreshold),
      "pose_minimum_face_height": ConfigProperty(self.SetPoseMinimumFaceHeight, self.GetPoseMinimumFaceHeight),
      "pose_minimum_body_height": ConfigProperty(self.SetPoseMinimumBodyHeight, self.GetPoseMinimumBodyHeight),
      "face_position_left_right_threshold": ConfigProperty(self.SetFacePositionLeftRightThreshold, self.GetFacePositionLeftRightThreshold),
      "face_position_straight_threshold": ConfigProperty(self.SetFacePositionStraightThreshold, self.GetFacePositionStraightThreshold),
      "face_rectangle_yfactor": ConfigProperty(self.SetFaceRectangleYFactor, self.GetFaceRectangleYFactor),
      "live_stream_port": ConfigProperty(self.SetLiveStreamPort, self.GetLiveStreamPort),
      "embeddings_filename": ConfigProperty(self.SetEmbeddingsFilename, self.GetEmbeddingsFilename),
    }

    self.Set("flip_video_frame", flip_video_frame)
    self.Set("person_tracking_creation_m", person_tracking_creation_m)
    self.Set("person_tracking_creation_n", person_tracking_creation_n)
    self.Set("object_tracking_info_history_count", object_tracking_info_history_count)
    self.Set("object_tracking_removal_count", object_tracking_removal_count)
    self.Set("object_tracking_centroid_weight", object_tracking_centroid_weight)
    self.Set("object_tracking_color_weight", object_tracking_color_weight)
    self.Set("object_tracking_vector_weight", object_tracking_vector_weight)
    self.Set("object_tracking_size_weight", object_tracking_size_weight)
    self.Set("object_tracking_color_sample_pixels", object_tracking_color_sample_pixels)
    self.Set("live_stream_port", live_stream_port)
    self.Set("pose_minimum_face_threshold", pose_minimum_face_threshold)
    self.Set("pose_minimum_body_threshold", pose_minimum_body_threshold)
    self.Set("pose_minimum_face_height", pose_minimum_face_height)
    self.Set("pose_minimum_body_height", pose_minimum_body_height)
    self.Set("face_position_left_right_threshold", face_position_left_right_threshold)
    self.Set("face_position_straight_threshold", face_position_straight_threshold)
    self.Set("face_rectangle_yfactor", face_rectangle_yfactor)
    self.Set("embeddings_filename", embeddings_filename)


  def SetLiveStreamPort(self, value):
    self.__live_stream_port = value


  def SetPoseMinimumFaceThreshold(self, value):
    self.__pose_minimum_face_threshold = value


  def GetPoseMinimumFaceThreshold(self):
    return self.__pose_minimum_face_threshold


  def SetPoseMinimumBodyThreshold(self, value):
    self.__pose_minimum_body_threshold = value


  def GetPoseMinimumBodyThreshold(self):
    return self.__pose_minimum_body_threshold


  def SetPoseMinimumFaceHeight(self, value):
    self.__pose_minimum_face_height = value


  def GetPoseMinimumFaceHeight(self):
    return self.__pose_minimum_face_height


  def SetPoseMinimumBodyHeight(self, value):
    self.__pose_minimum_body_height = value


  def GetPoseMinimumBodyHeight(self):
    return self.__pose_minimum_body_height


  def SetFacePositionLeftRightThreshold(self, value):
    self.__face_position_left_right_threshold = value


  def GetFacePositionLeftRightThreshold(self):
    return self.__face_position_left_right_threshold


  def SetFacePositionStraightThreshold(self, value):
    self.__face_position_straight_threshold = value


  def GetFacePositionStraightThreshold(self):
    return self.__face_position_straight_threshold


  def SetFaceRectangleYFactor(self, value):
    self.__face_rectangle_yfactor = value


  def GetFaceRectangleYFactor(self):
    return self.__face_rectangle_yfactor


  def SetPersonTrackingCreationM(self, value):
    self.__person_tracking_creation_m = value


  def SetPersonTrackingCreationN(self, value):
    self.__person_tracking_creation_n = value


  def SetObjectTrackingInfoHistoryCount(self, value):
    self.__object_tracking_info_history_count = value


  def SetObjectTrackingRemovalCount(self, value):
    self.__object_tracking_removal_count = value


  def SetObjectTrackingCentroidWeight(self, value):
    self.__object_tracking_centroid_weight = value


  def SetObjectTrackingColorWeight(self, value):
    self.__object_tracking_color_weight = value


  def SetObjectTrackingVectorWeight(self, value):
    self.__object_tracking_vector_weight = value


  def SetObjectTrackingSizeWeight(self, value):
    self.__object_tracking_size_weight = value


  def SetObjectTrackingColorSamplePixels(self, value):
    self.__object_tracking_color_sample_pixels = value


  def SetFlipVideoFrame(self, value):
    self.__flip_video_frame = value


  def SetEmbeddingsFilename(self, value):
    self.__embeddings_filename = value


  def GetPersonTrackingCreationM(self):
    return self.__person_tracking_creation_m


  def GetPersonTrackingCreationN(self):
    return self.__person_tracking_creation_n


  def GetObjectTrackingInfoHistoryCount(self):
    return self.__object_tracking_info_history_count


  def GetObjectTrackingRemovalCount(self):
    return self.__object_tracking_removal_count


  def GetObjectTrackingCentroidWeight(self):
    return self.__object_tracking_centroid_weight


  def GetObjectTrackingColorWeight(self):
    return self.__object_tracking_color_weight


  def GetObjectTrackingVectorWeight(self):
    return self.__object_tracking_vector_weight


  def GetObjectTrackingSizeWeight(self):
    return self.__object_tracking_size_weight


  def GetObjectTrackingColorSamplePixels(self):
    return self.__object_tracking_color_sample_pixels


  def GetFlipVideoFrame(self):
    return self.__flip_video_frame


  def GetLiveStreamPort(self):
    return self.__live_stream_port


  def GetEmbeddingsFilename(self):
    return self.__embeddings_filename

  
  def Set(self, property, value):
    if not property in self.__properties:
      raise Exception("Invalid config property")

    self.__properties[property].Set(value)

  
  def Get(self, property):
    if not property in self.__properties:
      raise Exception("Invalid config property")

    self.__properties[property].Get()
