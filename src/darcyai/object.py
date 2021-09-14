import enum


class ObjectState(enum.IntEnum):
    NORMAL = 1
    NEW    = 2
    POI    = 4
    GONE   = 8


class DetectedObject:
  def __init__(self):
    self.object_id = 0
    self.uuid = None
    self.state = ObjectState.NORMAL


  def IsPOI(self):
    return self.state & ObjectState.POI == ObjectState.POI


  def IsNew(self):
    return self.state & ObjectState.NEW == ObjectState.NEW


  def IsGone(self):
    return self.state & ObjectState.GONE == ObjectState.GONE