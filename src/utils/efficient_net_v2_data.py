from enum import Enum


# Enum class for the data provided to build an efficient net v2 model.
class EfficientNetV2Data(Enum):
    CHANNELS = (24, 24, 48, 64, 128, 160, 256, 1280)
    LAYERS = (1, 2, 4, 4, 6, 9, 15, 1)
    STRIDE = (2, 1, 2, 2, 2, 1, 2)
    EXPANSION = (1, 4, 4, 6, 6, 6)
