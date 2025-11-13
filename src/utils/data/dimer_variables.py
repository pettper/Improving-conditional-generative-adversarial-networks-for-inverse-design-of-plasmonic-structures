from enum import Enum


# Enum class for the available variables in DimerDataset
class DimerVariable(Enum):
    SCATTERING_CROSS_SECTION = 1
    ABSORPTION_CROSS_SECTION = 2
    ROTATION = 3
    ELLIPTICITY = 4
    ALL = [1, 2, 3, 4]
    CROSS_SECTIONS = [1, 2]

    def __eq__(self, other):
        if isinstance(other, DimerVariable):
            return self.value == other.value
        else:
            return False

    @staticmethod
    def keymap(key):
        if key == "sca":
            return DimerVariable.SCATTERING_CROSS_SECTION
        elif key == "abs":
            return DimerVariable.ABSORPTION_CROSS_SECTION
        elif key == "rot":
            return DimerVariable.ROTATION
        elif key == "elip":
            return DimerVariable.ELLIPTICITY
        elif key == "all":
            return DimerVariable.ALL
        elif key == "cross":
            return DimerVariable.CROSS_SECTIONS
        else:
            raise Exception("Unknown key, valid keys are 'sca', 'abs', 'rot', 'elip', 'all'.")

    def size(self):
        if (self == DimerVariable.SCATTERING_CROSS_SECTION or self == DimerVariable.ABSORPTION_CROSS_SECTION
                or self == DimerVariable.ROTATION or self == DimerVariable.ELLIPTICITY):
            return 1
        elif self == DimerVariable.CROSS_SECTIONS:
            return 2
        else:
            return 4

    