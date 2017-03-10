
import jpype
from jpype import *

class NormalizeFilipino:

    def normalize_Filipino(self, text):
        jvmPath = jpype.getDefaultJVMPath()
        jpype.startJVM(jvmPath, "-Djava.class.path=C:/Users/sharkscion/Documents/DLSU Computer Science/Development Files/JAR Files/NormAPI.jar")
        normapi = JPackage("normapi").NormAPI
        output = normapi.normalize_Text(text)
        jpype.shutdownJVM()

        return output