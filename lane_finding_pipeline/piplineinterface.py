import abc
class PipeLineInterface(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self, image):
        pass