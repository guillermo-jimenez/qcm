from abc import ABCMeta, abstractmethod;

class AbstractImage(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @property
    def imageType(self):
        return self.__imageType;

    @imageType.setter
    def imageType(self, imageType):
        self.__imageType = imageType;

    # @abstractmethod
    # def say_something(self): pass

# class Cat(Animal):
#     def say_something(self):
#         return "Miauuu!"