from abc import ABCMeta, abstractmethod;

class AbstractImage:
    __metaclass__ = ABCMeta

    @abstractmethod
    def say_something(self): pass

class Cat(Animal):
    def say_something(self):
        return "Miauuu!"