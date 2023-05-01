
from typing import Union


class CustomClass:
    def __init__(self, identifier: Union[str, int], name: str):
        self.identifier = identifier
        self.name = name


x = CustomClass(identifier=3.0, name='test')
print(x.identifiers)