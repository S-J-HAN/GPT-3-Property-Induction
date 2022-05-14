from abc import ABC, abstractmethod
from gibberish import Gibberish

import random

import pandas as pd


class PropertyGenerator(ABC):
    """Generate property strings"""

    @abstractmethod
    def get_property(self) -> str:
        pass


class OshersonPropertyGenerator(PropertyGenerator):

    def __init__(self):
        osherson_df = pd.read_csv("data/osherson_phenomena.csv")
        self.properties = list(osherson_df["property"].unique())

    def get_property(self) -> str:
        return random.sample(self.properties, k=1)[0].replace(".", "")


class GibberishPropertyGenerator(PropertyGenerator):

    def __init__(self):
        self.gibberish = Gibberish()

    def get_property(self) -> str:
        return "xufwop"#self.gibberish.generate_word()


class NamelessPropertyGenerator(PropertyGenerator):

    def get_property(self) -> str:
        return "have property P"