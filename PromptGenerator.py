from abc import ABC, abstractmethod
from typing import List

import pandas as pd

import PropertyGenerator
import schema


class PromptGenerator(ABC):

    def __init__(
        self, 
        property_generator: PropertyGenerator.PropertyGenerator,
        arguments_loc: str,
        read_arguments_df: bool = True,
    ):
        self.property_generator = property_generator

        if read_arguments_df:
            self.arguments_df = pd.read_csv(arguments_loc, index_col=0).fillna("")

    def get_property(self) -> str:
        return self.property_generator.get_property()

    @abstractmethod
    def generate_prompt(self, premise1: str, premise2: str, premise3: str, conclusion: str, property: str) -> str:
        """Generates a single prompt to be sent to the OpenAI API."""
        pass

    def generate_prompts(self) -> List[schema.Prompt]:
        """Generates a list of prompts to be sent to the OpenAI API."""
        prompts = []
        for _, row in self.arguments_df.iterrows():
            property = self.get_property()
            prompt_string_1 = self.generate_prompt(row["arg1_premise1"], row["arg1_premise2"], row["arg1_premise3"], row["arg1_conclusion"], property)
            prompt_string_2 = self.generate_prompt(row["arg2_premise1"], row["arg2_premise2"], row["arg2_premise3"], row["arg2_conclusion"], property)
            
            prompt_1 = schema.OshersonPrompt(
                phenomenon_number=row["phenomenon_number"],
                phenomenon_name=row["phenomenon_name"],
                phenomenon_type=row["phenomenon_type"],
                prompt=prompt_string_1,
                property=property,
                premise_category_1=row["arg1_premise1"],
                premise_category_2=row["arg1_premise2"],
                premise_category_3=row["arg1_premise3"],
                conclusion_category=row["arg1_conclusion"],
            )

            prompt_2 = schema.OshersonPrompt(
                phenomenon_number=row["phenomenon_number"],
                phenomenon_name=row["phenomenon_name"],
                phenomenon_type=row["phenomenon_type"],
                prompt=prompt_string_2,
                property=property,
                premise_category_1=row["arg2_premise1"],
                premise_category_2=row["arg2_premise2"],
                premise_category_3=row["arg2_premise3"],
                conclusion_category=row["arg2_conclusion"],
            )

            prompts.append(prompt_1)
            prompts.append(prompt_2)

        return prompts

    @staticmethod
    def convert_category_to_plural(category: str) -> str:
        """Converts a category to its plural form"""

        SPECIAL_CASES = {
            "deer": "deer",
            "fox": "foxes",
            "mouse": "mice",
            "sheep": "sheep",
            "wolf": "wolves",
            "buffalo": "buffaloes",
            "calf": "calves",
            "cavy": "cavies",
            "deer": "deer",
            "lynx": "lynxes",
            "oryx": "oryxes",
            "platypus": "platypuses",
            "pony": "ponies",
            "reindeer": "reindeer",
            "wallaby": "wallabies",
            "ostrich": "ostriches",
            "bass": "basses",
            "carp": "carp",
            "catfish": "catfish",
            "dogfish": "dogfish",
            "tuna": "tuna",
            "housefly": "houseflies",
            "horsefly": "horseflies",
            "crayfish": "crayfish",
            "octopus": "octopuses",
            "starfish": "starfish",
            "bison": "bison",
            "dromedary": "dromedaries",
            "rhinoceros": "rhinoceroses",
            "hippopotamus": "hippopotamuses",
            "strawberry": "strawberries",
            "blueberry": "blueberries",
            "blackberry": "blackberries",
            "mandarine": "mandarins",
            "dates": "dates",
            "raspberry": "raspberries",
            "cherry": "cherries",
            "mango": "mangoes",
            "peach": "peaches",
            "cabary": "cabaries",
            "anchovy": "anchovies",
            "goldfish": "goldfish",
            "cod": "cod",
            "carp": "carp",
            "flatfish": "flatfish",
            "swordfish": "swordfish",
            "salmon": "salmon",
            "leech": "leeches",
            "cockroach": "cockroaches",
            "dragonfly": "dragonflies",
            "fly": "flies",
            "butterfly": "butterflies",
            "wood louse": "woodlice",
            "fish": "all fish", 
            "mammal": "all mammals",
            "bird": "all birds",
            "insect": "all insects",
            "reptile": "all reptiles",
            "animal": "all animals",
            "trout": "trout",
            "praying mantis": "praying mantises",
            "canary": "canaries",
        }

        if category in SPECIAL_CASES:
            return SPECIAL_CASES[category]
        else:
            return f"{category}s"

    @staticmethod
    def convert_class_to_plural(query_class: str) -> str:
        """'Animals' becomes 'All animals', etc."""

        SPECIAL_CASES = {
            "fish": "All fish",
        }

        query_class_lower = query_class.lower()
        if query_class_lower in SPECIAL_CASES:
            return SPECIAL_CASES[query_class_lower]
        else:
            return f"All {query_class_lower}s"


class QuestionPromptGenerator(PromptGenerator):

    def generate_prompt(self, premise1: str, premise2: str, premise3: str, conclusion: str, property: str) -> schema.Prompt:

        if premise2 and not premise3:
            return f"Q: If {self.convert_category_to_plural(premise1)} and {self.convert_category_to_plural(premise2)} {property}, do {self.convert_category_to_plural(conclusion)} {property}? A: Yes"
        elif premise2 and premise3:
            return f"Q: If {self.convert_category_to_plural(premise1)}, {self.convert_category_to_plural(premise2)} and {self.convert_category_to_plural(premise3)} {property}, do {self.convert_category_to_plural(conclusion)} {property}? A: Yes"
        else:
            # not premise2 and not premise3
            return f"Q: If {self.convert_category_to_plural(premise1)} {property}, do {self.convert_category_to_plural(conclusion)} {property}? A: Yes"


class StatementPromptGenerator(PromptGenerator):

    def generate_prompt(self, premise1: str, premise2: str, premise3: str, conclusion: str, property: str) -> str:

        if premise2 and not premise3:
            return f"{self.convert_category_to_plural(premise1).capitalize()} and {self.convert_category_to_plural(premise2)} {property}. {self.convert_category_to_plural(conclusion).capitalize()} {property}."
        elif premise2 and premise3:
            return f"{self.convert_category_to_plural(premise1).capitalize()}, {self.convert_category_to_plural(premise2)} and {self.convert_category_to_plural(premise3)} {property}. {self.convert_category_to_plural(conclusion).capitalize()} {property}."
        else:
            # not premise2 and not premise3
            return f"{self.convert_category_to_plural(premise1).capitalize()} {property}. {self.convert_category_to_plural(conclusion).capitalize()} {property}."

class ListPromptGenerator(PromptGenerator):

    def generate_prompt(self, premise1: str, premise2: str, premise3: str, conclusion: str, property: str) -> str:

        if premise2 and not premise3:
            return f"{self.convert_category_to_plural(premise1).capitalize()}, {self.convert_category_to_plural(premise2)}, {self.convert_category_to_plural(conclusion)}"
        elif premise2 and premise3:
            return f"{self.convert_category_to_plural(premise1).capitalize()}, {self.convert_category_to_plural(premise2)}, {self.convert_category_to_plural(premise3)}, {self.convert_category_to_plural(conclusion)}"
        else:
            # not premise2 and not premise3
            return f"{self.convert_category_to_plural(premise1).capitalize()}, {self.convert_category_to_plural(conclusion)}"

class SimpleInstructPromptGenerator(PromptGenerator):

    def generate_prompt(self, premise1: str, premise2: str, premise3: str, conclusion: str, property: str) -> schema.Prompt:

        if premise2 and not premise3:
            return f"{self.convert_category_to_plural(premise1).capitalize()} and {self.convert_category_to_plural(premise2)} {property}.\nQ: Do {self.convert_category_to_plural(conclusion)} {property}?\nA: Yes"
        elif premise2 and premise3:
            return f"{self.convert_category_to_plural(premise1).capitalize()}, {self.convert_category_to_plural(premise2)} and {self.convert_category_to_plural(premise3)} {property}.\nQ: Do {self.convert_category_to_plural(conclusion)} {property}?\nA: Yes"
        else:
            # not premise2 and not premise3
            return f"{self.convert_category_to_plural(premise1).capitalize()} {property}.\nQ: Do {self.convert_category_to_plural(conclusion)} {property}?\nA: Yes"

class InstructPromptGenerator(PromptGenerator):

    def generate_prompt(self, premise1: str, premise2: str, premise3: str, conclusion: str, property: str) -> schema.Prompt:
        
        instruction = f"You are an expert on the properties that animals have, and you understand how animals share properties in common.\nRecently some animals have been discovered to {property}.\n"
        if premise2 and not premise3:
            return f"{instruction}We know that {self.convert_category_to_plural(premise1).lower()} and {self.convert_category_to_plural(premise2)} {property}. Does this mean that {self.convert_category_to_plural(conclusion)} {property}? Please answer Yes or No.\nA: Yes"
        elif premise2 and premise3:
            return f"{instruction}We know that {self.convert_category_to_plural(premise1).lower()}, {self.convert_category_to_plural(premise2)} and {self.convert_category_to_plural(premise3)} {property}. Does this mean that {self.convert_category_to_plural(conclusion)} {property}? Please answer Yes or No.\nA: Yes"
        else:
            # not premise2 and not premise3
            return f"{instruction}We know that {self.convert_category_to_plural(premise1).lower()} {property}. Does this mean that {self.convert_category_to_plural(conclusion)} {property}? Please answer Yes or No.\nA: Yes"