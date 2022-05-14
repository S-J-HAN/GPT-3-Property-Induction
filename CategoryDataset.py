from typing import Dict, List
from abc import ABC, abstractmethod
from collections import defaultdict as dd
from dataclasses import dataclass

import pandas as pd
import numpy as np

import os

import config


@dataclass
class MatrixTranslationMixup:
    """Some of the De Deyne csv's have incorrectly labelled Dutch and English columns and rows that need to be manually coded for correction."""
    
    columns_are_switched: bool
    rows_are_switched: bool


class CategoryDataset(ABC):
    
    @abstractmethod
    def generate_animal_feature_map(self) -> Dict[str, np.ndarray]:
        """Create and return a dictionary that maps category names to feature vectors"""
        pass

    @abstractmethod
    def class_list(self) -> List[str]:
        """Return a list of all classes in this dataset"""
        pass

    @abstractmethod
    def class_category_list(self, query_class: str) -> List[str]:
        """Return a list of all categories that fall under a given class"""
        pass

    @abstractmethod
    def calculate_category_similarity(self, category_a: str, category_b: str) -> float:
        """Calculate similarity between two categories"""
        pass

    def category_list(self) -> List[str]:
        """Return a list of all categories in this dataset"""
        return [a for b in [self.class_category_list(c) for c in self.class_list()] for a in b]

    
class KaggleCategoryDataset(CategoryDataset):
    """
    Uses the Zoo animal dataset from Kaggle, which includes both animal features and animal classes
    
    This data is NOT available in this repo.
    """
    
    def __init__(self, animal_features_path: str = "data/animals/zoo.csv", animal_types_path: str = "data/animals/class.csv"):

        self.animal_features = pd.read_csv(animal_features_path)
        
        self.animal_types = pd.read_csv(animal_types_path)
        self.animal_types["Animal_Names"] = self.animal_types["Animal_Names"].apply(lambda x: eval("['{}']".format(x.replace(", ", "', '"))))
        self.classname2classlist = pd.Series(self.animal_types["Animal_Names"].tolist(), index=self.animal_types["Class_Type"].tolist()).to_dict()

        animal2class = {}
        for class_name, class_list in self.classname2classlist.items():
            for animal_name in class_list:
                animal2class[animal_name] = class_name
        self.animal2class = animal2class

        legs = pd.get_dummies(self.animal_features["legs"])
        legs = legs.rename({c: f"{c}_legs" for c in legs.columns}, axis=1)
        self.animal_features = pd.concat([self.animal_features, legs], axis=1).drop("legs", axis=1)

        self.feature_map = self.generate_animal_feature_map()

        class_feature_centroids = {}
        for category_class, category_list in self.classname2classlist.items():
            features = [self.feature_map[category] for category in category_list]
            class_feature_centroids[category_class] = np.mean(features, axis=0)
        self.class_feature_centroids = class_feature_centroids

    def generate_animal_feature_map(self) -> Dict[str, np.ndarray]:
        feature_map: Dict[str, np.ndarray] = {}
        for _, row in self.animal_features.iterrows():
            feature_map[row["animal_name"]] = np.array(row[1:-2])
        return feature_map

    def class_list(self) -> List[str]:
        return list(self.classname2classlist.keys())

    def class_category_list(self, query_class: str) -> List[str]:
        return self.classname2classlist[query_class]

    def calculate_category_similarity(self, category_a: str, category_b: str) -> float:
        category_a_feature = self.feature_map[category_a]
        category_b_feature = self.feature_map[category_b]
        return np.linalg.norm(category_a_feature - category_b_feature)
    
    def calculate_category_typicality(self, category: str) -> float:
        category_feature = self.feature_map[category]
        category_class = [k for k,v in self.classname2classlist.items() if category in v][0]
        category_class_feature = self.class_feature_centroids[category_class]
        return np.linalg.norm(category_feature - category_class_feature)


class KempLeuvenCategoryDataset(CategoryDataset):
    """
    Uses the Kemp/Leuven dataset
    
    This is NOT available in this repo.
    """
    
    def __init__(self, animal_features_path: str = "data/kempTypeIIAnimalExemplarFeatureMatrix.csv"):

        self.animal_features = pd.read_csv(animal_features_path).iloc[1:,1:].rename({"feature/ exemplar ENGLISH": "category", "Unnamed: 2": "total"}, axis=1)
        for category in self.animal_features.columns[2:]:
            self.animal_features[category] = self.animal_features[category].apply(int).apply(lambda x: 1 if x > 2 else 0)

        # Hardcoded, ordered classes for each column
        self.animal_types = [('mammal', 30), ('bird', 30), ('fish', 23), ('insect', 26), ('reptile', 20)]

        i = 0
        c = 0
        self.classname2classlist = dd(list)
        for category in self.animal_features.columns[2:]:
            self.classname2classlist[self.animal_types[i][0]].append(category)

            c += 1
            if c == self.animal_types[i][1]:
                c = 0
                i += 1

        animal2class = {}
        for class_name, class_list in self.classname2classlist.items():
            for animal_name in class_list:
                animal2class[animal_name] = class_name
        self.animal2class = animal2class

        self.feature_map = self.generate_animal_feature_map()

        class_feature_centroids = {}
        for category_class, category_list in self.classname2classlist.items():
            features = [self.feature_map[category] for category in category_list]
            class_feature_centroids[category_class] = np.mean(features, axis=0)
        self.class_feature_centroids = class_feature_centroids

    def generate_animal_feature_map(self) -> Dict[str, np.ndarray]:
        feature_map: Dict[str, np.ndarray] = {}
        for category in self.animal_features.columns[2:]:
            feature_map[category] = np.array(self.animal_features[category])
        return feature_map

    def class_list(self) -> List[str]:
        return list(self.classname2classlist.keys())

    def class_category_list(self, query_class: str) -> List[str]:
        return self.classname2classlist[query_class]

    def calculate_category_similarity(self, category_a: str, category_b: str) -> float:
        category_a_feature = self.feature_map[category_a]
        category_b_feature = self.feature_map[category_b]
        return np.linalg.norm(category_a_feature - category_b_feature)
    
    def calculate_category_typicality(self, category: str) -> float:
        category_feature = self.feature_map[category]
        category_class = [k for k,v in self.classname2classlist.items() if category in v][0]
        category_class_feature = self.class_feature_centroids[category_class]
        return np.linalg.norm(category_feature - category_class_feature)


class DeDeyneCategoryDataset(CategoryDataset):
    """Gets category similarities from data collected by De Deyne et al. (2008)"""

    def __init__(self, category_class: str) -> None:
        
        self.mixups = {
            "Birds": MatrixTranslationMixup(False, True),
            "Clothing": MatrixTranslationMixup(False, True),
            "Fish": MatrixTranslationMixup(False, True),
            "Fruit": MatrixTranslationMixup(True,False),
            "Insects": MatrixTranslationMixup(False, False),
            "KitchenUtensils": MatrixTranslationMixup(False, False),
            "Mammals": MatrixTranslationMixup(False, False),
            "MusicalInstruments": MatrixTranslationMixup(False, True),
            "Professions": MatrixTranslationMixup(True, False),
            "Reptiles": MatrixTranslationMixup(False, False),
            "Sports": MatrixTranslationMixup(True, False),
            "Vegetables": MatrixTranslationMixup(True, False),
            "Vehicles": MatrixTranslationMixup(False, True),
            "Weapons": MatrixTranslationMixup(False, False),
        }

        self.category_class = category_class
        self.df = self._generate_similarity_df(category_class=category_class, folder=config.dedeyne_similarities_path(category_class))
        self.category2index = {a:i for i,a in enumerate(self.df.columns[1:])}

    def _generate_similarity_df(self, category_class: str, folder: str) -> pd.DataFrame:

        similarity_dfs = []
        for i in range(len(os.listdir(folder))):
            df = pd.read_csv(f"{folder}/pairwiseSimilarities{category_class}-{i+1}.csv", encoding = "ISO-8859-1")
            if self.mixups[category_class].columns_are_switched:
                df.columns = df.columns.tolist()[:2] + df.iloc[0,2:].tolist()
            if self.mixups[category_class].rows_are_switched:
                df["exemplar ENGLISH"] = df.iloc[:,0].tolist()
            similarity_dfs.append(df.iloc[1:,1:])
        average_similarities = np.mean([df.iloc[:,1:].values.astype(np.int) for df in similarity_dfs], axis=0)
        average_similarities = average_similarities + np.rot90(np.fliplr(average_similarities))
        average_similarity_df = pd.DataFrame(average_similarities, columns=similarity_dfs[0].columns[1:])
        average_similarity_df["category"] = similarity_dfs[0]["exemplar ENGLISH"].tolist()
        average_similarity_df = average_similarity_df[["category"] + similarity_dfs[0].columns[1:].tolist()]
        
        return average_similarity_df

    def generate_animal_feature_map(self) -> Dict[str, np.ndarray]:
        """Create and return a dictionary that maps category names to feature vectors"""
        return {c: self.df[c] for c in self.df.columns}

    def class_list(self) -> List[str]:
        """Return a list of all classes in this dataset"""
        return [self.category_class]

    def class_category_list(self) -> List[str]:
        """Return a list of all categories that fall under a given class"""
        return list(self.df.columns[1:])

    def calculate_category_similarity(self, category_a: str, category_b: str) -> float:
        return self.df[category_a].iloc[self.category2index[category_b]]

    @staticmethod
    def possible_class_list() -> List[str]:
        return ["Birds", "Clothing", "Fish", "Fruit", "Insects", "KitchenUtensils", "Mammals", "MusicalInstruments", "Professions", "Reptiles", "Sports", "Vegetables", "Vehicles", "Weapons"]
