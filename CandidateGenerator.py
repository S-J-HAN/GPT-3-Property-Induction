from CategoryDataset import *

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, astuple, fields
from typing import List

import random
import itertools

random.seed(1111)


@dataclass
class CandidateArgumentPair:

    phenomenon_number: int
    phenomenon_name: str
    phenomenon_type: str

    arg1_premise1: str
    arg1_premise2: str
    arg1_premise3: str
    arg1_conclusion: str

    arg2_premise1: str
    arg2_premise2: str
    arg2_premise3: str
    arg2_conclusion: str

    
class CandidateGenerator(ABC):
    """
    For a given set of phenomena, generates a list of randomly sampled argument pairs where the strength disparity between the two arguments is maximally large.
    """

    def __init__(self, category_dataset: CategoryDataset) -> None:
        
        self.category_dataset = category_dataset

    @abstractmethod
    def generate_candidate_argument_pairs(self, n: int) -> List[CandidateArgumentPair]:
        """
        For each phenomena under question, randomly samples n argument pairs and calculates each argument's 'ground truth' strength across each phenomena.
        """
        pass


    @abstractmethod
    def rank_candidate_argument_pairs(self, candidate_argument_pairs: List[CandidateArgumentPair]) -> pd.DataFrame:
        """
        Given a list of candidate argument pairs, ranks candidates according to their phenomena strengths and returns as a dataframe
        """
        pass

    def generate_candidate_argument_pair_dataset(self, file_loc: str, m: int) -> None:
        """
        Generates a csv dataset of m candidate argument pairs per phenomena that can be used for querying GPT-3.
        """

        # generate initial set of candidates
        candidate_argument_pairs = self.generate_candidate_argument_pairs(m*20)

        # rerank candidates
        ranked_candidate_argument_pair_df = self.rank_candidate_argument_pairs(candidate_argument_pairs)

        # remove duplicates
        ranked_candidate_argument_pair_df = ranked_candidate_argument_pair_df.drop_duplicates()

        # keep only top m pairs per phenomena
        ranked_candidate_argument_pair_df = ranked_candidate_argument_pair_df.groupby("phenomenon_number").head(m).reset_index(drop=True)

        ranked_candidate_argument_pair_df.to_csv(file_loc)

class SyntheticOshersonSCMCandidateGenerator(CandidateGenerator):

    def __init__(self, category_datasets: Dict[str, DeDeyneCategoryDataset], osherson_phenomena: str = "data/osherson_phenomena.csv") -> None:
        
        self.category_datasets = category_datasets
        self.phenomena_df = pd.read_csv(osherson_phenomena)[["phenomenon_number", "phenomenon_name", "phenomenon_type"]].drop_duplicates()

        self.category_class_rename = {
            "Reptiles": "reptile", 
            "Mammals": "mammal",
            "Fish": "fish",
            "Insects": "insect",
            "Birds": "bird",
        }

    def _sim(self, premise_1: str, premise_2: str, premise_3: str, conclusion: str, category_class:str) -> float:

        category_dataset = self.category_datasets[category_class]
        
        if conclusion == category_class:
            conclusion_categories = category_dataset.category_list()
        else:
            conclusion_categories = [conclusion]
            
        a = np.mean([
            np.max([
                category_dataset.calculate_category_similarity(p, c)
                for p in [premise_1, premise_2, premise_3] if p
            ])
            for c in conclusion_categories
        ])

        return a
        
    def _scm(self, premise_1: str, premise_2: str, premise_3: str, conclusion: str, category_class:str, alpha: float = 0.5) -> float:
    
        category_dataset = self.category_datasets[category_class]
        
        a = self._sim(premise_1, premise_2, premise_3, conclusion, category_class)
            
        # calculate b
        conclusion_categories = category_dataset.category_list()
        b = np.mean([
                np.max([
                    category_dataset.calculate_category_similarity(p, c)
                    for p in [premise_1, premise_2, premise_3] if p
                ])
                for c in conclusion_categories
        ]) 
        
        return alpha*a + (1-alpha)*b

    def generate_candidate_argument_pairs(self, n: int) -> List[CandidateArgumentPair]:

        candidate_argument_pairs = []
        for _, row in self.phenomena_df.iterrows():

            phenomenon_number = row["phenomenon_number"]

            for _ in range(n):

                c1, c2 = random.sample(list(self.category_datasets.items()), k=2)
                category_class, category_dataset = c1
                _, category_dataset_2 = c2

                if phenomenon_number == 1:

                    conclusion_category = category_class
                    premise_category_1, premise_category_2 = random.sample(category_dataset.category_list(), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=None,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_2,
                        arg2_premise2=None,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 2:

                    conclusion_category = category_class
                    # premise_category_1, premise_category_2 = random.sample(category_dataset.category_list(), k=2)

                    # # For second argument's second premise, find the premise category with the closest SCM score to the first argument's second premise
                    # arg1_p2_scm = self._sim(premise_category_2, "", "", conclusion_category, category_class)
                    # arg2_candidates = [i for i in category_dataset.category_list() if i not in (premise_category_1, premise_category_2)]
                    # premise_category_3 = arg2_candidates[0]
                    # closest_scm = np.inf
                    # for candidate in arg2_candidates:
                    #     candidate_scm = self._sim(candidate, "", "", conclusion_category, category_class)
                    #     scm_diff = abs(candidate_scm - arg1_p2_scm)
                    #     if scm_diff < closest_scm:
                    #         closest_scm = candidate_scm
                    #         premise_category_3 = candidate

                    # premise_category_1 = random.choice(category_dataset.category_list())
                    # similarities = sorted([(c, category_dataset.calculate_category_similarity(premise_category_1, c)) for c in category_dataset.category_list() if c not in (premise_category_1, conclusion_category)], reverse=False, key=lambda x: x[1])
                    # premise_category_2 = similarities[0][0]
                    # premise_category_3 = similarities[-2][0]

                    candidates = []

                    while len(candidates) < 2:
                        premise_category_1 = random.choice(category_dataset.category_list())
                        arg1_p1_scm = self._scm(premise_category_1, "", "", conclusion_category, category_class)
                        candidates = [i for i in category_dataset.category_list() if i not in (premise_category_1, conclusion_category) and self._scm(i, "", "", conclusion_category, category_class) < arg1_p1_scm]
                        candidates = sorted([(c, category_dataset.calculate_category_similarity(premise_category_1, c)) for c in candidates], reverse=True, key=lambda x: x[1])
                    premise_category_3 = candidates[0][0]
                    premise_category_2 = random.choice(candidates[1:])[0]
                        
                        #similarities = sorted([(c, category_dataset.calculate_category_similarity(premise_category_1, c)) for c in arg_candidates if c not in (premise_category_1, conclusion_category)], reverse=True, key=lambda x: x[1])
                        #premise_category_3 = similarities[1][0]

                        # For first argument's second premise, find the premise category with the closest SCM score to the first argument's first premise
                        
                        
                    #     for candidate in arg2_candidates:
                    #         candidate_scm = self._scm(candidate, "", "", conclusion_category, category_class)
                    #         if candidate_scm < arg1_p1_scm:
                    #             candidates.append((candidate, candidate_scm))
                    # candidates = sorted(candidates, reverse=True, key=lambda x: x[1])
                    # premise_category_2 = random.choice(candidates)[0]

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_3,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 3:

                    conclusion_category = category_class
                    premise_category_1, premise_category_2 = random.sample(category_dataset.category_list(), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion="animal",
                    )

                elif phenomenon_number == 4:

                    conclusion_category = category_class
                    #premise_category_1, premise_category_2, premise_category_3 = random.sample(category_dataset.category_list(), k=3)

                    candidates = []
                    while len(candidates) < 1:
                        premise_category_1, premise_category_2 = random.sample(category_dataset.category_list(), k=2)
                        max_scm = np.max([self._scm(p, "", "", conclusion_category, category_class) for p in (premise_category_1, premise_category_2)])
                        candidates = [c for c in category_dataset.category_list() if c not in (premise_category_1, premise_category_2, conclusion_category) and self._scm(c, "", "", conclusion_category, category_class) < max_scm]
                    premise_category_3 = random.choice(candidates)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=premise_category_3,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 5:

                    premise_category_1, premise_category_2, conclusion_category_1, conclusion_category_2 = random.sample(category_dataset.category_list(), k=4)
                    
                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category_1,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category_2,
                    )

                elif phenomenon_number == 6:

                    # premise_category_1, premise_category_2, conclusion_category = random.sample(category_dataset.category_list(), k=3)

                    # # For second argument's second premise, find the premise category with the closest SCM score to the first argument's second premise
                    # arg1_p2_scm = self._sim(premise_category_2, "", "", conclusion_category, category_class)
                    # arg2_candidates = [i for i in category_dataset.category_list() if i not in (premise_category_1, premise_category_2, conclusion_category)]
                    # premise_category_3 = arg2_candidates[0]
                    # closest_scm = np.inf
                    # for candidate in arg2_candidates:
                    #     candidate_scm = self._sim(candidate, "", "", conclusion_category, category_class)
                    #     scm_diff = abs(candidate_scm - arg1_p2_scm)
                    #     if scm_diff < closest_scm:
                    #         closest_scm = candidate_scm
                    #         premise_category_3 = candidate




                    # premise_category_1, conclusion_category = random.sample(category_dataset.category_list(), k=2)
                    # similarities = sorted([(c, category_dataset.calculate_category_similarity(premise_category_1, c)) for c in category_dataset.category_list() if c not in (premise_category_1, conclusion_category)], reverse=False, key=lambda x: x[1])
                    # premise_category_2 = similarities[0][0]
                    # premise_category_3 = similarities[-2][0]


                    candidates = []

                    while len(candidates) < 2:
                        premise_category_1, conclusion_category = random.sample(category_dataset.category_list(), k=2)
                        arg1_p1_scm = self._scm(premise_category_1, "", "", conclusion_category, category_class)
                        candidates = [i for i in category_dataset.category_list() if i not in (premise_category_1, conclusion_category) and self._scm(i, "", "", conclusion_category, category_class) < arg1_p1_scm]
                        candidates = sorted([(c, category_dataset.calculate_category_similarity(premise_category_1, c)) for c in candidates], reverse=True, key=lambda x: x[1])
                    premise_category_3 = candidates[0][0]
                    premise_category_2 = random.choice(candidates[1:])[0]
                    #     similarities = sorted([(c, category_dataset.calculate_category_similarity(premise_category_1, c)) for c in category_dataset.category_list() if c not in (premise_category_1, conclusion_category)], reverse=True, key=lambda x: x[1])
                    #     premise_category_3 = similarities[1][0]

                    #     # For first argument's second premise, find the premise category with the closest SCM score to the first argument's first premise
                    #     arg1_p1_scm = self._scm(premise_category_1, "", "", conclusion_category, category_class)
                    #     arg2_candidates = [i for i in category_dataset.category_list() if i not in (premise_category_1, premise_category_3, conclusion_category)]
                        
                    #     for candidate in arg2_candidates:
                    #         candidate_scm = self._scm(candidate, "", "", conclusion_category, category_class)
                    #         if candidate_scm < arg1_p1_scm:
                    #             candidates.append((candidate, candidate_scm))
                    # candidates = sorted(candidates, reverse=True, key=lambda x: x[1])
                    # premise_category_2 = random.choice(candidates)[0]
                    


                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_3,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 7:

                    #premise_category_1, premise_category_2, premise_category_3, conclusion_category = random.sample(category_dataset.category_list(), k=4)
                    candidates = []
                    while len(candidates) < 1:
                        premise_category_1, premise_category_2, conclusion_category = random.sample(category_dataset.category_list(), k=3)
                        max_scm = np.max([self._scm(p, "", "", conclusion_category, category_class) for p in (premise_category_1, premise_category_2)])
                        candidates = [c for c in category_dataset.category_list() if c not in (premise_category_1, premise_category_2, conclusion_category) and self._scm(c, "", "", conclusion_category, category_class) < max_scm]
                    premise_category_3 = random.choice(candidates)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=premise_category_3,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 8:

                    premise_category_1, premise_category_2 = random.sample(category_dataset.category_list(), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=None,
                        arg1_premise3=None,
                        arg1_conclusion=premise_category_2,
                        arg2_premise1=premise_category_2,
                        arg2_premise2=None,
                        arg2_premise3=None,
                        arg2_conclusion=premise_category_1,
                    )

                elif phenomenon_number == 9:

                    premise_category_1, premise_category_2 = random.sample(category_dataset.category_list(), k=2)
                    premise_category_3 = random.choice(category_dataset_2.category_list())

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=premise_category_3,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 10:

                    premise_category_1, conclusion_category = random.sample(category_dataset.category_list(), k=2)
                    premise_category_2 = random.choice(category_dataset_2.category_list())

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=None,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                else:
                    # phenomenon_number == 11

                    conclusion_category = category_class
                    premise_category_1, conclusion_category_2 = random.sample(category_dataset.category_list(), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=None,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1, 
                        arg2_premise2=None,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category_2,
                    )
        

                candidate_argument_pairs.append(candidate_argument_pair)
        
        return candidate_argument_pairs

    def rank_candidate_argument_pairs(self, candidate_argument_pairs: List[CandidateArgumentPair]) -> pd.DataFrame:
        
        columns = [f.name for f in fields(candidate_argument_pairs[0])]
        candidate_argument_pairs_df = pd.DataFrame([astuple(c) for c in candidate_argument_pairs], columns=columns)

        ranked_df = None
        for phenomenon_number, pdf in candidate_argument_pairs_df.groupby("phenomenon_number"):

            if phenomenon_number in (1,2,4,5,6,7,11):
                
                pdf["category_class"] = [[category_class for category_class, category_dataset in self.category_datasets.items() if row["arg1_conclusion"] in category_dataset.category_list() or row["arg1_conclusion"] == category_class][0] for _, row in pdf.iterrows()]
                pdf["arg1_scm"] = [self._scm(row["arg1_premise1"], row["arg1_premise2"], row["arg1_premise3"], row["arg1_conclusion"], row["category_class"]) for _, row in pdf.iterrows()]
                pdf["arg2_scm"] = [self._scm(row["arg2_premise1"], row["arg2_premise2"], row["arg2_premise3"], row["arg2_conclusion"], row["category_class"]) for _, row in pdf.iterrows()]
                pdf["scm_diff"] = pdf["arg1_scm"] - pdf["arg2_scm"]

                sorted_pdf = pdf.sort_values(by="scm_diff", ascending=False)[columns]

                if phenomenon_number == 1:
                    ranked_df = sorted_pdf
                else:
                    ranked_df = pd.concat([ranked_df, sorted_pdf], axis=0)

                tdf = pdf.sort_values(by="scm_diff", ascending=False)
                print(phenomenon_number, tdf.iloc[0]["scm_diff"], tdf.iloc[200]["scm_diff"])
            
            else:
                ranked_df = pd.concat([ranked_df, pdf], axis=0)
        
        ranked_df["arg1_conclusion"] = ranked_df["arg1_conclusion"].apply(lambda x: self.category_class_rename[x] if x in self.category_class_rename else x)
        ranked_df["arg2_conclusion"] = ranked_df["arg2_conclusion"].apply(lambda x: self.category_class_rename[x] if x in self.category_class_rename else x)

        return ranked_df



class SyntheticOshersonCandidateGenerator(CandidateGenerator):
    """
    Generates synthetic argument pairs for Osherson phenomena 1-7 and 9-11
    """

    def __init__(self, category_dataset: CategoryDataset, osherson_phenomena: str = "data/osherson_phenomena.csv") -> None:
        super().__init__(category_dataset)

        self.phenomena_df = pd.read_csv(osherson_phenomena)[["phenomenon_number", "phenomenon_name", "phenomenon_type"]].drop_duplicates()

    def generate_candidate_argument_pairs(self, n: int) -> List[CandidateArgumentPair]:
        
        candidate_argument_pairs = []
        for _, row in self.phenomena_df.iterrows():

            phenomenon_number = row["phenomenon_number"]

            for _ in range(n):

                if phenomenon_number == 1:

                    conclusion_category = random.choice(self.category_dataset.class_list())
                    # conclusion_category = "mammal"
                    premise_category_1, premise_category_2 = random.sample(self.category_dataset.class_category_list(conclusion_category), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=None,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_2,
                        arg2_premise2=None,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 2:

                    conclusion_category = random.choice(self.category_dataset.class_list())
                    # conclusion_category = "mammal"
                    premise_category_1, premise_category_2, premise_category_3 = random.sample(self.category_dataset.class_category_list(conclusion_category), k=3)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_3,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 3:

                    conclusion_category = random.choice(self.category_dataset.class_list())
                    # conclusion_category = "mammal"
                    premise_category_1, premise_category_2 = random.sample(self.category_dataset.class_category_list(conclusion_category), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion="animal",
                    )

                elif phenomenon_number == 4:

                    conclusion_category = random.choice(self.category_dataset.class_list())
                    # conclusion_category = "mammal"
                    premise_category_1, premise_category_2, premise_category_3 = random.sample(self.category_dataset.class_category_list(conclusion_category), k=3)
        
                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=premise_category_3,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 5:

                    parent_class = random.choice(self.category_dataset.class_list())
                    while len(self.category_dataset.class_category_list(parent_class)) < 4:
                        parent_class = random.choice(self.category_dataset.class_list())
                    # parent_class = "mammal"
                    premise_category_1, premise_category_2, conclusion_category_1, conclusion_category_2 = random.sample(self.category_dataset.class_category_list(parent_class), k=4)
                    
                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category_1,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category_2,
                    )

                elif phenomenon_number == 6:

                    parent_class = random.choice(self.category_dataset.class_list())
                    while len(self.category_dataset.class_category_list(parent_class)) < 4:
                        parent_class = random.choice(self.category_dataset.class_list())
                    # parent_class = "mammal"
                    premise_category_1, premise_category_2, premise_category_3, conclusion_category = random.sample(self.category_dataset.class_category_list(parent_class), k=4)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_3,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 7:

                    parent_class = random.choice(self.category_dataset.class_list())
                    while len(self.category_dataset.class_category_list(parent_class)) < 4:
                        parent_class = random.choice(self.category_dataset.class_list())
                    # parent_class = "mammal"
                    premise_category_1, premise_category_2, premise_category_3, conclusion_category = random.sample(self.category_dataset.class_category_list(parent_class), k=4)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=premise_category_3,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 9:

                    conclusion_category, parent_class_2 = random.sample(self.category_dataset.class_list(), k=2)
                    # conclusion_category = "mammal"
                    parent_class_2 = random.choice(self.category_dataset.class_list())
                    while parent_class_2 == conclusion_category:
                        parent_class_2 = random.choice(self.category_dataset.class_list())

                    premise_category_1, premise_category_2 = random.sample(self.category_dataset.class_category_list(conclusion_category), k=2)
                    premise_category_3 = random.choice(self.category_dataset.class_category_list(parent_class_2))

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=premise_category_3,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=premise_category_2,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                elif phenomenon_number == 10:

                    parent_class_1, parent_class_2 = random.sample(self.category_dataset.class_list(), k=2)
                    # parent_class_1 = "mammal"
                    parent_class_2 = random.choice(self.category_dataset.class_list())
                    while parent_class_2 == parent_class_1:
                        parent_class_2 = random.choice(self.category_dataset.class_list())

                    premise_category_1, conclusion_category = random.sample(self.category_dataset.class_category_list(parent_class_1), k=2)
                    premise_category_2 = random.choice(self.category_dataset.class_category_list(parent_class_2))

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=premise_category_2,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category,
                        arg2_premise1=premise_category_1,
                        arg2_premise2=None,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category,
                    )

                else:
                    # phenomenon_number == 11

                    conclusion_category_1 = random.choice(self.category_dataset.class_list())
                    # conclusion_category_1 = "mammal"

                    premise_category_1, conclusion_category_2 = random.sample(self.category_dataset.class_category_list(conclusion_category_1), k=2)

                    candidate_argument_pair = CandidateArgumentPair(
                        phenomenon_number=phenomenon_number,
                        phenomenon_name=row["phenomenon_name"],
                        phenomenon_type=row["phenomenon_type"],
                        arg1_premise1=premise_category_1,
                        arg1_premise2=None,
                        arg1_premise3=None,
                        arg1_conclusion=conclusion_category_1,
                        arg2_premise1=premise_category_1, 
                        arg2_premise2=None,
                        arg2_premise3=None,
                        arg2_conclusion=conclusion_category_2,
                    )
        

                candidate_argument_pairs.append(candidate_argument_pair)
        
        return candidate_argument_pairs

    
    def rank_candidate_argument_pairs(self, candidate_argument_pairs: List[CandidateArgumentPair]) -> pd.DataFrame:
        candidate_argument_pairs_df = pd.DataFrame([astuple(c) for c in candidate_argument_pairs], columns=[f.name for f in fields(candidate_argument_pairs[0])])

        phenomena_scores = defaultdict(list)
        for _, row in candidate_argument_pairs_df.iterrows():

            arg1_typicality = np.mean([self.category_dataset.calculate_category_typicality(c) for c in (row[["arg1_premise1", "arg1_premise2", "arg1_premise3"]]) if c])
            arg2_typicality = np.mean([self.category_dataset.calculate_category_typicality(c) for c in (row[["arg2_premise1", "arg2_premise2", "arg2_premise3"]]) if c])

            typicality_diff = arg2_typicality - arg1_typicality

            arg1_premise_similarity = np.mean([self.category_dataset.calculate_category_similarity(a, b) for (a,b) in itertools.combinations([c for c in row[["arg1_premise1", "arg1_premise2", "arg1_premise3"]] if c], 2)])
            arg2_premise_similarity = np.mean([self.category_dataset.calculate_category_similarity(a, b) for (a,b) in itertools.combinations([c for c in row[["arg2_premise1", "arg2_premise2", "arg2_premise3"]] if c], 2)])

            premise_similarity_diff = arg2_premise_similarity - arg1_premise_similarity

            if row["arg1_conclusion"] in self.category_dataset.feature_map and row["arg2_conclusion"] in self.category_dataset.feature_map:
                arg1_conclusion_premise_similarity = np.mean([self.category_dataset.calculate_category_similarity(c, row["arg1_conclusion"]) for c in (row[["arg1_premise1", "arg1_premise2", "arg1_premise3"]]) if c])
                arg2_conclusion_premise_similarity = np.mean([self.category_dataset.calculate_category_similarity(c, row["arg2_conclusion"]) for c in (row[["arg2_premise1", "arg2_premise2", "arg2_premise3"]]) if c])

                conclusion_premise_similarity_diff = arg2_conclusion_premise_similarity - arg1_conclusion_premise_similarity
            else:
                conclusion_premise_similarity_diff = None

            arg1_num_premises = len([c for c in row[["arg1_premise1", "arg1_premise2", "arg1_premise3"]] if c])
            arg2_num_premises = len([c for c in row[["arg2_premise1", "arg2_premise2", "arg2_premise3"]] if c])

            num_premises_diff = arg2_num_premises - arg1_num_premises

            phenomena_scores["typicality_difference"].append(typicality_diff)
            phenomena_scores["premise_similarity_difference"].append(premise_similarity_diff)
            phenomena_scores["conclusion_premise_similarity_difference"].append(conclusion_premise_similarity_diff)
            phenomena_scores["premise_number_difference"].append(num_premises_diff)

        for k, v in phenomena_scores.items():
            candidate_argument_pairs_df[k] = v

        output_df = None
        for phenomenon_number, pdf in candidate_argument_pairs_df.groupby("phenomenon_number"):

            if phenomenon_number == 1:

                output_df = pdf.sort_values(by="typicality_difference", ascending=False)
            
            elif phenomenon_number in (2,6):

                output_df = pd.concat([output_df, pdf.sort_values(by="premise_similarity_difference", ascending=True)], axis=0)

            elif phenomenon_number in (3, 4, 7, 9, 10, 11):

                output_df = pd.concat([output_df, pdf], axis=0)

            elif phenomenon_number == 5:

                output_df = pd.concat([output_df, pdf.sort_values(by="conclusion_premise_similarity_difference", ascending=False)], axis=0)

        return output_df
