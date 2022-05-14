from dataclasses import dataclass


@dataclass
class ExperimentResult:

    max_tokens: int
    engine: str
    raw_api_response: str
    temperature: float
    logprobs: int


@dataclass
class YesProbabilityExperimentResult(ExperimentResult):

    yes_logprob: float
    no_logprob: float


@dataclass
class ConclusionProbabilityExperimentResult(ExperimentResult):

    conclusion_logprob: float


@dataclass
class Prompt:

    prompt: str
    property: str


@dataclass
class OshersonPrompt(Prompt):

    phenomenon_number: int
    phenomenon_name: str
    phenomenon_type: str

    premise_category_1: str
    premise_category_2: str
    premise_category_3: str
    conclusion_category: str