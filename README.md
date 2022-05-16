# GPT-3-Property-Induction

This repo contains the code and data for our paper:

Han, S. J., Ransom, K. J., Perfors, A., & Kemp, C. (2022). Human-like property induction is a challenge for large language models. *Proceedings of the 44th Annual Meeting of the Cognitive Science Society*.

To generate the data used in our analysis, run `main.py`. Please note that this requires an active OpenAI API key.

To run our analysis and regenerate our figures, use the code provided in `Analysis.ipynb`.

#### Overview

If you'd like to extend our code, here's an overview of what everything does:

- `CandidateGenerator` defines a class of objects that generate the synthetic argument dataset that we use in figure 1.
- `ExperimentSubmitter` defines a class of objects that allow us to submit sets of prompts to the OpenAI API. The abstract base class provides a method for estimating the cost of an experiment, while concrete implementations specify how GPT-3 argument strength is calculated via the `submit_experiment` method.
- `PromptGenerator` defines a class of objects that convert a set of categories and properties into a stylised prompt for GPT-3 to respond to.
- `PropertyGenerator` defines a class of objects that provide an argument property.
- `CategoryDataset` defines a class of objects that allows us to interface with various category feature datasets.

