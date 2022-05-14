import CandidateGenerator
import CategoryDataset
import PromptGenerator
import ExperimentSubmitter
import PropertyGenerator
import schema
import config
import helpers

import pandas as pd

import dataclasses
import torch
import scipy.special
import logging


def generate_figure_1_data(property_generator: PropertyGenerator.PropertyGenerator) -> None:
    """
    Figure 1: Does GPT-3 account for Osherson phenomena?
    This function generates two sets of property induction arguments that can be passed onto GPT-3: one set based on the arguments used by Osherson et al. (1990), and one 'synthetic' set based on categories from the Leuven category dataset.
    It then generates prompts using both of these sets, sends the prompts to the OpenAI API and records its responses.
    """

    logging.info("Generating figure 1 data...")

    # Generate prompts using real osherson data
    df = pd.read_csv(config.osherson_arguments_path).fillna("")
    osherson_df_rows = [(p, pdf["phenomenon_name"].iloc[0], pdf["phenomenon_type"].iloc[0], pdf["p1_cat"].iloc[0].lower()[:-1], pdf["p2_cat"].iloc[0].lower()[:-1], pdf["p3_cat"].iloc[0].lower()[:-1], pdf["c_cat"].iloc[0].lower()[:-1], pdf["p1_cat"].iloc[1].lower()[:-1], pdf["p2_cat"].iloc[1].lower()[:-1], pdf["p3_cat"].iloc[1].lower()[:-1], pdf["c_cat"].iloc[1].lower()[:-1]) for p, pdf in df.groupby("phenomenon_number")]
    osherson_df = pd.DataFrame(osherson_df_rows, columns=["phenomenon_number","phenomenon_name","phenomenon_type","arg1_premise1","arg1_premise2","arg1_premise3","arg1_conclusion","arg2_premise1","arg2_premise2","arg2_premise3","arg2_conclusion"])
    osherson_df.to_csv(config.prompt_arguments_path("osherson"))

    # Generate prompts using synthetic osherson data
    category_datasets = {c: CategoryDataset.DeDeyneCategoryDataset(c) for c in config.CATEGORY_CLASSES}
    candidate_generator = CandidateGenerator.SyntheticOshersonSCMCandidateGenerator(category_datasets)
    candidate_generator.generate_candidate_argument_pair_dataset(config.prompt_arguments_path("osherson"), config.NUM_SYNTHETIC_ARGUMENTS)

    for argument_type in ("osherson", "synthetic"):
        prompt_generator = PromptGenerator.InstructPromptGenerator(property_generator=property_generator, arguments_loc=config.prompt_arguments_path(argument_type))
        experiment_submitter = ExperimentSubmitter.YesProbabilityExperimentSubmitter(prompt_generator=prompt_generator)
        experiment_submitter.run_experiment(output_filepath=config.figure_1_data_path(argument_type), max_tokens=0, check_experiment=True, engine=config.openai_engine)


def generate_figure_2a_data(property_generator: PropertyGenerator.PropertyGenerator) -> None:
    """
    Figure 2A: Does GPT-3 account for human argument strength ratings?
    This function converts the sets of human rated specific and general arguments into GPT-3 prompts before sending the prompts to the OpenAI API and recording its responses
    """

    logging.info("Generating figure 2a data...")

    for conclusion_type in ("specific", "general"):
        
        human_strength_df = helpers.read_osherson_ranked_arguments(conclusion_type=conclusion_type)

        # Generate prompts
        prompt_generator = PromptGenerator.InstructPromptGenerator(None, None, False)
        prompts = [
            schema.OshersonPrompt(
                phenomenon_number=None,
                phenomenon_name=None,
                phenomenon_type=None,
                prompt=prompt_generator.generate_prompt(row["premise_1"], row["premise_2"], row["premise_3"], row["conclusion"], property_generator.get_property()),
                property=property_generator.get_property(),
                premise_category_1=row["premise_1"],
                premise_category_2=row["premise_2"],
                premise_category_3=row["premise_3"],
                conclusion_category=row["conclusion"]
            )
            for _, row in human_strength_df.iterrows()
        ]
        
        # Get results
        experiment_submitter = ExperimentSubmitter.YesProbabilityExperimentSubmitter(prompt_generator=None)
        results = experiment_submitter.submit_experiment(prompts=prompts, output_filepath=config.figure_2a_data_path(conclusion_type), logprobs=5, temperature=0, max_tokens=0, engine=config.openai_engine, check_experiment=True)
        rows = [dataclasses.astuple(prompts[i]) + dataclasses.astuple(results[i]) for i in range(len(prompts))]
        columns = [f.name for f in dataclasses.fields(prompts[0])] + [f.name for f in dataclasses.fields(results[0])]
        experiment_df = pd.DataFrame(rows, columns=columns)

        experiment_df.to_csv(config.figure_2a_data_path(conclusion_type))


def generate_figure_2bc_data() -> None:
    """
    Figure 2B/C: Can GPT-3 account for human argument strength ratings?
    This function generates embeddings for category strings by quering the OpenAI Embeddings API.
    """

    logging.info("Generating figure 2b/c data...")


    feature_generator = CategoryDataset.KempLeuvenCategoryDataset()
    rows = []

    embedding = helpers.get_embedding("all animals")
    rows.append(("all animals", embedding))

    for category_class in feature_generator.class_list():
        for category in feature_generator.class_category_list(category_class):
            
            pluralised_category = PromptGenerator.convert_category_to_plural(category.lower())
            embedding = helpers.get_embedding(pluralised_category)
            rows.append((pluralised_category.capitalize(), embedding))

        pluralised_category = PromptGenerator.convert_category_to_plural(category_class.lower())
        embedding = helpers.get_embedding(pluralised_category)
        rows.append((pluralised_category.capitalize(), embedding))
        
    df = pd.DataFrame(rows, columns=["category", "embedding"])
    df.to_csv(config.figure_2bc_data_path)


def generate_figure_3_data(property_generator: PropertyGenerator.PropertyGenerator) -> None:
    """
    Figure 3: Will GPT-3 improve with scale?
    Here we do the same as generate_figure_2_data, except we use GPT/GPT-2 (as made available on HuggingFace)
    """

    logging.info("Generating figure 3 data...")

    argument_rows = []
    for model_string in config.GPT_MODELS:

        logging.info(f"    Generating data for {model_string}...")

        model, tokenizer = config.model_init(model_string)

        # Get Osherson diversity argument judgements
        yes_id = tokenizer.encode(" Yes")[0]
        no_id = tokenizer.encode(" No")[0]
        for conclusion_type in ("specific", "general"):
    
            human_strength_df = helpers.read_osherson_ranked_arguments(conclusion_type=conclusion_type)

            prompt_generator = PromptGenerator.InstructPromptGenerator(None, None, False)
            prompts = [
                (
                    prompt_generator.generate_prompt(row["premise_1"], row["premise_2"], row["premise_3"], row["conclusion"], property_generator.get_property()),
                    row["premise_1"],
                    row["premise_2"],
                    row["premise_3"],
                    row["conclusion"]
                )
                for _, row in human_strength_df.iterrows()
            ]
            
            for prompt in prompts:

                input_ids = torch.tensor(tokenizer.encode(prompt[0])).unsqueeze(0)
                
                assert input_ids[0][-1] == yes_id

                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)

                _, logits = outputs[:2]
                logits = scipy.special.softmax(logits[0][-1].numpy())
                yes_prob = logits[yes_id]
                no_prob = logits[no_id]

                argument_rows.append((model_string, conclusion_type, yes_prob / (yes_prob + no_prob)) + prompt)

        df = pd.DataFrame(argument_rows, columns=["model", "conclusion_type", "yes_prob", "prompt", "premise_1", "premise_2", "premise_3", "conclusion"])
        df.to_csv(config.figure_3_data_path)


if __name__ == "__main__":

    property_generator = PropertyGenerator.NamelessPropertyGenerator() 

    generate_figure_1_data(property_generator=property_generator)
    generate_figure_2a_data(property_generator=property_generator)
    generate_figure_2bc_data()
    generate_figure_3_data(property_generator=property_generator)
