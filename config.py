from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

openai_engine = "text-davinci-001"

osherson_arguments_path = "data/osherson_phenomena.csv"
ranked_arguments_path = lambda x: f"data/{x}arguments.txt"

prompt_arguments_path = lambda x: f"data/{x}_prompt_arguments.csv"

figure_1_data_path = lambda x: f"figure_1_{x}_data.csv"
figure_2a_data_path = lambda x: f"data/figure_2a_{x}_data.csv"
figure_2bc_data_path = f"data/figure_2bc_data.csv"
figure_3_data_path = "data/figure_3_data.csv"



# Hardcoded version of 'data/structstat/premisenumbering.txt'
PREMISE_NUMBERS = {
    1:"horse",
    2:"cow",
    3:"chimp",
    4:"gorilla",
    5:"mouse",
    6:"squirrel",
    7:"dolphin",
    8:"seal",
    9:"elephant",
    10:"rhino",
}

CATEGORY_CLASSES = ["Mammals", "Fish", "Birds", "Insects", "Reptiles"]
GPT_MODELS = ["gpt", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
NUM_SYNTHETIC_ARGUMENTS = 200


def model_init(model_string):
    if model_string.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        model = GPT2LMHeadModel.from_pretrained(model_string)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
    model.eval()
    return model, tokenizer