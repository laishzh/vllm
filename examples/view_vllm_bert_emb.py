
from vllm import LLM
from vllm.model_executor.models.bert_embedding import BertEmbeddingModel

# Sample prompts.
prompts = [
    # "This is an example sentence.",
    # "Another sentence.",
    "今天天气怎么样？好一些了吧？"
]

# Create an LLM.
model = LLM(model="bert-base-uncased", enforce_eager=True)
# model = LLM(model="google-bert/bert-base-multilingual-uncased", enforce_eager=True)
# model = LLM(model="google-bert/bert-large-uncased", enforce_eager=True)

def output_model_parameters(model: BertEmbeddingModel):
    for name, weight in model.named_parameters():
        print(f"Name: {name}".ljust(60) + f"Weight: {weight.shape}".ljust(40) + f"dtype: {weight.dtype}".ljust(30) + f"\nweight_value: {weight.data}")

output_model_parameters(model)