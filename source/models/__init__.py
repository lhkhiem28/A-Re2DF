from source.models.baseline_llm import BaselineLLM

load_model = {
    'baseline_llm': BaselineLLM,
}

# Replace the following with the model paths
get_llm_model_path = {
    'llama-3.2-1b'    : 'meta-llama/Llama-3.2-1B-Instruct'          ,
    'llama-3.2-3b'    : 'meta-llama/Llama-3.2-3B-Instruct'          ,
    'llama-3.1-8b'    : 'meta-llama/Llama-3.1-8B-Instruct'          ,
    'llama-3.1-70b'   : 'meta-llama/Llama-3.1-70B-Instruct'         ,
}