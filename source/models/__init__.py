from source.models.baseline_llm import BaselineLLM

load_model = {
    'baseline_llm': BaselineLLM,
}

# Replace the following with the model paths
get_llm_model_path = {
    'llama-3.1-8b'    : 'meta-llama/Llama-3.1-8B-Instruct'          ,

    'qwen2.5-0.5b'    : 'Qwen/Qwen2.5-0.5B-Instruct'                ,
    'qwen2.5-1.5b'    : 'Qwen/Qwen2.5-1.5B-Instruct'                ,
}