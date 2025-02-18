from source.models.baseline_llm import BaselineLLM

load_model = {
    'baseline_llm': BaselineLLM,
}

# Replace the following with the model paths
get_llm_model_path = {
    #llama-3.2
    'llama-3.2-1b'    : 'meta-llama/Llama-3.2-1B-Instruct'      ,
    'llama-3.2-3b'    : 'meta-llama/Llama-3.2-3B-Instruct'      ,
    #qwen2.5
    'qwen2.5-7b'      : 'Qwen/Qwen2.5-7B-Instruct'              ,
    'qwen2.5-14b'     : 'Qwen/Qwen2.5-14B-Instruct'             ,
}