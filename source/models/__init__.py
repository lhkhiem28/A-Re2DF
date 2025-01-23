from source.models.baseline_llm import BaselineLLM

load_model = {
    'baseline_llm': BaselineLLM,
}

# Replace the following with the model paths
get_llm_model_path = {
    #llama-3.1
    'llama-3.1-8b'    : 'meta-llama/Meta-Llama-3.1-8B-Instruct' ,
    'llama-3.1-70b'   : 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    #qwen2.5
    'qwen2.5-7b'      : 'Qwen/Qwen2.5-7B-Instruct'              ,
    'qwen2.5-14b'     : 'Qwen/Qwen2.5-14B-Instruct'             ,
    'qwen2.5-32b'     : 'Qwen/Qwen2.5-32B-Instruct'             ,
    'qwen2.5-72b'     : 'Qwen/Qwen2.5-72B-Instruct'             ,
}