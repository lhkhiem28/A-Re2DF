import argparse

def parse_args_llm():
    parser = argparse.ArgumentParser(description="ARe2DF")
    parser.add_argument("--project", type=str, default="ARe2DF")
    parser.add_argument("--seed", type=int, default=0)

    # Model related
    parser.add_argument("--model_name", type=str, default='baseline_llm')
    parser.add_argument("--llm_model_name", type=str)
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--n_gpus", type=int, default=2)

    # Model Training
    parser.add_argument("--dataset", type=str, default='generation')
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--prompting", type=str, default="None")
    parser.add_argument("--refine", type=str, default="None")
    parser.add_argument("--refine_steps", type=int, default=3)
    parser.add_argument("--hit_thres", type=int, default=0)
    parser.add_argument('--doc_refer', action='store_true', default=False)
    parser.add_argument("--path", type=str, default='../ARe2DF-datasets')
    parser.add_argument("--data", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task", type=str, default="Label")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)

    # Inference
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Checkpoint
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser