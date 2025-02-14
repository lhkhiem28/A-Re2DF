import os
import tqdm
import torch
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings("ignore")

from source.utils.help_funcs import seed_everything
from source.config import parse_args_llm
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.models import load_model, get_llm_model_path
from source.datasets import load_dataset
from source.utils.evaluation import *
from source.utils.help_funcs import collate_fn, listize_fn
from utils import doc_retrieve, get_validity_feedback
import re
import tempfile
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor

temp_dir = tempfile.TemporaryDirectory()
executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir=temp_dir.name,
)
code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="NEVER",
)

def main(args):
    seed = args.seed
    seed_everything(seed=seed)

    # Step 1: Build Dataset
    train_dataset = load_dataset[args.dataset](path = args.path, data = args.data, split = "train", task = args.task, k_shot = args.k_shot, prompting = args.prompting, 
        hit_thres = args.hit_thres
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True, collate_fn=collate_fn)

    # Step 2: Build Model and Optimizer
    args.llm_model_path = get_llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    trainable_params, all_param = model.print_trainable_params()
    print("-"*len(f"No. Trainable Params: {trainable_params} ({100 * trainable_params / all_param:.4f} %)"))
    print(f"No. Trainable Params: {trainable_params} ({100 * trainable_params / all_param:.4f} %)")
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}],
        betas=(0.9, 0.999)
    )
    if args.checkpoint_path is not None:
        model = _reload_model(model, args.checkpoint_path)

    # Step 3. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps))
    for epoch in range(1, args.num_epochs+1):
        model.train()
        epoch_loss = 0.
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

            progress_bar.update(1)

        print(f"Epoch {epoch}|{args.num_epochs}: Train Loss: {epoch_loss / len(train_loader):.4f}")
        _save_checkpoint(model, optimizer, epoch, args, is_best=True)

if __name__ == "__main__":
    args = parse_args_llm().parse_args()
    print(f'{args.output_dir}/inference/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.split}_k_shot{args.k_shot}_{args.prompting}_{args.refine}_refine_steps{args.refine_steps}_{args.hit_thres}.csv')
    main(args)