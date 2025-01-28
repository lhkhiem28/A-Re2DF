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
    test_dataset = load_dataset[args.dataset](path = args.path, data = args.data, split = "test", task = args.task, k_shot = args.k_shot, prompting = args.prompting, 
        selfies = "biot5" in args.llm_model_name, hit_thres = args.hit_thres
    )

    # Step 2: Build Model
    args.llm_model_path = get_llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    if args.checkpoint_path is not None:
        model = _reload_model(model, args.checkpoint_path)

    # Step 3. Evaluating
    model.eval()
    eval_output = []
    progress_bar_test = tqdm.tqdm(range(len(test_dataset)))

    code, code_error = 0, 0
    validity_work, total_work = 0, 1e-8
    for index in range(len(test_dataset)):
        batch = test_dataset[index]
        with torch.no_grad():
            if args.prompting != "react":
                if args.refine == "None":
                    if "t5" not in args.llm_model_name and "fm" not in args.llm_model_name:
                        output = model.inference(listize_fn(batch))
                        output["pred"] = [p.strip() if "->" not in p else p.split('->')[1].strip() for p in output["pred"]]
                        output["pred"] = [p.strip() if "becomes" not in p else p.split('becomes')[1].strip() for p in output["pred"]]
                        eval_output.append(output)
                    else:
                        prompt = batch["prompt"]
                        if "molt5" in args.llm_model_name:
                            prompt = prompt.split("\nDescription:")[1].split("\nAnswer:")[0]
                        if "biot5" in args.llm_model_name:
                            prompt = prompt.split("\nDescription:")[1].split("\nAnswer:")[0]
                            prompt = f'Definition: You are given a molecule description in English. Your job is to generate the molecule SELFIES that fits the description.\n\nNow complete the following example -\nInput: {prompt}\nOutput: '
                        if "text-chemt5" in args.llm_model_name:
                            prompt = prompt.split("\nDescription:")[1].split("\nAnswer:")[0]
                            prompt = f'Write in SMILES the described molecule: {prompt}'
                        pred = model.model.generate(
                            **model.tokenizer(prompt, return_tensors="pt").to("cuda"),
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred = model.tokenizer.decode(pred[0], skip_special_tokens=True).replace(' ', '')
                        if "\nAnswer:" in pred:
                            pred = pred.split("\nAnswer:")[1]
                        output = {'id': index,
                                'pred': [pred.strip()],
                                'label': batch['label'],
                        }
                        eval_output.append(output)
                elif args.refine == "self":
                    prop = args.data.split("/")[-1]
                    ori_prompt = batch['prompt']
                    if args.k_shot > 0:
                        ori_prompt = re.split(r'Examples:|Question:', ori_prompt)[0] + re.split(r'Examples:|Question:', ori_prompt)[2]
                    max_steps = args.refine_steps
                    for i in range(1, max_steps + 1):
                        total_work += 1
                        output = model.inference(listize_fn(batch))
                        output["pred"] = [p.strip() if "->" not in p else p.split('->')[1].strip() for p in output["pred"]]
                        output["pred"] = [p.strip() if "becomes" not in p else p.split('becomes')[1].strip() for p in output["pred"]]
                        if "MDesign" in args.data:
                            if i < max_steps:
                                feedback_batch = {
                                    'id': 0,
                                    'prompt': ori_prompt + output["pred"][0] + '\nEvaluate the designed molecule on its chemical validity and the desired specifications regarding functional groups according to the given description. Please provide two pieces of feedback. Start your feedback about validity with the phrase "Validity:", and start your feedback about the desired specifications with the phrase "Desired specifications:".',
                                    'label': None,
                                }
                                feedback_output = model.inference(listize_fn(feedback_batch))['pred'][0].strip().replace("\n\n", "\n")
                                batch['prompt'] = ori_prompt + output["pred"][0] + f"\n\nImprove the designed molecule based on the following feedback:\n{feedback_output}\nPlease answer with only the SMILES string of your designed molecule."
                            else:
                                eval_output.append(output)
                        elif "MModify" in args.data:
                            if i < max_steps:
                                feedback_batch = {
                                    'id': 0,
                                    'prompt': ori_prompt + output["pred"][0] + '\nEvaluate the modified molecule on its chemical validity and the desired property according to the requirement. Please provide two pieces of feedback. Start your feedback about validity with the phrase "Validity:", and start your feedback about the desired property with the phrase "Desired property:".',
                                    'label': None,
                                }
                                feedback_output = model.inference(listize_fn(feedback_batch))['pred'][0].strip().replace("\n\n", "\n")
                                batch['prompt'] = ori_prompt + output["pred"][0] + f"\n\nImprove the modified molecule based on the following feedback:\n{feedback_output}\nRespond with only the SMILES string of your modified molecule. No explanation is needed."
                            else:
                                eval_output.append(output)
                        else:
                            pass
                elif args.refine == "molt":
                    prop = args.data.split("/")[-1]
                    ori_prompt = batch['prompt']
                    if args.k_shot > 0:
                        ori_prompt = re.split(r'Examples:|Question:', ori_prompt)[0] + re.split(r'Examples:|Question:', ori_prompt)[2]
                    max_steps = args.refine_steps + 1
                    for i in range(1, max_steps + 1):
                        total_work += 1
                        output = model.inference(listize_fn(batch))
                        output["pred"] = [p.strip() if "->" not in p else p.split('->')[1].strip() for p in output["pred"]]
                        output["pred"] = [p.strip() if "becomes" not in p else p.split('becomes')[1].strip() for p in output["pred"]]
                        if "MDesign" in args.data:
                            if i < max_steps:
                                try:
                                    validity_feedback = get_validity_feedback(output["pred"][0], code_executor_agent)
                                    if "Error" in validity_feedback:
                                        feedback_output = "Validity:\n{}".format(validity_feedback)
                                    else:
                                        mol_pred, mol_label = Chem.MolFromSmiles(output["pred"][0]), Chem.MolFromSmiles(batch["smiles"])
                                        fgs_pred, fgs_label = count_fgs(mol_pred), count_fgs(mol_label)
                                        hit, _, str_pred, str_label = check_funchit(fgs_pred, fgs_label)
                                        if hit:
                                            eval_output.append(output)
                                            break
                                        else:
                                            try:
                                                specifications_feedback = f"The designed molecule has {str_pred} while the required molecule should have {str_label}. Therefore, the designed molecule is not correct."
                                                feedback_output = "Validity:\n{}\nDesired specifications:\n{}".format(validity_feedback, specifications_feedback)
                                            except:
                                                pass
                                except:
                                    continue
                                batch['prompt'] = ori_prompt + output["pred"][0] + f"\n\nImprove the designed molecule based on the following feedback:\n{feedback_output}\nPlease answer with only the SMILES string of your designed molecule."
                            else:
                                eval_output.append(output)
                        elif "MModify" in args.data:
                            if i < max_steps:
                                try:
                                    validity_feedback = get_validity_feedback(output["pred"][0], code_executor_agent)
                                    if "Error" in validity_feedback:
                                        feedback_output = "Validity:\n{}".format(validity_feedback)
                                    else:
                                        input_mol = Chem.MolFromSmiles(batch["smiles"])
                                        output_mol = Chem.MolFromSmiles(output["pred"][0])
                                        if "single" in args.data:
                                            input_prop = task2func[prop](input_mol)
                                            output_prop = task2func[prop](output_mol)
                                            if prop == "LogP+":
                                                hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_prop, 4)} and the original one has a LogP value of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "LogP-":
                                                hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_prop, 4)} and the original one has a LogP value of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "TPSA+":
                                                hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                                property_feedback = f"The modified molecule has a topological polar surface area (TPSA) of {round(output_prop, 4)} and the original one has a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "TPSA-":
                                                hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a topological polar surface area (TPSA) of {round(output_prop, 4)} and the original one has a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "HBD+":
                                                hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                                property_feedback = f"The modified molecule has {output_prop} hydrogen bond donors and the original one has {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                            if prop == "HBD-":
                                                hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has {output_prop} hydrogen bond donors and the original one has {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                            if prop == "HBA+":
                                                hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                                property_feedback = f"The modified molecule has {output_prop} hydrogen bond acceptors and the original one has {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                            if prop == "HBA-":
                                                hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has {output_prop} hydrogen bond acceptors and the original one has {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                            if prop == "QED+":
                                                hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                                property_feedback = f"The modified molecule has a quantitative estimation of drug-likeness of {round(output_prop, 4)} and the original one has a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "QED-":
                                                hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a quantitative estimation of drug-likeness of {round(output_prop, 4)} and the original one has a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if "multi" in args.data:
                                            logp, prop = prop[:5], prop[5:]
                                            input_logp, input_prop = task2func[logp](input_mol), task2func[prop](input_mol)
                                            output_logp, output_prop = task2func[logp](output_mol), task2func[prop](output_mol)
                                            if logp == "LogP+":
                                                if prop == "TPSA+":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                                if prop == "TPSA-":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                                if prop == "HBD+":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                                if prop == "HBD-":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                                if prop == "HBA+":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                                if prop == "HBA-":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                                if prop == "QED+":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                                if prop == "QED-":
                                                    hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if logp == "LogP-":
                                                if prop == "TPSA+":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                                if prop == "TPSA-":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                                if prop == "HBD+":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                                if prop == "HBD-":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                                if prop == "HBA+":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                                if prop == "HBA-":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                                if prop == "QED+":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                                if prop == "QED-":
                                                    hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                    DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                    property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if hit:
                                            eval_output.append(output)
                                            break
                                        else:
                                            try:
                                                feedback_output = "Validity:\n{}\nDesired property:\n{}".format(validity_feedback, property_feedback)
                                            except:
                                                pass
                                except:
                                    continue
                                batch['prompt'] = ori_prompt + output["pred"][0] + f"\n\nImprove the modified molecule based on the following feedback:\n{feedback_output}\nRespond with only the SMILES string of your modified molecule. No explanation is needed."
                            else:
                                eval_output.append(output)
                        else:
                            pass
                elif args.refine == "molt-retrieve":
                    prop = args.data.split("/")[-1]
                    ori_prompt = batch['prompt']
                    max_steps = args.refine_steps + 1
                    for i in range(1, max_steps + 1):
                        total_work += 1
                        output = model.inference(listize_fn(batch))
                        output["pred"] = [p.strip() if "->" not in p else p.split('->')[1].strip() for p in output["pred"]]
                        output["pred"] = [p.strip() if "becomes" not in p else p.split('becomes')[1].strip() for p in output["pred"]]
                        if i < max_steps:
                            try:
                                validity_feedback = get_validity_feedback(output["pred"][0], code_executor_agent)
                                if "Error" in validity_feedback:
                                    validity_work += 1
                                    feedback_output = "Validity:\n{}".format(validity_feedback)
                                else:
                                    input_mol = Chem.MolFromSmiles(batch["smiles"])
                                    output_mol = Chem.MolFromSmiles(output["pred"][0])
                                    if "single" in args.data:
                                        input_prop = task2func[prop](input_mol)
                                        output_prop = task2func[prop](output_mol)
                                        if prop == "LogP+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                            property_feedback = f"The modified molecule has a LogP value of {round(output_prop, 4)} and the original one has a LogP value of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if prop == "LogP-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            property_feedback = f"The modified molecule has a LogP value of {round(output_prop, 4)} and the original one has a LogP value of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if prop == "TPSA+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                            property_feedback = f"The modified molecule has a topological polar surface area (TPSA) of {round(output_prop, 4)} and the original one has a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if prop == "TPSA-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            property_feedback = f"The modified molecule has a topological polar surface area (TPSA) of {round(output_prop, 4)} and the original one has a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if prop == "HBD+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                            property_feedback = f"The modified molecule has {output_prop} hydrogen bond donors and the original one has {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                        if prop == "HBD-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            property_feedback = f"The modified molecule has {output_prop} hydrogen bond donors and the original one has {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                        if prop == "HBA+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                            property_feedback = f"The modified molecule has {output_prop} hydrogen bond acceptors and the original one has {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                        if prop == "HBA-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            property_feedback = f"The modified molecule has {output_prop} hydrogen bond acceptors and the original one has {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                        if prop == "QED+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                            property_feedback = f"The modified molecule has a quantitative estimation of drug-likeness of {round(output_prop, 4)} and the original one has a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if prop == "QED-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            property_feedback = f"The modified molecule has a quantitative estimation of drug-likeness of {round(output_prop, 4)} and the original one has a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                    if "multi" in args.data:
                                        logp, prop = prop[:5], prop[5:]
                                        input_logp, input_prop = task2func[logp](input_mol), task2func[prop](input_mol)
                                        output_logp, output_prop = task2func[logp](output_mol), task2func[prop](output_mol)
                                        if logp == "LogP+":
                                            if prop == "TPSA+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "TPSA-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "HBD+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                            if prop == "HBD-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                            if prop == "HBA+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                            if prop == "HBA-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                            if prop == "QED+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "QED-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                        if logp == "LogP-":
                                            if prop == "TPSA+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "TPSA-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a topological polar surface area (TPSA) of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a topological polar surface area (TPSA) of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "HBD+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                            if prop == "HBD-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond donors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond donors. Therefore, the modified molecule is not correct."
                                            if prop == "HBA+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                            if prop == "HBA-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and {output_prop} hydrogen bond acceptors, and the original one has a LogP value of {round(input_logp, 4)} and {input_prop} hydrogen bond acceptors. Therefore, the modified molecule is not correct."
                                            if prop == "QED+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                            if prop == "QED-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                                property_feedback = f"The modified molecule has a LogP value of {round(output_logp, 4)} and a quantitative estimation of drug-likeness of {round(output_prop, 4)}, and the original one has a LogP value of {round(input_logp, 4)} and a quantitative estimation of drug-likeness of {round(input_prop, 4)}. Therefore, the modified molecule is not correct."
                                    if hit:
                                        eval_output.append(output)
                                        break
                                    else:
                                        try:
                                            feedback_output = "Validity:\n{}\nDesired property:\n{}".format(validity_feedback, property_feedback)
                                            DB["sim"] = DB["mol"].apply(lambda m: DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(output_mol, 2), AllChem.GetMorganFingerprint(m, 2)))
                                            example = DB.sort_values(by=['sim'], ascending=False).iloc[0]["SMILES"]
                                            feedback_output += f"\n\nFor your reference, we find a molecule {example} which is correct and similar to your modified molecule."
                                        except:
                                            pass
                            except:
                                continue
                            batch['prompt'] = ori_prompt + output["pred"][0] + f"\n\nImprove the modified molecule based on the following feedback:\n{feedback_output}\nRespond with only the SMILES string of your modified molecule. No explanation is needed."
                        else:
                            eval_output.append(output)
                elif args.refine == "retrieve":
                    prop = args.data.split("/")[-1]
                    ori_prompt = batch['prompt']
                    max_steps = args.refine_steps
                    for i in range(1, max_steps + 1):
                        total_work += 1
                        output = model.inference(listize_fn(batch))
                        output["pred"] = [p.strip() if "->" not in p else p.split('->')[1].strip() for p in output["pred"]]
                        output["pred"] = [p.strip() if "becomes" not in p else p.split('becomes')[1].strip() for p in output["pred"]]
                        if i < max_steps:
                            try:
                                input_mol = Chem.MolFromSmiles(batch["smiles"])
                                output_mol = Chem.MolFromSmiles(output["pred"][0])
                                if output_mol is None:
                                    eval_output.append(output)
                                    break
                                else:
                                    if "single" in args.data:
                                        input_prop = task2func[prop](input_mol)
                                        output_prop = task2func[prop](output_mol)
                                        if prop == "LogP+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                        if prop == "LogP-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                        if prop == "TPSA+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                        if prop == "TPSA-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                        if prop == "HBD+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                        if prop == "HBD-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                        if prop == "HBA+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                        if prop == "HBA-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                        if prop == "QED+":
                                            hit = output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                            DB = test_dataset.DB[test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0]]
                                        if prop == "QED-":
                                            hit = output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                            DB = test_dataset.DB[test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                    if "multi" in args.data:
                                        logp, prop = prop[:5], prop[5:]
                                        input_logp, input_prop = task2func[logp](input_mol), task2func[prop](input_mol)
                                        output_logp, output_prop = task2func[logp](output_mol), task2func[prop](output_mol)
                                        if logp == "LogP+":
                                            if prop == "TPSA+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "TPSA-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            if prop == "HBD+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "HBD-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            if prop == "HBA+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "HBA-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            if prop == "QED+":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "QED-":
                                                hit = output_logp > input_logp + task2thres[logp][args.hit_thres][0] and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] > input_logp + task2thres[logp][args.hit_thres][0]) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                        if logp == "LogP-":
                                            if prop == "TPSA+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "TPSA-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            if prop == "HBD+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "HBD-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            if prop == "HBA+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "HBA-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                            if prop == "QED+":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop > input_prop + task2thres[prop][args.hit_thres][0]
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & (test_dataset.DB["prop"] > input_prop + task2thres[prop][args.hit_thres][0])]
                                            if prop == "QED-":
                                                hit = output_logp + task2thres[logp][args.hit_thres][0] < input_logp and output_prop + task2thres[prop][args.hit_thres][0] < input_prop
                                                DB = test_dataset.DB[(test_dataset.DB["logp"] + task2thres[logp][args.hit_thres][0] < input_logp) & test_dataset.DB["prop"] + task2thres[prop][args.hit_thres][0] < input_prop]
                                    if hit:
                                        eval_output.append(output)
                                        break
                                    else:
                                        DB["sim"] = DB["mol"].apply(lambda m: DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(output_mol, 2), AllChem.GetMorganFingerprint(m, 2)))
                                        example = DB.sort_values(by=['sim'], ascending=False).iloc[0]["SMILES"]
                                        feedback_output = f"The provided molecule is not correct. We find a molecule {example} which is correct and similar to the provided molecule. Can you give me a new molecule?"
                                        batch['prompt'] = ori_prompt + output["pred"][0] + f"\n{feedback_output}\nRespond with only the SMILES string of your new molecule. No explanation is needed."
                            except:
                                eval_output.append(output)
                                break
                        else:
                            eval_output.append(output)
                else:
                    pass
            else:
                max_steps = 5
                for i in range(1, max_steps + 1):
                    batch['prompt'] += f"\nThought {i}: "
                    output = model.inference(listize_fn(batch))
                    try:
                        thought, action = output['pred'][0].split(f"Observation {i}: ")[0].split(f"Action {i}: ")
                        thought, action = thought.strip().replace(f"Thought {i}: ", ""), action.strip()
                        if action.startswith("RDKit"):
                            smiles, prop = re.search(r'\<(.*?)\>', action).group(1).split(", ")
                            smiles, prop = smiles.strip(), prop.strip()
                            code_batch = {
                                'id': 0,
                                'prompt': f"Write a Python code snippet using RDKit to compute the {prop} of the molecule with the SMILES string {smiles}\nThe code should print: The {prop} of the given molecule is {{computed {prop}}}, rounded to one decimal places.\nPlease only reply with the code in a markdown code block with the language set to Python.",
                                'label': None,
                            }
                            if args.doc_refer:
                                try:
                                    code_batch["prompt"] += f"\nRefer to the documentation below for accurate code generation.\n"
                                    code_batch["prompt"] += doc_retrieve(prop)
                                except:
                                    continue
                            code_output = model.inference(listize_fn(code_batch))['pred'][0]
                            obs = code_executor_agent.generate_reply(messages=[{"role": "user", "content": code_output}]).split("Code output: ")[-1].strip()
                            status = True
                            code += 1
                            if "Error" in obs:
                                obs = f'Error in using RDKit to compute the {prop} of the given molecule, predict its {prop} by yourself and proceed with the next step.'
                                status = False
                                code_error += 1

                            if not status:
                                print("Error in using RDKit")
                            batch['prompt'] += f"{thought}\nAction {i}: {action}\nObservation {i}: {obs}"
                        elif action.startswith("Finish"):
                            eval_output.append(output)
                            break
                    except:
                        eval_output.append(output)
                        break

                    if i == max_steps:
                        eval_output.append(output)

        progress_bar_test.update(1)

    # Step 4. Post-processing & Evaluating
    os.makedirs(f'{args.output_dir}/inference/{args.data}', exist_ok=True)
    path = f'{args.output_dir}/inference/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.split}_k_shot{args.k_shot}_{args.prompting}_{args.refine}_refine_steps{args.refine_steps}_{args.hit_thres}.csv'
    scores = eval_funcs[args.dataset](eval_output, path, model.tokenizer, args.data, 
        selfies = "biot5" in args.llm_model_name, hit_thres = args.hit_thres
    )
    if "MPP" in args.data:
        print("Accuracy: {:.4f} F1: {:.4f}".format(
            *scores
        ))
        if args.prompting == "react":
            print("Coding Error: {:.2f}%".format(
                100*(code_error/code)
            ))
    elif "MDesign" in args.data:
        print("ExactMatch: {:.4f} Levenshtein: {:.4f} BLEU-2: {:.4f} BLEU-4: {:.4f} MACCS-FTS: {:.4f} Morgan-FTS: {:.4f} FCD: {:.4f} Validity: {:.4f}".format(
            *scores
        ))
    elif "MModify" in args.data:
        print("Hit: {:.2f} Hit@0.4: {:.2f} Hit@0.5: {:.2f} Morgan-FTS: {:.2f} Validity: {:.2f} Validity check: {:.2f}".format(
            *scores, 100*validity_work/total_work
        ))
    else:
        pass

if __name__ == "__main__":
    args = parse_args_llm().parse_args()
    print(f'{args.output_dir}/inference/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.split}_k_shot{args.k_shot}_{args.prompting}_{args.refine}_refine_steps{args.refine_steps}_{args.hit_thres}.csv')
    main(args)