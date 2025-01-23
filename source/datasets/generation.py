import tqdm
import selfies as sf
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from rank_bm25 import BM25Okapi
from source.utils.evaluation import *

class DatasetGeneration(Dataset):
    def __init__(self, path, data, split, task="", k_shot=0, prompting="None", selfies=False, hit_thres=0):
        super().__init__()
        self.data = data
        self.task = task
        self.selfies = selfies
        self.hit_thres = hit_thres
        self.k_shot = k_shot
        self.prompting = prompting
        self.questions = pd.read_csv(f'{path}/{data}/{split}.csv')
        self.questions["SMILES"] = self.questions["SMILES"].str.replace('\\\\', '\\')
        if "MDesign" in data:
            self.questions = self.questions.sample(n=100, random_state=0).reset_index(drop=True)
        elif "MModify" in data:
            DB_path = "/".join(data.split("/")[:-1])
            self.DB = pd.read_csv(f'{path}/{DB_path}/DB.csv')
            self.DB["mol"] = self.DB["SMILES"].apply(Chem.MolFromSmiles)
            prop = data.split("/")[-1]
            if "single" in data:
                self.DB["prop"] = self.DB["mol"].apply(task2func[prop])
                if self.hit_thres > 0:
                    if "HB" in data:
                        self.questions["Text"] = self.questions["Text"].apply(lambda s: s.replace(". The modified molecule should be similar to the original one.", f' by at least {task2thres[prop][self.hit_thres][0] + 1}. The modified molecule should be similar to the original one.'))
                    else:
                        self.questions["Text"] = self.questions["Text"].apply(lambda s: s.replace(". The modified molecule should be similar to the original one.", f' by at least {task2thres[prop][self.hit_thres][0]}. The modified molecule should be similar to the original one.'))
            if "multi" in data:
                logp, prop = prop[:5], prop[5:]
                self.DB["logp"], self.DB["prop"] = self.DB["mol"].apply(task2func[logp]), self.DB["mol"].apply(task2func[prop])
                if self.hit_thres > 0:
                    if "HB" in data:
                        self.questions["Text"] = self.questions["Text"].apply(lambda s: s.replace("LogP value", f'LogP value by at least {task2thres[logp][self.hit_thres][0]}').replace(". The modified molecule should be similar to the original one.", f' by at least {task2thres[prop][self.hit_thres][0] + 1}. The modified molecule should be similar to the original one.'))
                    else:
                        self.questions["Text"] = self.questions["Text"].apply(lambda s: s.replace("LogP value", f'LogP value by at least {task2thres[logp][self.hit_thres][0]}').replace(". The modified molecule should be similar to the original one.", f' by at least {task2thres[prop][self.hit_thres][0]}. The modified molecule should be similar to the original one.'))

        if self.k_shot > 0:
            self.examples = json.load(open(f'{path}/{data}/examples.json'))
        if self.prompting == "retrieve":
            self.train = pd.read_csv(f'{path}/{data}/train.csv')
            self.bm25 = BM25Okapi([desc.split(" ") for desc in self.train["Description"].values.tolist()])

    def retrieve_Examples(self, description):
        indices = np.argsort(self.bm25.get_scores(description.split(" ")))[::-1][:self.k_shot].tolist()
        Examples = ""
        for i in indices:
            item = self.train.iloc[i]
            Examples += f'\nDescription:{item["Description"]}\nAnswer:{item["SMILES"]}'
        return Examples

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        item = self.questions.iloc[index]

        if "MPP" in self.data:
            if self.k_shot > 0:
                if self.prompting == "None":
                    question = f'{item["Text"]} Please answer the question with only Yes or No.'
                    Examples = self.examples[f'{self.k_shot}_shot']
                    question += f'\nExamples:{Examples}'
                    smiles = f'\nQuestion:\nMolecule:{item["SMILES"]}\nAnswer:'
                elif self.prompting == "cot":
                    question = f'{item["Text"]}'
                    Examples = self.examples[f'{self.k_shot}_shot_cot']
                    question += f'\nExamples:{Examples}'
                    smiles = f'\nPlease follow the format of the given examples.\nQuestion:\nMolecule:{item["SMILES"]}\nAnswer:'
                elif self.prompting == "react":
                    question = f'{item["Text"]}\nSolve the question with interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be two types:\n(1) RDKit<property>, which returns the property of the given molecule computed by RDKit.\n(2) Finish<answer>, which returns the answer and finishes the task.'
                    Examples = self.examples[f'{self.k_shot}_shot_react']
                    question += f'\nHere are some examples:{Examples}'
                    smiles = f'\nQuestion:\nMolecule:{item["SMILES"]}'
                else:
                    pass
            else:
                question = f'{item["Text"]} Please answer the question with only Yes or No.'
                smiles = f'\nMolecule:{item["SMILES"]}\nAnswer:'
            return {
                'id': index,
                'smiles': item["SMILES"],
                'prompt': f'{question}\n{smiles}',
                'label': str(item[self.task]),
            }
        elif "MDesign" in self.data:
            if self.k_shot > 0:
                if self.prompting == "None":
                    question = f'{item["Text"]} Please answer with only the SMILES string of your designed molecule.'
                    Examples = self.examples[f'{self.k_shot}_shot']
                    question += f'\nExamples:{Examples}'
                    description = f'\nQuestion:\nDescription:{item["Description"]}\nAnswer:'
                elif self.prompting == "retrieve":
                    question = f'{item["Text"]} Please answer with only the SMILES string of your designed molecule.'
                    Examples = self.retrieve_Examples(item["Description"])
                    question += f'\nExamples:{Examples}'
                    description = f'\nQuestion:\nDescription:{item["Description"]}\nAnswer:'
                else:
                    pass
            else:
                question = f'{item["Text"]} Please answer with only the SMILES string of your designed molecule.'
                description = f'\nDescription:{item["Description"]}\nAnswer:'
            if not self.selfies:
                return {
                    'id': index,
                    'smiles': item["SMILES"],
                    'description': item["Description"],
                    'prompt': f'{question}\n{description}',
                    'label': item["SMILES"],
                }
            else:
                return {
                    'id': index,
                    'smiles': sf.encoder(item["SMILES"]),
                    'description': item["Description"],
                    'prompt': f'{question}\n{description}',
                    'label': sf.encoder(item["SMILES"]),
                }
        elif "MModify" in self.data:
            if self.k_shot > 0:
                pass
            else:
                question = f'{item["Text"]}\nRespond with only the SMILES string of your modified molecule. No explanation is needed.'
                smiles = f'\nMolecule:{item["SMILES"]}\nAnswer:'
            return {
                'id': index,
                'smiles': item["SMILES"],
                'prompt': f'{question}\n{smiles}',
                'label': item["SMILES"],
            }
        else:
            pass