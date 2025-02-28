import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import DataStructs

task2thres = {
    "LogP+" : [[0], [0.5]],
    "LogP-" : [[0], [0.5]],
    "TPSA+" : [[0], [10]],
    "TPSA-" : [[0], [10]],
    "HBD+"  : [[0], [1]],
    "HBD-"  : [[0], [1]],
    "HBA+"  : [[0], [1]],
    "HBA-"  : [[0], [1]],
    "QED+"  : [[0], [0.1]],
    "QED-"  : [[0], [0.1]],
}
task2prop = {
    "LogP+" : "MolLogP",
    "LogP-" : "MolLogP",
    "TPSA+" : "TPSA",
    "TPSA-" : "TPSA",
    "HBD+"  : "NumHDonors",
    "HBD-"  : "NumHDonors",
    "HBA+"  : "NumHAcceptors",
    "HBA-"  : "NumHAcceptors",
    "QED+"  : "qed",
    "QED-"  : "qed",
}
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in list(set(task2prop.values()))]
prop2func = {}
for prop, func in prop_pred:
    prop2func[prop] = func
task2func = {k:prop2func[task2prop[k]] for k in task2prop.keys()}

def get_scores_generation(eval_output, path, tokenizer, data, hit_thres=0):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    # save to csv
    df.to_csv(path, index=False)

    preds, labels = df['pred'].values.tolist(), df['label'].values.tolist()
    if "MPP" in data:
        import re
        pattern = r'\b(Yes|No)\b'
        re_preds, re_labels = [], []
        for pred, label in tqdm.tqdm(zip(preds, labels)):
            match = re.search(pattern, pred)
            if match:
                answer = match.group(0)
                re_preds.append(str(answer))
                re_labels.append(str(label))

        re_preds = [1 if "Yes" in x else 0 for x in re_preds]
        re_labels = [1 if "Yes" in x else 0 for x in re_labels]

        # compute accuracy
        acc = metrics.accuracy_score(
            re_labels, re_preds, 
        )
        f1 = metrics.f1_score(
            re_labels, re_preds, 
            average = "macro", 
        )
        return acc, f1
    elif "MModify" in data:
        prop = data.split("/")[-1]
        validities = []
        hits = []
        hits5 = []
        Morgan_sims = []
        if "single" in data:
            for pred, label in tqdm.tqdm(zip(preds, labels)):
                try:
                    mol_pred, mol_label = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(label)
                    validities.append(1)

                    sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_pred, 2), AllChem.GetMorganFingerprint(mol_label, 2))
                    Morgan_sims.append(sim)

                    prop_pred, prop_label = task2func[prop](mol_pred), task2func[prop](mol_label)
                    if prop == "LogP+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "LogP-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "TPSA+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "TPSA-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "HBD+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "HBD-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "HBA+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "HBA-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "QED+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "QED-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                except:
                    validities.append(0)
        if "multi" in data:
            logp, prop = prop[:5], prop[5:]
            for pred, label in tqdm.tqdm(zip(preds, labels)):
                try:
                    mol_pred, mol_label = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(label)
                    validities.append(1)

                    sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_pred, 2), AllChem.GetMorganFingerprint(mol_label, 2))
                    Morgan_sims.append(sim)

                    logp_pred, logp_label, prop_pred, prop_label = task2func[logp](mol_pred), task2func[logp](mol_label), task2func[prop](mol_pred), task2func[prop](mol_label)
                    if logp == "LogP+":
                        if prop == "TPSA+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "TPSA-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBD+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBD-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBA+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBA-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "QED+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "QED-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if logp == "LogP-":
                        if prop == "TPSA+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "TPSA-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBD+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBD-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBA+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBA-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "QED+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "QED-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                except:
                    validities.append(0)
        return 100*sum(hits)/len(validities), 100*sum(hits5)/len(validities), 100*np.mean(Morgan_sims), 100*sum(validities)/len(validities)
    else:
        pass

eval_funcs = {
    'generation': get_scores_generation,
}