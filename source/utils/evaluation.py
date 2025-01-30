import tqdm
import selfies as sf
import numpy as np
import pandas as pd
from sklearn import metrics
from Levenshtein import distance
from nltk.translate.bleu_score import corpus_bleu
from fcd_torch import FCD

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys

fg2smart = {
    'Alkyl': '[CX4]',
    'Alkenyl': '[$([CX3]=[CX3])]',
    'Alkynyl': '[$([CX2]#C)]',
    'Phenyl': 'c',
    'bromoalkane': '[Br]',
    'chloro': '[Cl]',
    'fluoro': '[F]',
    'halo': '[#6][F,Cl,Br,I]',
    'iodo': '[I]',
    'Acetal': 'O[CH1][OX2H0]',
    'Haloformyl': '[CX3](=[OX1])[F,Cl,Br,I]',
    'Hydroxyl': '[#6][OX2H]',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'CarbonateEster': '[CX3](=[OX1])(O)O',
    'Carboxylate': '[CX3](=O)[O-]',
    'Carboxyl': '[CX3](=O)[OX2H1]',
    'Carboalkoxy': '[CX3](=O)[OX2H0]',
    'Ether': '[OD2]',
    'Hemiacetal': 'O[CH1][OX2H1]',
    'Hemiketal': 'OC[OX2H1]',
    'Methylenedioxy': 'C([OX2])([OX2])',
    'Hydroperoxy': 'O[OX2H]',
    'Ketal': 'OC[OX2H0]',
    'Carbonyl': '[CX3]=[OX1]',
    'CarboxylicAnhydride': '[CX3](=O)[OX2H0][CX3](=O)',
    'OrthocarbonateEster': 'C([OX2])([OX2])([OX2])([OX2])',
    'Orthoester': 'C([OX2])([OX2])([OX2])',
    'Peroxy': 'O[OX2H0]',
    'Carboxamide': '[NX3][CX3](=[OX1])[#6]',
    'Amidine': '[NX3][CX3]=[NX2]',
    '4ammoniumIon': '[NX4+]',
    'PrimaryAmine': '[NX3;H2,H1;!$(NC=O)]',
    'SecondaryAmine': '[NX3;H1;!$(NC=O)]',
    'TertiaryAmine': '[NX3;!$(NC=O)]',
    'Azide': '[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]',
    'Azo': '[NX2]=N',
    'Carbamate': '[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]',
    'Cyanate': 'OC#N',
    'Isocyanate': '[O]=[CX2]=[NX2]',
    'Imide': '[CX3](=[OX1])[NX3H][CX3](=[OX1])',
    'PrimaryAldimine': '[CX3H1]=[NX2H1]',
    'PrimaryKetimine': '[CX3]=[NX2H1]',
    'SecondaryAldimine': '[CX3H1]=[NX2H0]',
    'SecondaryKetimine': '[CX3]=[NX2H0]',
    'Nitrate': '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]',
    'Isonitrile': '[CX1-]#[NX2+]',
    'Nitrile': '[NX1]#[CX2]',
    'Nitrosooxy': 'O[NX2]=[OX1]',
    'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    'Nitroso': '[NX2]=[OX1]',
    'Oxime': 'C=N[OX2H1]',
    'Pyridyl': 'ccccnc',
    'Disulfide': '[#16X2H0]S',
    'CarbodithioicAcid': '[#16X2H1]C=[#16]',
    'Carbodithio': '[#16X2H0]C=[#16]',
    'Sulfide': '[#16X2H0]',
    'Sulfino': '[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]',
    'Sulfoate': '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]',
    'Sulfonyl': '[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]',
    'Sulfo': '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]',
    'Sulfinyl': '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]',
    'Thial': '[#16]=[CX3H1]',
    'CarbothioicOAcid': '[OX2H1]C=[#16]',
    'CarbothioicSAcid': '[#16X2H1]C=O',
    'Isothiocyanate': '[#16]=[CX2]=[NX2]',
    'Thiocyanate': '[#16]C#N',
    'Thiolester': '[#16X2H0]C=O',
    'Thionoester': '[OX2H0]C=[#16]',
    'Thioketone': '[#16]=[CX3H0]',
    'Sulfhydryl': '[#16X2H]',
    'Phosphate': '[OX2H0][PX4](=[OX1])([OX2H1])([OX2H1])',
    'Phosphino': '[PX3]',
    'Phosphodiester': '[OX2H1][PX4](=[OX1])([OX2H0])([OX2H0])',
    'Phosphono': '[PX4](=[OX1])([OX2H1])([OX2H1])',
    'Borino': '[BX3]([OX2H1])',
    'Borinate': '[BX3]([OX2H0])',
    'Borono': '[BX3]([OX2H1])([OX2H1])',
    'Boronate': '[BX3]([OX2H0])([OX2H0])',
    'Alkylaluminium': '[#13].[#13]',
    'Alkyllithium': '[#3]',
    'AlkylmagnesiumHalide': '[#12X2][F,Cl,Br,I]',
    'SilylEther': '[#14X4][OX2]'
}
fg2smart = {fg:Chem.MolFromSmarts(fg2smart[fg]) for fg in fg2smart.keys()}
def count_fgs(mol, fg2smart=fg2smart):
    fg2count = {}
    for name, smart in fg2smart.items():
        matches = mol.GetSubstructMatches(smart)
        if len(matches) > 0:
            fg2count[name] = len(matches)
    return fg2count
def check_funchit(fgs_pred, fgs_label):
    hit = 0
    str_pred, str_label = [], []
    for fg, count in fgs_label.items():
        if fg not in fgs_pred:
            fgs_pred[fg] = 0
        if fgs_pred[fg] > 0:
            if fgs_pred[fg] == 1:
                str_pred.append(f"{fgs_pred[fg]} {fg} group")
            else:
                str_pred.append(f"{fgs_pred[fg]} {fg} groups")
        if count == 1:
            str_label.append(f"{count} {fg} group")
        else:
            str_label.append(f"{count} {fg} groups")

        if fgs_pred[fg] == count:
            hit += 1
    return hit == len(fgs_label), hit, ", ".join(str_pred), ", ".join(str_label)

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

def get_scores_generation(eval_output, path, tokenizer, data, selfies=False, hit_thres=0):
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
        hits4 = []
        hits5 = []
        Morgan_sims = []
        valid_preds, valid_labels = [], []
        if "single" in data:
            for pred, label in tqdm.tqdm(zip(preds, labels)):
                try:
                    if selfies:
                        pred, label = sf.decoder(pred), sf.decoder(label)
                    mol_pred, mol_label = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(label)
                    pred, label = Chem.MolToSmiles(mol_pred, isomericSmiles=False, canonical=True), Chem.MolToSmiles(mol_label, isomericSmiles=False, canonical=True)
                    valid_preds.append(pred), valid_labels.append(label)
                    inchi_pred, inchi_label = Chem.MolToInchi(mol_pred), Chem.MolToInchi(mol_label)
                    validities.append(1)

                    sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_pred, 2), AllChem.GetMorganFingerprint(mol_label, 2))
                    Morgan_sims.append(sim)

                    prop_pred, prop_label = task2func[prop](mol_pred), task2func[prop](mol_label)
                    if prop == "LogP+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits4.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "LogP-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits4.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "TPSA+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits4.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "TPSA-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits4.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "HBD+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits4.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "HBD-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits4.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "HBA+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits4.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "HBA-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits4.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if prop == "QED+":
                        hits.append(prop_pred > prop_label + task2thres[prop][hit_thres][0])
                        hits4.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                        hits5.append(prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                    if prop == "QED-":
                        hits.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                        hits4.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                        hits5.append(prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                except:
                    validities.append(0)
        if "multi" in data:
            logp, prop = prop[:5], prop[5:]
            for pred, label in tqdm.tqdm(zip(preds, labels)):
                try:
                    if selfies:
                        pred, label = sf.decoder(pred), sf.decoder(label)
                    mol_pred, mol_label = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(label)
                    pred, label = Chem.MolToSmiles(mol_pred, isomericSmiles=False, canonical=True), Chem.MolToSmiles(mol_label, isomericSmiles=False, canonical=True)
                    valid_preds.append(pred), valid_labels.append(label)
                    inchi_pred, inchi_label = Chem.MolToInchi(mol_pred), Chem.MolToInchi(mol_label)
                    validities.append(1)

                    sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_pred, 2), AllChem.GetMorganFingerprint(mol_label, 2))
                    Morgan_sims.append(sim)

                    logp_pred, logp_label, prop_pred, prop_label = task2func[logp](mol_pred), task2func[logp](mol_label), task2func[prop](mol_pred), task2func[prop](mol_label)
                    if logp == "LogP+":
                        if prop == "TPSA+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "TPSA-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBD+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBD-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBA+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBA-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "QED+":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "QED-":
                            hits.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred > logp_label + task2thres[logp][hit_thres][0] and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                    if logp == "LogP-":
                        if prop == "TPSA+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "TPSA-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBD+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBD-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "HBA+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "HBA-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                        if prop == "QED+":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0])
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred > prop_label + task2thres[prop][hit_thres][0] and sim >= 0.5)
                        if prop == "QED-":
                            hits.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label)
                            hits4.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.4)
                            hits5.append(logp_pred + task2thres[logp][hit_thres][0] < logp_label and prop_pred + task2thres[prop][hit_thres][0] < prop_label and sim >= 0.5)
                except:
                    validities.append(0)
        return 100*sum(hits)/len(validities), 100*sum(hits4)/len(validities), 100*sum(hits5)/len(validities), 100*np.mean(Morgan_sims), 100*sum(validities)/len(validities)
    else:
        pass

eval_funcs = {
    'generation': get_scores_generation,
}