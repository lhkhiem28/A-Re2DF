### single
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/LogP$2"     --refine "molt-retrieve" --refine_steps 3 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/TPSA$2"     --refine "molt-retrieve" --refine_steps 3 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/HBD$2"      --refine "molt-retrieve" --refine_steps 3 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/HBA$2"      --refine "molt-retrieve" --refine_steps 3 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/QED$2"      --refine "molt-retrieve" --refine_steps 3 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/LogP$2"     --refine "molt-retrieve" --refine_steps 3 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/TPSA$2"     --refine "molt-retrieve" --refine_steps 3 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/HBD$2"      --refine "molt-retrieve" --refine_steps 3 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/HBA$2"      --refine "molt-retrieve" --refine_steps 3 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/single/QED$2"      --refine "molt-retrieve" --refine_steps 3 --hit_thres 1
### multi
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+TPSA$2" --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-TPSA$2" --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+HBD$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-HBD$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+HBA$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-HBA$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+QED$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-QED$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 0
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+TPSA$2" --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-TPSA$2" --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+HBD$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-HBD$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+HBA$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-HBA$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP+QED$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 1
python inference.py --llm_model_name $1 --data "MGen/MModify/ZINC200/multi/LogP-QED$2"  --refine "molt-retrieve" --refine_steps 4 --hit_thres 1