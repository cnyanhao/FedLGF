# python domainbed/download.py --data_dir=./data --dataset RotatedMNIST
# python domainbed/download.py --data_dir=./data --dataset VLCS
# python domainbed/download.py --data_dir=./data --dataset PACS
# python domainbed/download.py --data_dir=./data --dataset OfficeHome

# python main.py --dataset RotatedMNIST --algorithm FedAvg --num_clients 50 --output_dir train_output --skip_model_save
# python main.py --dataset VLCS --algorithm FedAvg --num_clients 50 --output_dir train_output --skip_model_save
# python main.py --dataset PACS --algorithm FedAvg --num_clients 50 --output_dir train_output --skip_model_save
# python main.py --dataset OfficeHome --algorithm FedAvg --num_clients 50 --output_dir train_output --skip_model_save

python main.py --dataset PACS --algorithm FedSVD --num_clients 50 --seed 1 --output_dir train_output --skip_model_save --train_index seed_1 \
        --hparams '{"perturb_type": "singular", "perturb_dist": "normal", "local_smooth": 0.1, "global_smooth": 0.1, "perturb_init_scale": 10, "perturb_grad_scale": 0.1}'

python main.py --dataset PACS --algorithm FedSVD --num_clients 50 --seed 2 --output_dir train_output --skip_model_save --train_index seed_2 \
        --hparams '{"perturb_type": "singular", "perturb_dist": "normal", "local_smooth": 0.1, "global_smooth": 0.1, "perturb_init_scale": 10, "perturb_grad_scale": 0.1}'

python main.py --dataset PACS --algorithm FedSVD --num_clients 3 --seed 1 --output_dir train_output --skip_model_save --train_index seed_1 \
        --hparams '{"perturb_type": "singular", "perturb_dist": "normal", "local_smooth": 0.1, "global_smooth": 0.1, "perturb_init_scale": 10, "perturb_grad_scale": 0.1}'

python main.py --dataset PACS --algorithm FedSVD --num_clients 3 --seed 2 --output_dir train_output --skip_model_save --train_index seed_2 \
        --hparams '{"perturb_type": "singular", "perturb_dist": "normal", "local_smooth": 0.1, "global_smooth": 0.1, "perturb_init_scale": 10, "perturb_grad_scale": 0.1}'
