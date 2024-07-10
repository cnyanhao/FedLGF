# Local and Global Flatness for Federated Domain Generalization (ECCV 2024)

## Installation
```bash
conda create -n fdg python=3.10 -y
conda activate fdg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gdown
```

## Training
Download datasets
```
python domainbed/download.py --data_dir=./data --dataset RotatedMNIST
python domainbed/download.py --data_dir=./data --dataset VLCS
python domainbed/download.py --data_dir=./data --dataset PACS
python domainbed/download.py --data_dir=./data --dataset OfficeHome
```

Running FedLGF
```
python main.py --dataset PACS --algorithm FedSVD --num_clients 50 --seed 1 --output_dir train_output --skip_model_save --train_index seed_1 \
        --hparams '{"perturb_type": "singular", "perturb_dist": "normal", "local_smooth": 0.1, "global_smooth": 0.1, "perturb_init_scale": 10, "perturb_grad_scale": 0.1}'
```

## Reference Repo

- https://github.com/YamingGuo98/FedIIR
