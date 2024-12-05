srun --partition=gpu --nodes=1 --gres=gpu:v100-sxm2:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00 python3 -u wayfaster/train.py --cfg_file configs/temporal_model.yaml

module load cuda/12.1
source activate vml2
srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
gdb --args python3 -u wayfaster/train.py --cfg_file configs/temporal_model.yaml

Test:

import torch
torch.cuda.is_available()
x = torch.tensor([1.0, 2.0]).cuda()

Use Python 3.12.5

conda create --name vml -c conda-forge python=3.12.5 -y
conda env remove -n <envname>

Notes:

BEFORE INSTALLING ANYTHING CHECK WHICH PIP
conda install pip (if necessary)





Lesson learned:

You need to download torch that matches the same CUDA version you used on `module load cuda/12.1`. Check the requirement.txt (Using torch from cuda11.8 will cause segmentation fault)