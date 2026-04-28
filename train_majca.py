import sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--finetune", action="store_true", help="Finetune with hard negatives and lower lr")
args = parser.parse_args()

# Resolve before os.chdir so relative paths given by the user still work
if args.resume:
    args.resume = os.path.abspath(args.resume)

TRAIN_DIR      = os.path.join(os.path.dirname(__file__), "unigarmentmanip", "train", "train")
BASE_DIR       = os.path.join(os.path.dirname(__file__), "unigarmentmanip", "train")
MODEL_DIR      = os.path.join(os.path.dirname(__file__), "unigarmentmanip", "train", "model")
DATALOADER_DIR = os.path.join(os.path.dirname(__file__), "unigarmentmanip", "train", "dataloader")

sys.path.insert(0, TRAIN_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, os.path.dirname(__file__))

os.chdir(TRAIN_DIR)

from train_only_cd import train

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "unigarmentmanip", "checkpoints", "majca")

data_dir = os.path.join(PROJECT_ROOT, "data", "majca")
if not os.path.isdir(data_dir) or not any(
    os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)
):
    print(f"ERROR: No training data found at {data_dir}")
    print("Run first:  PYTHON_PATH test/data_gen.py --samples 1000")
    sys.exit(1)

npz_count = sum(
    len([f for f in os.listdir(os.path.join(data_dir, d)) if f.endswith(".npz")])
    for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
)
print(f"Training data: {npz_count} samples in {data_dir}")
print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

resume = args.resume

if resume:
    # If the checkpoint is a raw state dict (not wrapped), wrap it so train() can load it
    import torch as _torch
    from model.pointnet2_Sofa_Model import Sofa_Model as _Sofa
    from base.config import Config as _Config
    _ckpt = _torch.load(resume, map_location="cpu", weights_only=False)
    if "model_state_dict" not in _ckpt:
        _cfg = _Config().train_config
        _model = _Sofa(normal_channel=True, feature_dim=_cfg.feature_dim)
        _model.load_state_dict(_ckpt, strict=False)
        _opt = _torch.optim.Adam(_model.parameters(), lr=_cfg.lr, weight_decay=_cfg.weight_decay)
        _wrapped = {"model_state_dict": _ckpt, "optimizer": _opt}
        resume = os.path.join(CHECKPOINT_DIR, "_pretrained_wrapped.pth")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        _torch.save(_wrapped, resume)
        print(f"Wrapped checkpoint saved to: {resume}")
    print(f"Resuming from: {resume}")
else:
    print("Training from scratch (random init)")

train(CHECKPOINT_DIR, resume_path=resume, finetune=args.finetune)
