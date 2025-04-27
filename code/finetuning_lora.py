import torch
from torch.utils.data import DataLoader
from model import SBERT
from trainer import SBERTFineTuner
from dataset import FinetuneDataset
import numpy as np
import random
import argparse
from peft import LoraConfig, get_peft_model  # NEW: LoRA imports

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

def Config():
    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--pretrain_path", type=str, required=False)
    parser.add_argument("--finetune_path", type=str, required=False)
    parser.add_argument("--valid_rate", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_features", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    # NEW: LoRA arguments
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()

def add_lora(model, config):
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "linear_layers.0",  # Query
            "linear_layers.1",  # Key
            "linear_layers.2",  # Value
            "output_linear",    # Output
            "w_1",              # FFN layer 1
            "w_2"               # FFN layer 2
        ],
        lora_dropout=config.lora_dropout,
        bias="none"
    )
    return get_peft_model(model, lora_config)

if __name__ == "__main__":
    config = Config()

    # Data loading (unchanged)
    train_file = config.file_path + 'Train_cleaned.csv'
    valid_file = config.file_path + 'Validate_cleaned.csv'
    test_file = config.file_path + 'Test_cleaned.csv'

    print("Loading Data sets...")
    train_dataset = FinetuneDataset(train_file, config.num_features, config.max_length)
    valid_dataset = FinetuneDataset(valid_file, config.num_features, config.max_length)
    test_dataset = FinetuneDataset(test_file, config.num_features, config.max_length)
    print(f"training samples: {train_dataset.TS_num}, validation samples: {valid_dataset.TS_num}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Model initialization with LoRA
    print("Initialing SITS-BERT...")
    model = SBERT(
        config.num_features,
        hidden=config.hidden_size,
        n_layers=config.layers,
        attn_heads=config.attn_heads,
        dropout=config.dropout
    )

    if config.pretrain_path:
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(config.pretrain_path + "checkpoint.bert.pth"))

    if config.use_lora:  # NEW: LoRA integration
        print("Applying LoRA...")
        model = add_lora(model, config)
        model.print_trainable_parameters()

    # Training (unchanged)
    trainer = SBERTFineTuner(
        model,
        config.num_classes,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader
    )

    print("Fine-tuning...")
    best_accuracy = 0
    for epoch in range(config.epochs):
        train_oa, _, valid_oa, _ = trainer.train(epoch)
        if valid_oa > best_accuracy:
            best_accuracy = valid_oa
            trainer.save(epoch, config.finetune_path)

    # Testing (unchanged)
    print("\nTesting...")
    trainer.load(config.finetune_path)
    oa, kappa, aa, _ = trainer.test(test_loader)
    print(f'Test OA: {oa:.2f}, Kappa: {kappa:.3f}, AA: {aa:.3f}')
