import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from datasets import load_dataset
import pandas as pd

from tags import POS_TAGS, SYNDP_TAGS, SEMDP_TAGS
from Dataset import Seq2SeqOIE
from Model import T5
from utils import *
from Inference import InferOnCarb

import argparse
import os
import time

def Trainer(args, model, tokenizer, device, optimizer, dataloaders):
    path = os.path.join(args.output_dir, "model_files")
    if not os.path.exists(path):
        os.makedirs(path)

    best_valid_loss = float('inf')
    train_start = time.time()
    print("[Train]")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(args, epoch, tokenizer, model, device, dataloaders["train"], optimizer)
        end_time = time.time()
        valid_loss = validate(args, tokenizer, model, device, dataloaders["validation"])
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"[Epoch {epoch}] Time: {epoch_mins}m {epoch_secs}s | Train Loss {train_loss} | Valid Loss {valid_loss} \n")
        if valid_loss < best_valid_loss:
            print(f"[Saving Model]")
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            best_valid_loss = valid_loss

    train_end = time.time()
    train_mins, train_secs = epoch_time(train_start, train_end)
    print(f"""Total Training Time was {train_mins} m {train_secs} s for {args.epochs} epochs""")

    print("[Test]")
    predictions, actuals = test(args, tokenizer, model, device, test_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(os.path.join(args.output_dir,'predictions.csv'))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="t5-base", type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--prefix', default="info_extract", type=str)
    parser.add_argument('--src_len', default=128, type=int)
    parser.add_argument('--trg_len', default=128, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--output_dir', default="results/", type=str)
    parser.add_argument('--use_wa', default=False, type=bool)
    parser.add_argument('--use_lc', default=False, type=bool)
    parser.add_argument('--use_pos', default=False, type=bool)
    parser.add_argument('--use_syndp', default=False, type=bool)
    parser.add_argument('--use_semdp', default=False, type=bool)
    parser.add_argument('--wt_src', default=1.0, type=float)
    parser.add_argument('--wt_pos', default=0.4, type=float)
    parser.add_argument('--wt_syndp', default=0.4, type=float)
    parser.add_argument('--wt_semdp', default=0.4, type=float)
    parser.add_argument('--dim_pos', default=20, type=int)
    parser.add_argument('--dim_syndp', default=20, type=int)
    parser.add_argument('--dim_semdp', default=20, type=int)
    parser.add_argument('--infer', default=True, type=bool)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset("Thanmay/lsoie_seq2seq")
    model = T5(args).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(args.model)

    for name, param in model.named_parameters():                
        if not param.requires_grad:
            print(name)

    train_set = Seq2SeqOIE(dataset["train"], args.prefix, tokenizer, args.src_len, args.trg_len, "source", "clausie_trg", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)
    valid_set = Seq2SeqOIE(dataset["validation"], args.prefix, tokenizer, args.src_len, args.trg_len, "source", "clausie_trg", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)
    test_set = Seq2SeqOIE(dataset["test"], args.prefix, tokenizer, args.src_len, args.trg_len, "source", "clausie_trg", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    dataloaders = {"train":train_loader, "validation":valid_loader, "test":test_loader}
    
    Trainer(args, model, tokenizer, device, optimizer, dataloaders)

    if args.infer:
        model_path = f"{args.output_dir}/model_files/pytorch_model.bin"

        print(f"""[Model]: Loading {model_path}...\n""")
        # model.load_state_dict(torch.load(model_path))

        dataset = load_dataset("Thanmay/carb_seq2seq")

        InferOnCarb(args, model, tokenizer, device, dataset)

