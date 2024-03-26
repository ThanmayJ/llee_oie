import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from datasets import load_dataset
import pandas as pd

from tags import POS_TAGS, SYNDP_TAGS, SEMDP_TAGS
from Dataset import Seq2SeqOIE
from Model import T5
from utils import *

import argparse
import os
import time

def InferOnCarb(args, model, tokenizer, device, dataset):

    test_set = Seq2SeqOIE(dataset["test"], args.prefix, tokenizer, args.src_len, args.trg_len, "source", "target", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    predictions, _ = test(args, tokenizer, model, device, test_loader)
    final_df = pd.DataFrame({'Input Text':dataset["test"], 'Generated Text':predictions})
    final_df.to_csv(os.path.join(args.output_dir,'carb_predictions.csv'))
    convert_predictions_to_carb_format(final_df)
    
        
if __name__ == "__main__":  
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

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    model_path = f"{args.output_dir}/model_files/pytorch_model.bin"

    print(f"""[Model]: Loading {model_path}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5TokenizerFast.from_pretrained(args.model)

    model = T5(args).to(device)
    # model.load_state_dict(torch.load(model_path))

    dataset = load_dataset("Thanmay/carb_seq2seq")

    InferOnCarb(args, model, tokenizer, device, dataset)
