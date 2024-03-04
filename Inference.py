import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from datasets import load_dataset
import pandas as pd

from tags import POS_TAGS, SYNDP_TAGS
from Dataset import Seq2SeqOIE
from Model import T5
from utils import *

import argparse
import os
import time

#Puttng it all up and calling the above functions:
def InferOnCarb(args, model, tokenizer, device, dataset):

    test_set = Seq2SeqOIE(dataset["test"], args.prefix, tokenizer, args.src_len, args.trg_len, "source", "target", pos_column="POS", syndp_column="SynDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    outputs, _ = test(args, tokenizer, model, device, test_loader)

    df = outputs
    # convert the array of strings to the required format
    new_arr = []
    test_data = dataset["test"]['source']
    for i, sent in enumerate(df, 1):
        # detect if multiple 3-tuples are present
        if sent.count('(') >= 1:
            # split the sentence into 3-tuples
            sent = sent.split(')')
            # remove any leftover parenthesis
            sent = [s.replace('(', '') for s in sent]
            sent = [s.replace(')', '') for s in sent]
            # split all elements of the 3-tuple into a list
            sent = [s.split(';') for s in sent]
            # add sentence id i to the beginning of each tuple
            # sent = [s.insert(0, i) for s in sent]
            for x in range(len(sent)):
                sent[x].insert(0, test_data[i-1])
                sent[x].insert(1, "1")
            # if sent[i] is longer than 4, merge the additional elements into the last element
            for s in sent:
                if s is not None:
                    if len(s) > 5:
                        s[4:] = [' '.join(s[4:])]
                    while len(s) > 5:
                        s.pop()
                if len(s) > 3:
                    s[3], s[2] = s[2], s[3]
            print(sent)
            sent = [s for s in sent if len(s) > 4] # remove any empty lists
            try: #exception for TANL
                temp = sent[-1]
            except IndexError:
                continue
            if " ".join(temp[2]).strip() == "":
                sent.pop()
        # add the sentence to the new array, if it is a single 3-tuple
        else:
            sent = sent.replace('(', '')
            sent = sent.replace(')', '')
            sent = sent.split(';')
            sent.insert(0, test_data[i-1])
            sent.insert(1, "1")
            # if sent is longer than 5, merge subsequent elements
            if len(sent) > 5:
                sent[4:] = [' '.join(sent[4:])]
                while len(sent) > 5:
                    sent.pop()
            if len(sent) > 3:
                sent[3], sent[2] = sent[2], sent[3]
            sent = [sent]
        # add the sentence to the new array
        sent = [s for s in sent if len(s) > 4]
        if sent != None:
            new_arr.extend(sent)
    
    print(len(new_arr), len(new_arr[0]))
    for i in range(len(new_arr)):
        if  new_arr[i] is None: 
            print("Null: ", i, new_arr[i])
        elif len(new_arr[i]) != 5:
            print("Not 5, but ",len(new_arr[i]),i, new_arr[i])
        
    # drop None rows from the array
    new_arr = [x for x in new_arr if x is not None]
    # convert the array to a dataframe
    df = pd.DataFrame(new_arr, columns=['sent', 'prob', 'predicate', 'subject', 'object'])
    # write the dataframe to a tsv file
    df.to_csv(f"{args.output_dir}/carb.tsv", sep='\t', index=False, header=False)
        
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