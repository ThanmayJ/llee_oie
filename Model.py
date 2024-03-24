import os
import argparse

import torch
from torch import nn

from transformers import T5ForConditionalGeneration

from tags import POS_TAGS, SYNDP_TAGS, SEMDP_TAGS

class T5(nn.Module):
    def __init__(self, args):
        super(T5, self).__init__()
        self.args = args

        self.t5 = T5ForConditionalGeneration.from_pretrained(args.model)
        self.emb_dim = self.t5.encoder.embed_tokens.weight.shape[-1]

        self.num_special_tags = 3 # Refer Dataset.py, we have three additional tags - <pad>, <unk> and <eos>. This is added in the code below.      
        
        if self.args.use_wa:
            self.get_combined_embeds = self.get_combined_embeds_wa
            self.wt_src = args.wt_src  
            if self.args.use_pos:
                self.wt_pos = self.args.wt_pos
                self.embed_pos = nn.Embedding(len(POS_TAGS)+self.num_special_tags, self.emb_dim)
                nn.init.xavier_uniform_(self.embed_pos.weight.data)
            if self.args.use_syndp:
                self.wt_syndp = self.args.wt_syndp
                self.embed_syndp = nn.Embedding(len(SYNDP_TAGS)+self.num_special_tags, self.emb_dim)
                nn.init.xavier_uniform_(self.embed_syndp.weight.data)
            if self.args.use_semdp:
                self.wt_semdp = self.args.wt_semdp
                self.embed_semdp = nn.Embedding(len(SEMDP_TAGS)+self.num_special_tags, self.emb_dim)
                nn.init.xavier_uniform_(self.embed_semdp.weight.data)
       	
        elif self.args.use_lc:
            self.get_combined_embeds = self.get_combined_embeds_lc
            self.dim_afterconcat = self.emb_dim
            if self.args.use_pos:
                self.dim_pos = self.args.dim_pos
                self.embed_pos = nn.Embedding(len(POS_TAGS)+self.num_special_tags, self.dim_pos)
                nn.init.xavier_uniform_(self.embed_pos.weight.data)
                self.dim_afterconcat += self.dim_pos
            if self.args.use_syndp:
                self.dim_syndp = self.args.dim_syndp
                self.embed_syndp = nn.Embedding(len(SYNDP_TAGS)+self.num_special_tags, self.dim_syndp)
                nn.init.xavier_uniform_(self.embed_syndp.weight.data)
                self.dim_afterconcat += self.dim_syndp
            if self.args.use_semdp:
                self.dim_semdp = self.args.dim_semdp
                self.embed_semdp = nn.Embedding(len(SEMDP_TAGS)+self.num_special_tags, self.dim_semdp)
                nn.init.xavier_uniform_(self.embed_semdp.weight.data)
                self.dim_afterconcat += self.dim_semdp
            self.linear = nn.Linear(self.dim_afterconcat, self.emb_dim)
            nn.init.xavier_uniform_(self.linear.weight.data)
        
        else:
            self.get_combined_embeds = self.get_vanilla_embeds
    
    def get_combined_embeds_lc(self, src_ids, pos_ids=None, syndp_ids=None, semdp_ids=None):
        src_embeds = self.t5.encoder.embed_tokens(src_ids)
        combined_embeds = src_embeds

        if self.args.use_pos:
            pos_embeds = self.embed_pos(pos_ids)
            combined_embeds = torch.cat((combined_embeds, pos_embeds), dim=2)
        if self.args.use_syndp:
            syndp_embeds = self.embed_syndp(syndp_ids)
            combined_embeds = torch.cat((combined_embeds, syndp_embeds), dim=2)
        if self.args.use_semdp: 
            semdp_embeds = self.embed_semdp(semdp_ids)
            combined_embeds = torch.cat((combined_embeds, semdp_embeds), dim=2)
        
        combined_embeds = self.linear(combined_embeds)
        return combined_embeds
    
    def get_combined_embeds_wa(self, src_ids, pos_ids=None, syndp_ids=None, semdp_ids=None):
        src_embeds = self.t5.encoder.embed_tokens(src_ids)

        if self.args.use_pos:
            pos_embeds = self.embed_pos(pos_ids)
            combined_embeds += self.wt_pos * pos_embeds
        if self.args.use_syndp:
            syndp_embeds = self.embed_syndp(syndp_ids)
            combined_embeds += self.wt_syndp * syndp_embeds
        if self.args.use_semdp:
            semdp_embeds = self.embed_semdp(semdp_ids)
            combined_embeds += self.wt_semdp * semdp_embeds

        return combined_embeds

    def get_vanilla_embeds(self, src_ids, pos_ids=None, syndp_ids=None, semdp_ids=None):
        src_embeds = self.t5.encoder.embed_tokens(src_ids)
        return src_embeds
    
    def forward(self, src_ids, attention_mask, decoder_input_ids, labels, pos_ids=None, syndp_ids=None, semdp_ids=None):
        combined_embeds = self.get_combined_embeds(src_ids, pos_ids, syndp_ids, semdp_ids)
        outputs = self.t5(inputs_embeds = combined_embeds, attention_mask = attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        return outputs
    
    def generate(self, src_ids, attention_mask, pos_ids=None, syndp_ids=None, semdp_ids=None):
        combined_embeds = self.get_combined_embeds(src_ids, pos_ids, syndp_ids, semdp_ids)
        
        gen = self.t5.generate(
              inputs_embeds = combined_embeds,
              attention_mask = attention_mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
        
        return gen
    
    def save_pretrained(self, output_path):
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="t5-base", type=str)
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

    model = T5(args)
    print(model)
    print(len(POS_TAGS),len(SYNDP_TAGS), len(SEMDP_TAGS))