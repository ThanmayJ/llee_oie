import torch
from torch.utils.data import Dataset
from transformers import T5TokenizerFast

from tags import POS_TAGS, SYNDP_TAGS, SEMDP_TAGS
from datasets import load_dataset

class Seq2SeqOIE(Dataset):

    def __init__(self, data, prefix, tokenizer, source_len, target_len, source_column, target_column, pos_column=None, syndp_column=None, semdp_column=None, pos_tags=None, syndp_tags=None, semdp_tags=None):
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.target_len = target_len
        
        self.prefix = prefix
        self.source_text = self.data[source_column]
        self.target_text = self.data[target_column]
        
        # POS Tags
        if pos_column is not None:
            self.source_pos = self.data[pos_column]
            pos_tags = [self.tokenizer.pad_token] + pos_tags + [self.tokenizer.eos_token, self.tokenizer.unk_token]
            self.pos_tag2idx = dict(zip(pos_tags, range(len(pos_tags))))
      
        # SynDP Tags
        if syndp_column is not None:
            self.source_syndp = self.data[syndp_column]
            syndp_tags = [self.tokenizer.pad_token] + syndp_tags + [self.tokenizer.eos_token, self.tokenizer.unk_token]
            self.syndp_tag2idx = dict(zip(syndp_tags, range(len(syndp_tags))))

        # SemDP Tags
        if semdp_column is not None:
            self.source_semdp = self.data[semdp_column]
            semdp_tags = [self.tokenizer.pad_token] + semdp_tags + [self.tokenizer.eos_token, self.tokenizer.unk_token]
            self.semdp_tag2idx = dict(zip(semdp_tags, range(len(semdp_tags))))            

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        source_pos = self.source_pos[index]
        source_syndp = self.source_syndp[index]
        source_semdp = self.source_semdp[index]

        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        prepended_text = self.prefix
        source_text = prepended_text + ": " + source_text

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.target_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()


        # Source Text Tokens
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
            
        # Source POS Tokens
        prefix_words = self.prefix.split() + [":"] 
        source = self.tokenizer.batch_encode_plus([prefix_words + source_pos["words"]], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt', is_split_into_words=True)        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
        pos_tokens = [self.tokenizer.pad_token if i in [None, *list(range(len(prefix_words)))] else source_pos["tags"][i-len(prefix_words)] for i in source.word_ids()]
        pos_ids = [self.pos_tag2idx[tok] if tok in self.pos_tag2idx.keys() else self.tokenizer.unk_token_id for tok in pos_tokens]

        # Source SynDP Tokens
        prefix_words = self.prefix.split() + [":"] 
        source = self.tokenizer.batch_encode_plus([prefix_words + source_syndp["words"]], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt', is_split_into_words=True)        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
        syndp_tokens = [self.tokenizer.pad_token if i in [None, *list(range(len(prefix_words)))] else source_syndp["tags"][i-len(prefix_words)] for i in source.word_ids()]
        syndp_ids = [self.syndp_tag2idx[tok] if tok in self.syndp_tag2idx.keys() else self.tokenizer.unk_token_id for tok in syndp_tokens]

        # Source SemDP Tokens
        prefix_words = self.prefix.split() + [":"] 
        source = self.tokenizer.batch_encode_plus([prefix_words + source_semdp["words"]], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt', is_split_into_words=True)        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
        semdp_tokens = [self.tokenizer.pad_token if i in [None, *list(range(len(prefix_words)))] else source_semdp["tags"][i-len(prefix_words)] for i in source.word_ids()]
        semdp_ids = [self.semdp_tag2idx[tok] if tok in self.semdp_tag2idx.keys() else self.tokenizer.unk_token_id for tok in semdp_tokens]


        # Target Text Tokens
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        target_tokens = self.tokenizer.convert_ids_to_tokens(target_ids)

        return {
            'source_text': source_text,
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long),
            'source_tokens': source_tokens, 
            'target_text': target_text,
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'target_tokens': target_tokens,
            'pos_tokens': pos_tokens,
            'pos_ids': torch.tensor(pos_ids).to(dtype=torch.long),
            'syndp_tokens': syndp_tokens,
            'syndp_ids': torch.tensor(syndp_ids).to(dtype=torch.long),
            'semdp_tokens': semdp_tokens,
            'semdp_ids': torch.tensor(semdp_ids).to(dtype=torch.long)
        }
    

if __name__ == "__main__":
    dataset = load_dataset("Thanmay/lsoie_seq2seq")
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")

    train_set = Seq2SeqOIE(dataset["train"], "info_extract", tokenizer, 128, 128, "source", "target", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)
    valid_set = Seq2SeqOIE(dataset["validation"], "info_extract", tokenizer, 128, 128, "source", "target", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)
    test_set = Seq2SeqOIE(dataset["test"], "info_extract ", tokenizer, 128, 128, "source", "target", pos_column="POS", syndp_column="SynDP", semdp_column="SemDP", pos_tags=POS_TAGS, syndp_tags=SYNDP_TAGS, semdp_tags=SEMDP_TAGS)

    print(train_set[0]["source_tokens"])
    print()
    print(train_set[0]["pos_tokens"])
    print()
    print(train_set[0]["syndp_tokens"])
    print()
    print(train_set[0]["semdp_tokens"])






