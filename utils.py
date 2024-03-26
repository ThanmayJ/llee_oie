import torch
import pandas as pd

def train(args, epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    epoch_loss = 0
    for _,data in enumerate(loader, 0):
        optimizer.zero_grad()
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        src_ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        
        pos_ids = data['pos_ids'].to(device, dtype = torch.long) #if args.use_pos else None
        syndp_ids = data['syndp_ids'].to(device, dtype = torch.long) #if args.use_syndp else None
        semdp_ids = data['semdp_ids'].to(device, dtype = torch.long) #if args.use_semdp else None

        outputs = model(src_ids=src_ids, pos_ids=pos_ids, syndp_ids=syndp_ids, semdp_ids=semdp_ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if _%10==0:
            print(f"Epoch {epoch} | Step {_} | Loss {loss}")

        loss.backward()    
        optimizer.step()
        epoch_loss+=loss
    
    return epoch_loss/len(loader)

def validate(args, tokenizer, model, device, loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            src_ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            
            pos_ids = data['pos_ids'].to(device, dtype = torch.long) if args.use_pos else None
            syndp_ids = data['syndp_ids'].to(device, dtype = torch.long) if args.use_syndp else None
            semdp_ids = data['semdp_ids'].to(device, dtype = torch.long) if args.use_semdp else None

            outputs = model(src_ids=src_ids, pos_ids=pos_ids, syndp_ids=syndp_ids, semdp_ids=semdp_ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            if _%10==0:
                print(f"Validation Completed {_}")
            
            epoch_loss+=loss
    
    return epoch_loss/len(loader)

def test(args, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            
            y = data['target_ids'].to(device, dtype = torch.long)
            src_ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            
            pos_ids = data['pos_ids'].to(device, dtype = torch.long) if args.use_pos else None
            syndp_ids = data['syndp_ids'].to(device, dtype = torch.long) if args.use_syndp else None
            semdp_ids = data['semdp_ids'].to(device, dtype = torch.long) if args.use_semdp else None

            generated_ids = model.generate(src_ids=src_ids, pos_ids=pos_ids, syndp_ids=syndp_ids, semdp_ids=semdp_ids, attention_mask=mask)
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%10==0:
                print(f"Test Completed {_}")

            predictions.extend(preds)
            actuals.extend(target)
            # print(predictions)
            # print(actuals)
    return predictions, actuals

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_predictions_to_carb_format(df):
    # convert the array of strings to the required format
    new_arr = []
    test_data = df["Input Text"].tolist()
    predictions = df["Generated Text"].tolist()
    for i, sent in enumerate(predictions, 1):
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
            # print(sent)
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
    output_df = pd.DataFrame(new_arr, columns=['sent', 'prob', 'predicate', 'subject', 'object'])
    # write the dataframe to a tsv file
    output_df.to_csv(f"{args.output_dir}/carb.tsv", sep='\t', index=False, header=False)    
