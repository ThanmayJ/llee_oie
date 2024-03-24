import torch

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