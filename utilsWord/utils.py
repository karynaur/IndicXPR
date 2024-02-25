import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utilsWord.tools import seed_everything, AverageMeter
from torch.cuda.amp import GradScaler, autocast
import math

def test_model_single_encoder(args, model, val_loader, device):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.inference_mode():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples,first_trg_examples = None,None
        second_src_examples,second_trg_examples = None,None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            first_src_examples = first_src if first_src_examples is None else torch.cat([first_src_examples,first_src],dim=0)
            first_trg_examples = first_trg if first_trg_examples is None else torch.cat([first_trg_examples,first_trg],dim=0)
            second_src_examples = second_src if second_src_examples is None else torch.cat([second_src_examples,second_src],dim=0)
            second_trg_examples = second_trg if second_trg_examples is None else torch.cat([second_trg_examples,second_trg],dim=0)
        
        first_src_examples = torch.nn.functional.normalize(first_src_examples,dim=1)
        first_trg_examples = torch.nn.functional.normalize(first_trg_examples,dim=1)
        second_src_examples = torch.nn.functional.normalize(second_src_examples,dim=1)
        second_trg_examples = torch.nn.functional.normalize(second_trg_examples,dim=1)
        first_st_sim_matrix = F.softmax(torch.mm(first_src_examples,first_trg_examples.T)/math.sqrt(first_src_examples.size(-1))/0.1,dim=1)
        second_st_sim_matrix = F.softmax(torch.mm(second_trg_examples,second_src_examples.T)/math.sqrt(second_trg_examples.size(-1))/0.1,dim=1)
        label = torch.LongTensor(list(range(first_st_sim_matrix.size(0)))).to(first_src_examples.device)
        st_acc = torch.argmax(first_st_sim_matrix, dim=1)    # [B]
        ts_acc = torch.argmax(second_st_sim_matrix, dim=1)
        all_predictions.extend(st_acc.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())
        all_predictions.extend(ts_acc.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())

        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        acc = accuracy_score(all_labels, all_predictions)
    return acc, precision, recall, f1


def test_model_dual_encoder(args, model, val_loader, device):  # Dual Encoder
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples,first_trg_examples = None,None
        second_src_examples,second_trg_examples = None,None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_k(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.module.encoder_k(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.encoder_k(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.encoder_k(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            first_src_examples = first_src if first_src_examples is None else torch.cat([first_src_examples,first_src],dim=0)
            first_trg_examples = first_trg if first_trg_examples is None else torch.cat([first_trg_examples,first_trg],dim=0)
            second_src_examples = second_src if second_src_examples is None else torch.cat([second_src_examples,second_src],dim=0)
            second_trg_examples = second_trg if second_trg_examples is None else torch.cat([second_trg_examples,second_trg],dim=0)
        first_src_examples = torch.nn.functional.normalize(first_src_examples,dim=1)
        first_trg_examples = torch.nn.functional.normalize(first_trg_examples,dim=1)
        second_src_examples = torch.nn.functional.normalize(second_src_examples,dim=1)
        second_trg_examples = torch.nn.functional.normalize(second_trg_examples,dim=1)
        first_st_sim_matrix = F.softmax(torch.mm(first_src_examples,first_trg_examples.T)/math.sqrt(first_src_examples.size(-1))/0.1,dim=1)
        second_st_sim_matrix = F.softmax(torch.mm(second_trg_examples,second_src_examples.T)/math.sqrt(second_trg_examples.size(-1))/0.1,dim=1)
        label = torch.LongTensor(list(range(first_st_sim_matrix.size(0)))).to(first_src_examples.device)
        st_acc = torch.argmax(first_st_sim_matrix, dim=1)    # [B]
        ts_acc = torch.argmax(second_st_sim_matrix, dim=1)
        all_predictions.extend(st_acc.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())
        all_predictions.extend(ts_acc.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())

        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        acc = accuracy_score(all_labels, all_predictions)
    return acc, precision, recall, f1


def train_model(args, model, train_loader, lossFunc, optimizer, scheduler, device):  # Train an epoch
    scaler = GradScaler()
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    clips = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, batch in enumerate(tk):
        batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
        batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
        with autocast():
            output0, output1 = model(batch_src,batch_trg)
            loss1, acc1 = lossFunc(output0, output1)
            output0, output1 = model(batch_trg,batch_src)
            loss2, acc2 = lossFunc(output0, output1)
        loss = loss1 + loss2
        with open('./ano_record.txt','a+')  as f:
            f.write("STEP : " + str(step) + '\n')
            f.write(str(loss1) + " | "+  str(loss2) + '\n')
        acc = (acc1 + acc2)/2
        loss = loss / 2
        input_ids = batch_src[0]
        scaler.scale(loss).backward()

        losses.update(loss.item(), input_ids.size(0))
        accs.update(acc, input_ids.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

        if ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader)): 
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    return losses.avg, accs.avg