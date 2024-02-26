import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import math
import os
from sklearn.model_selection import *
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, XLMRobertaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from DictMatching.Loss import MoCoLoss
from DictMatching.moco import MoCo
from utilsWord.args import getArgs
from utilsWord.tools import seed_everything, AverageMeter
from utilsWord.sentence_process import load_words_mapping, WordWithContextDatasetWW, load_word2context_from_tsv
from utilsWord.utils import test_model_single_encoder, train_model

args = getArgs()
# from new acc
seed_everything(args.seed)  

if args.distributed:
    args.distributed = False

if args.distributed:
    dist.init_process_group(backend='nccl')

def main():
    base = args.data
    args.train_phrase_path = base + "train/train-en-" + args.lg + "-" + args.sn + "-phrase.txt"
    args.dev_phrase_path = base + "dev/dev-en-" + args.lg + "-" + args.sn + "-phrase.txt"
    args.test_phrase_path = base + "test/test-en-" + args.lg + "-" + args.sn + "-phrase.txt"
    args.src_context_path = base + "sentences/en-" + args.lg + "-phrase-sentences." + args.sn + ".tsv"
    args.trg_context_path =  base + "sentences/" + args.lg + "-phrase-sentences." +args.sn + ".tsv"
    
    if args.distributed:
        device = torch.device('cuda', args.local_rank)
    elif torch.cuda.is_available():
        num = args.gpu_id
        device = torch.device('cuda:{}'.format(str(num)))
        # torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        
    lossFunc = MoCoLoss().to(device)

    queue_length = int(args.queue_length)
    para_T = args.T_para
    with_span_eos = True if args.wo_span_eos == 'true' else False
    dev_filename = '-dev_qq' if args.dev_only_q_encoder == 1 or args.simclr == 1  else '-dev_qk'
    wolinear = '-wolinear' if args.wolinear == 1 else ''
    args.output_loss_dir = './' + args.output_log_dir + '/' + str(args.train_sample_num) + '-' + args.lg+ '-'+str(args.all_sentence_num)+ '-' +args.wo_span_eos + '-' + str(queue_length) + '-' + str(para_T)  + '-' + str(args.seed) \
            + '-' + str(args.num_train_epochs) + '-' + str(args.momentum) + '-' + str(args.simclr) + dev_filename + '-layer_' + str(args.layer_id) + wolinear
    args.output_model_path = './' + args.output_log_dir+ '/' + str(args.train_sample_num) + '-' + args.lg+ '-'+str(args.all_sentence_num) + '-' +args.wo_span_eos + '-' + str(queue_length) + '-' + str(para_T) + '-' + str(args.seed) \
        + '-' + str(args.num_train_epochs)  + '-' + str(args.momentum) + '-' + str(args.simclr) + dev_filename + '-layer_' + str(args.layer_id) + wolinear +  '/best.pt'
    best_acc = 0
    # Data
    train_phrase_pairs = load_words_mapping(args.train_phrase_path)
    dev_phrase_pairs = load_words_mapping(args.dev_phrase_path)
    test_phrase_pairs = load_words_mapping(args.test_phrase_path)
    en_word2context = load_word2context_from_tsv(args.src_context_path,args.all_sentence_num)
    lg_word2context = load_word2context_from_tsv(args.trg_context_path,args.all_sentence_num)
    train_dataset = WordWithContextDatasetWW(train_phrase_pairs, en_word2context, lg_word2context,prepend_bos=with_span_eos,append_eos=with_span_eos,sampleNum=args.train_sample_num,
        max_len=args.sentence_max_len)
    dev_dataset = WordWithContextDatasetWW(dev_phrase_pairs, en_word2context, lg_word2context,prepend_bos=with_span_eos,append_eos=with_span_eos,sampleNum=args.dev_sample_num,
        max_len=args.sentence_max_len)
    test_dataset = WordWithContextDatasetWW(test_phrase_pairs, en_word2context, lg_word2context,prepend_bos=with_span_eos,append_eos=with_span_eos,sampleNum=args.dev_sample_num,
        max_len=args.sentence_max_len)
    
    # Data Loader
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=args.local_rank)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              collate_fn=train_dataset.collate,drop_last=True,num_workers=16)
    val_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate, shuffle=False,num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate, shuffle=False,num_workers=16)

    # Model Init
    config = AutoConfig.from_pretrained("cwszz/XPR")
    model = MoCo(config=config,args=args,K=queue_length,T=para_T,m=args.momentum).to(device)

    bert_param_optimizer = model.named_parameters()
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(
        train_loader) // args.gradient_accumulation_steps,
                                                args.num_train_epochs * len(
                                                    train_loader) // args.gradient_accumulation_steps)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
    else:
        model.to(device)

    if args.local_rank == 1 or args.local_rank == -1:
        print(device)
        print(args)
        print(model)

    if not os.path.exists(args.output_loss_dir):
        os.mkdir(args.output_loss_dir)
    with open(args.output_loss_dir + '/loss_acc.txt','a+') as f:
        f.write("epoch,train_loss,train_acc,val_acc,val_p,val_rec,val_f1\n")

    for epoch in range(args.num_train_epochs):

        print('epoch:', epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_model(args, model, train_loader, lossFunc, optimizer, scheduler, device)
        val_acc, val_p, val_rec, val_f1 = test_model_single_encoder(args, model, val_loader, device)

        if args.local_rank == 1 or args.local_rank == -1:
            if not os.path.exists(args.output_loss_dir):
                os.mkdir(args.output_loss_dir)
            with open(args.output_loss_dir + '/loss_acc.txt','a+') as f:
                f.write(f"{str(epoch)},")
                f.write(f"{str(train_loss)},{str(train_acc)},")
                f.write(f"{str(val_acc)},{str(val_p)},{str(val_rec)},{str(val_f1)}\n")
            print("acc:", val_acc, "best_acc", best_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                if args.distributed:
                    torch.save(model.state_dict(),args.output_model_path)  # save as distributed
                else:
                    torch.save(model.state_dict(),args.output_model_path)

    if args.local_rank == 1 or args.local_rank == -1:
        model.load_state_dict(torch.load(args.output_model_path))
        val_acc, val_p, val_rec, val_f1 = test_model_single_encoder(args, model, test_loader, device)
        with open('result.txt','a+') as f:
            f.write(f"lang,acc,p,rec,f1\n")
            f.write(f"{args.lg},{str(val_acc)},{str(val_p)},{str(val_rec)},{str(val_f1)}\n")

main()
