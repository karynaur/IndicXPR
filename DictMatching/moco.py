import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.cuda.amp import autocast

def _get_simclr_projection_head(num_ftrs: int, out_dim: int):
    modules = [
        nn.Linear(num_ftrs, num_ftrs*4),
        nn.ReLU(),
        nn.Linear(num_ftrs*4, out_dim)
    ]
    return nn.Sequential(*modules)

class BackBone_Model(nn.Module):
    
    def __init__(self, model='PrimeQA/DrDecr_XOR-TyDi_whitebox-2', layer_id=8, wo_linear_head=False):
        super(BackBone_Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.linear_head = _get_simclr_projection_head(self.config.hidden_size, self.config.hidden_size)
        self.criterion = nn.CrossEntropyLoss()
        self.layer_id = layer_id
        
    def convert(self, opt, mask, index, length):  # Conver Phrase to [1,H]
        '''obtain the opt and word position information (index, length),
        return tensor with shape [B, H]'''
        # opt: [B, S, H]; index: [B]; length: [B]; mask: [B, S]
        word_embd, new_opt, new_mask = [], [], []
        for item, item_mask, idx, l in zip(opt, mask, index, length):
            word_embd.append(item[idx:idx+l, :].mean(dim=0))    # [S, H] -> [H]
            if l > 1:
                item = torch.cat(
                    (
                        item[:idx, :],     # [S_, H]
                        word_embd[-1].unsqueeze(0),     # [1, H]
                        item[idx+l:, :],     # [S_, H]
                        torch.zeros(l-1, self.config.hidden_size).half().to(item_mask.device)
                    ), 
                    dim=0
                )
                item_mask = torch.cat(
                    (
                        item_mask[:idx], 
                        item_mask[idx:idx+1], 
                        item_mask[idx+l:], 
                        torch.zeros(l-1).long().to(item_mask.device)
                    )
                )
            new_opt.append(item)   # [S', H]
            new_mask.append(item_mask)
        word_embd = torch.stack(word_embd)    # [B, H] 
        new_opt = torch.stack(new_opt)
        new_mask = torch.stack(new_mask)
        
        # word_embd: [B, H]
        # new_opt: [B, S, H]
        # new_mask: [B, S]
        
        return word_embd
    
    def extra_first_cls(self,opt):
        word_embd = []
        for item in opt:
            word_embd.append(item[0])
        word_embd = torch.stack(word_embd)
        return word_embd

    def extra_cls(self,opt, mask, index, length):
        word_embd = []
        for item, idx in zip(opt, index):
            word_embd.append(item[idx])
        word_embd = torch.stack(word_embd)  
        return word_embd

    def _encode(self, input_ids, mask,type_ids):
        all_layers_hidden = self.model(input_ids=input_ids, attention_mask=mask,token_type_ids=None,output_hidden_states=True)[2]
        return all_layers_hidden[self.layer_id]
        
    def forward(self, w_indices=None, w_mask=None, w_index=None, w_length=None,w_type=None,sample_num=None):
        if sample_num != 0:
            '''w*_indices/w*_mask: [B, S]; w*_index/w*_length: [B]'''
            opt_w = self.linear_head(self._encode(w_indices, w_mask,w_type))
            w1_embd = self.convert(opt_w, w_mask, w_index, w_length)   # [B, H] Mean
            w1_embd = torch.mean(w1_embd.reshape(-1,sample_num,w1_embd.size(-1)),dim=1)
        else: 
            w1_embd = torch.mean(self.linear_head(self._encode(w_indices,w_mask,w_type)),dim=1)
        return w1_embd


class MoCo(nn.Module):
    def __init__(self,args, K=2048, m=0.999, T=0.04, config=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.train_sample_num = args.train_sample_num
        self.is_distributed = args.distributed
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = BackBone_Model(layer_id=args.layer_id)
        self.encoder_k = BackBone_Model(layer_id=args.layer_id)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(config.hidden_size, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.is_distributed:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self,list_x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        # for each_x in x:
        x = list_x[0] if isinstance(list_x,list) else list_x
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        idx_shuffle = torch.randperm(batch_size_all).to(x.device)
        torch.distributed.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        if isinstance(list_x,list):
            tensor_gather = [concat_all_gather(each_x)[idx_this] for each_x in list_x]
            return tensor_gather,idx_unshuffle
        else:
            return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, list_x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        x = list_x[0] if isinstance(list_x,list) else list_x
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        if isinstance(list_x,list):
            print('here')
            tensor_gather = [concat_all_gather(each_x)[idx_this] for each_x in list_x]
            return tensor_gather
        else:
            return x_gather[idx_this]


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        with autocast():
            q = self.encoder_q(*im_q,sample_num=self.train_sample_num)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                # shuffle for making use of BN
                if self.training:
                    self._momentum_update_key_encoder()  # update the key encoder
                  
                k = self.encoder_k(*im_k,sample_num=self.train_sample_num)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
 
            l_pos = torch.einsum('nc,mc->nm', [q, k])

            # negative logits: NxK
            if self.K != 0:
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                if self.training:
                    logits = torch.cat([l_pos, l_neg], dim=1)
                else:
                    logits = l_pos
            else:
                logits = l_pos
            logits /= self.T

        # labels: positive key indicators
            labels = torch.tensor([i for i in range(l_pos.shape[0])], dtype=torch.long).to(im_q[0].device)
            # dequeue and enqueue
            if self.training and self.K != 0:
                self._dequeue_and_enqueue(k)
            return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
