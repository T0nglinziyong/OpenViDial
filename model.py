import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import collections

MAX_LENGTH = 30

class MyModel(pl.LightningModule):
    def __init__(self, num_hiddens=256, num_layers=6, nhead=4, dim_feedforward=1024, vocab=None, 
                 dropout=0.1, max_src_sent=2, img_dim=None, lr=1e-3
                 ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.vocab = vocab
        self.use_img = False if img_dim is None else True
        self.img_dim = img_dim
        self.lr = lr

        self.token_embedding = nn.Embedding(len(self.vocab), num_hiddens)
        self.position_embedding = nn.Embedding(MAX_LENGTH, num_hiddens)
        self.seq_embedding = nn.Embedding(max_src_sent, num_hiddens)

        self.fuse_img_token = nn.Linear(num_hiddens + img_dim, num_hiddens) if self.use_img else None
        encoder_layer = nn.TransformerEncoderLayer(num_hiddens, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, 
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(num_hiddens, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, 
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.linear = nn.Linear(num_hiddens, len(vocab))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'], reduction='mean')

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        src_imgs, src_tokens, tgt_tokens = batch
        batch_size = len(src_imgs)

        tgt_with_cls = tgt_tokens[:, :-1]
        tgt_tokens = tgt_tokens[:, 1:]
        
        x = self.forward_embedding(src_tokens, src_imgs)
        tgt = self.forward_embedding(tgt_with_cls)
        
        src_padding_mask, tgt_padding_mask, tgt_mask, memory_padding_mask = self.get_mask(src_tokens, tgt_tokens)

        memory = self.encoder(x, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, 
                               tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_padding_mask)
        output = self.linear(output)

        loss = self.loss_fn(output.reshape(-1, len(self.vocab)), tgt_tokens.reshape(-1))
        # Logging to TensorBoard by default
        self.log('train_loss', loss, batch_size=batch_size)   
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward_embedding(self, src_tokens, src_imgs=None):
        # embed tokens and positions
        x = token_embedding = self.token_embedding(src_tokens)
        # concat imgs features and reduce dimension
        if self.use_img:
            assert src_imgs is not None
            # src_imgs的形状为(B, T, C') T序列长度，C维度大小
            # src_tokens的形状为(B, T)
            # token_img_idxs 的形状为（批次大小，序列长度，图像特征维度），其中每个位置的值代表对应位置之前的 eos_index 的累计数量
            token_img_idxs = torch.cumsum((src_tokens == self.vocab['<sep>']).long(), dim=1).unsqueeze(-1).expand([-1, -1, self.img_dim])    
            # [B, T, C']  f[b][t][c] = src_imgs[b][token_img_idxs[b][t][c]][c]
            # 根据累加和张量 token_img_idxs，从图像特征张量 src_imgs 中选取对应的图像特征。这里采用了类似于索引的操作
            token_img_features = torch.gather(src_imgs, 1, token_img_idxs)
            # [B, T, C]
            x = self.fuse_img_token(torch.cat([x, token_img_features], dim=-1))

        seq = self.get_seq_and_position(src_tokens)
        # x += self.position_embedding(position)
        x += self.seq_embedding(seq)
        return x
    
    
    def get_mask(self, src_tokens, tgt_tokens):
        memory_padding_mask = src_padding_mask = src_tokens == self.vocab['<pad>']
        tgt_padding_mask = tgt_tokens == self.vocab['<pad>']

        tgt_len = tgt_tokens.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))) == 0
        return src_padding_mask, tgt_padding_mask, tgt_mask, memory_padding_mask

    def get_seq_and_position(self, src_tokens):
        seq = torch.cumsum((src_tokens == self.vocab['<sep>']).long(), dim=1)
        # position = torch.stack(torch.arange(0, counter[num], 1) for num in range(seq[-1] + 1))
        return seq#,  position


