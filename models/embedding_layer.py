# ============================================
# 2. models/embedding_layer.py
# ============================================
#!/usr/bin/env python
# coding:utf-8

import numpy as np
import torch
import helper.logger as logger
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
from transformers import BertModel

INIT_FUNC = {
    'uniform': uniform_,
    'kaiming_uniform': kaiming_uniform_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_normal': kaiming_normal_
}


# class BertEmbeddingLayer(torch.nn.Module):
#     def __init__(self, config, vocab, device):
#         super(BertEmbeddingLayer, self).__init__()
#         self.config = config
#         self.device = device
        
#         self.bert = BertModel.from_pretrained(
#             config['embedding']['token']['pretrained_model']
#         )
        
#         # if config['embedding']['token'].get('freeze_bert', False):
#         if hasattr(config['embedding']['token'], 'freeze_bert') and config['embedding']['token']['freeze_bert']:
#             for param in self.bert.parameters():
#                 param.requires_grad = False
        
#         self.dropout = torch.nn.Dropout(p=config['embedding']['token']['dropout'])
    
#     # def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#     #     outputs = self.bert(
#     #         input_ids=input_ids,
#     #         attention_mask=attention_mask,
#     #         token_type_ids=token_type_ids
#     #     )
#     #     # pooled_output = outputs.pooler_output
#     #     sequence_output = outputs.last_hidden_state 
#     #     return self.dropout(sequence_output)
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         cls_output = outputs.last_hidden_state[:, 0, :]  # 只取 [CLS]
#         return cls_output.unsqueeze(1)  # [batch, 1, 768]


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 vocab=None,
                 device=None
                 ):
        super(EmbeddingLayer, self).__init__()
        
        # 檢查是否使用 BERT
        print(f"vocab is not None: {vocab is not None}")
        print(f"vocab.use_bert: {vocab.use_bert if vocab else 'N/A'}")
        print(f"vocab_name == 'token': {vocab_name == 'token'}")
        self.use_bert = vocab is not None and vocab.use_bert and vocab_name == 'token'
        
        if self.use_bert:
            self.bert_embedding = BertEmbeddingLayer(config, vocab, device)
            print('Use BERT')
        else:
            self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout'])
            self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index)

            assert initial_type in INIT_FUNC
            if initial_type.startswith('kaiming'):
                self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                        embedding_dim),
                                                            a=negative_slope,
                                                            mode=mode_fan,
                                                            nonlinearity=activation_type)
            elif initial_type.startswith('xavier'):
                self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                        embedding_dim),
                                                            gain=torch.nn.init.calculate_gain(activation_type))
            else:
                self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                        embedding_dim),
                                                            a=-0.25,
                                                            b=0.25)

            if model_mode == 'TRAIN' and config['embedding'][vocab_name]['type'] == 'pretrain' \
                    and pretrained_dir is not None and pretrained_dir != '':
                self.load_pretrained(embedding_dim, vocab_map, vocab_name, pretrained_dir)

            if padding_index is not None:
                self.lookup_table[padding_index] = 0.0
            self.embedding.weight.data.copy_(self.lookup_table)
            self.embedding.weight.requires_grad = True
            del self.lookup_table

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, pretrained_dir):
        logger.info('Loading {}-dimension {} embedding from pretrained file: {}'.format(
            embedding_dim, vocab_name, pretrained_dir))
        with open(pretrained_dir, 'r', encoding='utf8') as f_in:
            num_pretrained_vocab = 0
            for line in f_in:
                row = line.rstrip('\n').split(' ')
                if len(row) == 2:
                    assert int(row[1]) == embedding_dim, 'Pretrained dimension %d dismatch the setting %d' \
                                                         % (int(row[1]), embedding_dim)
                    continue
                if row[0] in vocab_map:
                    current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
                    self.lookup_table[vocab_map[row[0]]] = current_embedding
                    num_pretrained_vocab += 1
        logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def forward(self, vocab_id_list=None, input_ids=None, attention_mask=None, token_type_ids=None):
        if self.use_bert:
            return self.bert_embedding(input_ids, attention_mask, token_type_ids)
        else:
            embedding = self.embedding(vocab_id_list)
            return self.dropout(embedding)
        

class BertEmbeddingLayer(torch.nn.Module):
    def __init__(self, config, vocab, device):
        super(BertEmbeddingLayer, self).__init__()
        self.config = config
        self.device = device
        
        self.bert = BertModel.from_pretrained(
            config['embedding']['token']['pretrained_model']
        )
        
        if hasattr(config['embedding']['token'], 'freeze_bert') and config['embedding']['token']['freeze_bert']:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = torch.nn.Dropout(p=config['embedding']['token']['dropout'])
        
        # ===== 新增：Pooling module =====
        pooling_type = config['text_encoder']['pooling']['type'] if 'pooling' in config['text_encoder'] else 'mean'
        hidden_dim = config['embedding']['token']['dimension']
        
        if pooling_type == 'attention':
            self.pooling = AttentionPooling(hidden_dim)
        elif pooling_type == 'mean':
            self.pooling = lambda x, counts: self._mean_pooling(x, counts)
        elif pooling_type == 'max':
            self.pooling = lambda x, counts: self._max_pooling(x)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        self.pooling_type = pooling_type
        # ================================
    
    def _mean_pooling(self, chunk_embeddings, chunk_counts):
        """
        Mean pooling over chunks
        :param chunk_embeddings: [batch, max_chunks, hidden]
        :param chunk_counts: [batch], 每個樣本實際的 chunk 數
        """
        # Mask out padding chunks
        mask = torch.arange(chunk_embeddings.size(1)).unsqueeze(0).to(chunk_embeddings.device)
        mask = mask < chunk_counts.unsqueeze(1)  # [batch, max_chunks]
        mask = mask.unsqueeze(-1).float()  # [batch, max_chunks, 1]
        
        # Weighted sum
        pooled = (chunk_embeddings * mask).sum(dim=1) / chunk_counts.unsqueeze(-1).float()
        return pooled  # [batch, hidden]
    
    def _max_pooling(self, chunk_embeddings):
        """Max pooling over chunks"""
        return chunk_embeddings.max(dim=1)[0]  # [batch, hidden]
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, chunk_counts=None, chunked=False):
        if chunked:
            # ===== Chunked mode =====
            batch_size, max_chunks, seq_len = input_ids.size()
            
            # Reshape to process all chunks at once
            input_ids_flat = input_ids.view(-1, seq_len)  # [batch*max_chunks, seq_len]
            attention_mask_flat = attention_mask.view(-1, seq_len)
            
            # BERT encoding
            outputs = self.bert(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat
            )
            cls_outputs = outputs.last_hidden_state[:, 0, :]  # [batch*max_chunks, hidden]
            
            # Reshape back
            chunk_embeddings = cls_outputs.view(batch_size, max_chunks, -1)  # [batch, max_chunks, hidden]
            
            # Pooling
            if self.pooling_type == 'attention':
                pooled = self.pooling(chunk_embeddings, chunk_counts)
            else:
                pooled = self.pooling(chunk_embeddings, chunk_counts)
            
            return pooled.unsqueeze(1)  # [batch, 1, hidden] 保持原格式
        else:
            # ===== 原本的單一輸入模式 =====
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            cls_output = outputs.last_hidden_state[:, 0, :]
            return cls_output.unsqueeze(1)


class AttentionPooling(torch.nn.Module):
    """Attention-based pooling over chunks"""
    def __init__(self, hidden_dim):
        super().__init__()
        # 或 super(AttentionPooling, self).__init__()
        self.attention = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, chunk_embeddings, chunk_counts):
        """
        :param chunk_embeddings: [batch, max_chunks, hidden]
        :param chunk_counts: [batch]
        """
        # Compute attention scores
        scores = self.attention(chunk_embeddings).squeeze(-1)  # [batch, max_chunks]
        
        # Mask padding chunks
        mask = torch.arange(chunk_embeddings.size(1)).unsqueeze(0).to(chunk_embeddings.device)
        mask = mask < chunk_counts.unsqueeze(1)  # [batch, max_chunks]
        scores = scores.masked_fill(~mask, -1e9)
        
        # Softmax
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [batch, max_chunks, 1]
        
        # Weighted sum
        pooled = (chunk_embeddings * weights).sum(dim=1)  # [batch, hidden]
        return pooled
    
    