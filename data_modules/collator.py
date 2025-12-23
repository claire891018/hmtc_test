# ============================================
# 1. data_modules/collator.py
# ============================================
#!/usr/bin/env python
# coding: utf-8

import torch

# class Collator(object):
#     def __init__(self, config, vocab):
#         super(Collator, self).__init__()
#         self.device = config.train.device_setting.device
#         self.label_size = len(vocab.v2i['label'].keys())
#         self.vocab = vocab
#         self.use_bert = vocab.use_bert
#         if self.use_bert:
#             self.max_length = config.text_encoder.max_length

#     def _multi_hot(self, batch_labels):
#         batch_size = len(batch_labels)
#         max_length = max([len(sample) for sample in batch_labels])
#         aligned_batch_labels = []
#         for sample_label in batch_labels:
#             aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
#         aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
#         batch_labels_multi_hot = torch.zeros(batch_size, self.label_size).scatter_(1, aligned_batch_labels, 1)
#         return batch_labels_multi_hot

#     def __call__(self, batch):
#         if self.use_bert:
#             return self._bert_collate(batch)
#         else:
#             return self._original_collate(batch)
    
#     # def _bert_collate(self, batch):
#     #     texts = [' '.join(sample['token']) for sample in batch]
#     #     batch_label = [sample['label'] for sample in batch]
        
#     #     encoded = self.vocab.bert_batch_tokenize(texts, max_length=self.max_length)
#     #     batch_multi_hot_label = self._multi_hot(batch_label)
        
#     #     return {
#     #         'input_ids': encoded['input_ids'],
#     #         'attention_mask': encoded['attention_mask'],
#     #         'token_type_ids': encoded.get('token_type_ids'),
#     #         'label': batch_multi_hot_label,
#     #         'label_list': batch_label
#     #     }
        
#     def _bert_collate(self, batch):
#         # 改成：從原始 token 取得
#         texts = []
#         for sample in batch:
#             # sample['token'] 是 indices，需要從原始數據取 token 字符串
#             # 或者直接用 token_text（如果有的話）
#             if 'token_text' in sample:
#                 texts.append(sample['token_text'])
#             else:
#                 # 如果沒有原始文本，從 vocab 轉回來
#                 tokens = [self.vocab.i2v['token'].get(idx, '<UNK>') for idx in sample['token']]
#                 texts.append(' '.join(tokens))
        
#         batch_label = [sample['label'] for sample in batch]
        
#         encoded = self.vocab.bert_batch_tokenize(texts, max_length=self.max_length)
#         batch_multi_hot_label = self._multi_hot(batch_label)
        
#         return {
#             'input_ids': encoded['input_ids'],
#             'attention_mask': encoded['attention_mask'],
#             'token_type_ids': encoded.get('token_type_ids'),
#             'label': batch_multi_hot_label,
#             'label_list': batch_label
#         }
        
#     def _original_collate(self, batch):
#         batch_token = []
#         batch_label = []
#         batch_doc_len = []
#         for sample in batch:
#             batch_token.append(sample['token'])
#             batch_label.append(sample['label'])
#             batch_doc_len.append(sample['token_len'])

#         batch_token = torch.tensor(batch_token)
#         batch_multi_hot_label = self._multi_hot(batch_label)
#         batch_doc_len = torch.FloatTensor(batch_doc_len)
#         return {
#             'token': batch_token,
#             'label': batch_multi_hot_label,
#             'token_len': batch_doc_len,
#             'label_list': batch_label
#         }
        
class Collator(object):
    def __init__(self, config, vocab):
        super(Collator, self).__init__()
        self.device = config.train.device_setting.device
        self.label_size = len(vocab.v2i['label'].keys())
        self.vocab = vocab
        self.use_bert = vocab.use_bert
        if self.use_bert:
            self.max_length = config.text_encoder.max_length
            # ===== 新增：讀取 chunking config =====
            self.chunking_config = {
                'enabled': config.text_encoder.chunking.enabled if hasattr(config.text_encoder, 'chunking') else False,
                'chunk_size': config.text_encoder.chunking.chunk_size if hasattr(config.text_encoder.chunking, 'chunk_size') else 450,
                'overlap': config.text_encoder.chunking.overlap if hasattr(config.text_encoder.chunking, 'overlap') else 50,
                'max_chunks': config.text_encoder.chunking.max_chunks if hasattr(config.text_encoder.chunking, 'max_chunks') else 10,
                'max_length': self.max_length
            }
            self.pooling_type = config.text_encoder.pooling.type if hasattr(config.text_encoder, 'pooling') else 'mean'
            # =====================================

    def _bert_collate(self, batch):
        texts = []
        for sample in batch:
            if 'token_text' in sample:
                texts.append(sample['token_text'])
            else:
                tokens = [self.vocab.i2v['token'].get(idx, '<UNK>') for idx in sample['token']]
                texts.append(' '.join(tokens))
        
        batch_label = [sample['label'] for sample in batch]
        
        # ===== 修改：使用 chunking tokenization =====
        if self.chunking_config['enabled']:
            # 返回 List[List[Dict]] - 每個樣本有多個 chunks
            batch_chunks = self.vocab.bert_batch_chunk_tokenize(texts, self.chunking_config)
            
            # 需要處理成統一格式
            # 找最大 chunk 數量
            max_num_chunks = max(len(chunks) for chunks in batch_chunks)
            
            # Padding 到相同數量的 chunks
            input_ids_list = []
            attention_mask_list = []
            chunk_counts = []  # 記錄每個樣本實際有幾個 chunks
            
            for chunks in batch_chunks:
                chunk_counts.append(len(chunks))
                # Pad chunks
                while len(chunks) < max_num_chunks:
                    # 添加 dummy chunk (全是 padding)
                    chunks.append({
                        'input_ids': [self.vocab.padding_index] * 512,
                        'attention_mask': [0] * 512
                    })
                
                # 堆疊
                input_ids_list.append([chunk['input_ids'] for chunk in chunks])
                attention_mask_list.append([chunk['attention_mask'] for chunk in chunks])
            
            input_ids = torch.tensor(input_ids_list)  # [batch, max_chunks, 512]
            attention_mask = torch.tensor(attention_mask_list)
            chunk_counts = torch.tensor(chunk_counts)  # [batch]
            
            batch_multi_hot_label = self._multi_hot(batch_label)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'chunk_counts': chunk_counts,  # 新增：記錄實際 chunk 數
                'label': batch_multi_hot_label,
                'label_list': batch_label,
                'chunked': True  # 標記這是 chunked data
            }
        else:
            # ===== 原本的邏輯（不 chunking）=====
            encoded = self.vocab.bert_batch_tokenize(texts, max_length=self.max_length)
            batch_multi_hot_label = self._multi_hot(batch_label)
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'token_type_ids': encoded.get('token_type_ids'),
                'label': batch_multi_hot_label,
                'label_list': batch_label,
                'chunked': False
            }