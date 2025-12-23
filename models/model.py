#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
from models.structure_model.structure_encoder import StructureEncoder
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.multi_label_attention import HiAGMLA
from models.text_feature_propagation import HiAGMTP
from models.origin import Classifier


DATAFLOW_TYPE = {
    'HiAGM-TP': 'serial',
    'HiAGM-LA': 'parallel',
    'Origin': 'origin'
}


class HiAGM(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN'):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_type: Str, ('HiAGM-TP' for the serial variant of text propagation,
                                 'HiAGM-LA' for the parallel variant of multi-label soft attention,
                                 'Origin' without hierarchy-aware module)
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(HiAGM, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']

        self.token_embedding = EmbeddingLayer(
            vocab_map=self.token_map,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type,
            vocab=vocab,    # BERT
            device=self.device  # BERT
        )

        self.dataflow_type = DATAFLOW_TYPE[model_type]

        # self.text_encoder = TextEncoder(config)
        self.text_encoder = TextEncoder(config, vocab=vocab)

        self.structure_encoder = StructureEncoder(config=config,
                                                label_map=vocab.v2i['label'],
                                                device=self.device,
                                                graph_model_type=config.structure_encoder.type)

        if self.dataflow_type == 'serial':
            self.hiagm = HiAGMTP(config=config,
                                device=self.device,
                                graph_model=self.structure_encoder,
                                label_map=self.label_map)
        elif self.dataflow_type == 'parallel':
            self.hiagm = HiAGMLA(config=config,
                                device=self.device,
                                graph_model=self.structure_encoder,
                                label_map=self.label_map,
                                model_mode=model_mode)
        else:
            self.hiagm = Classifier(config=config,
                                    vocab=vocab,
                                    device=self.device)

    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                        'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                        }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.hiagm.parameters()})
        return params

    # def forward(self, batch):
    #     """
    #     forward pass of the overall architecture
    #     :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
    #     :return: 
    #     """
    #     # 檢查是否 BERT 模式
    #     # print(batch)
    #     if 'input_ids' in batch:
    #         # BERT 模式
    #         # print("[model.py] Use BERT")
    #         embedding = self.token_embedding(
    #             input_ids=batch['input_ids'].to(self.config.train.device_setting.device),
    #             attention_mask=batch['attention_mask'].to(self.config.train.device_setting.device),
    #             token_type_ids=batch.get('token_type_ids').to(self.config.train.device_setting.device) if batch.get('token_type_ids') is not None else None
    #         )
    #         token_output = self.text_encoder(embedding)
    #     else:        
    #         # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
    #         embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
    #         # get the length of sequences for dynamic rnn, (batch_size, 1)
    #         seq_len = batch['token_len']
    #         token_output = self.text_encoder(embedding, seq_len)

    #     logits = self.hiagm(token_output)

    #     return logits
    
    def forward(self, batch):
        chunked = batch.get('chunked', False)
        
        if 'input_ids' in batch:
            # BERT 模式
            embedding = self.token_embedding(
                input_ids=batch['input_ids'].to(self.config.train.device_setting.device),
                attention_mask=batch['attention_mask'].to(self.config.train.device_setting.device),
                token_type_ids=batch.get('token_type_ids').to(self.config.train.device_setting.device) if batch.get('token_type_ids') is not None else None,
                chunk_counts=batch.get('chunk_counts').to(self.config.train.device_setting.device) if chunked else None,
                chunked=chunked
            )
            token_output = self.text_encoder(embedding)
        else:
            # 原本的模式
            embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
            seq_len = batch['token_len']
            token_output = self.text_encoder(embedding, seq_len)

        logits = self.hiagm(token_output)
        return logits
