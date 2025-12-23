#!/usr/bin/env python
# coding:utf-8

import pickle
from collections import Counter
import helper.logger as logger
import tqdm
import os
import json
from transformers import BertTokenizer


class Vocab(object):
    def __init__(self, config, min_freq=1, special_token=['<PADDING>', '<OOV>'], max_size=None):
        """
        vocabulary class for text classification, initialized from pretrained embedding file
        and update based on minimum frequency and maximum size
        :param config: helper.configure, Configure Object
        :param min_freq: int, the minimum frequency of tokens
        :param special_token: List[Str], e.g. padding and out-of-vocabulary
        :param max_size: int, maximum size of the overall vocabulary
        """
        logger.info('Building Vocabulary....')
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file)}
        counter = Counter()
        self.config = config
        # counter for tokens
        self.freqs = {'token': counter.copy(), 'label': counter.copy()}
        # vocab to index
        self.v2i = {'token': dict(), 'label': dict()}
        # index to vocab
        self.i2v = {'token': dict(), 'label': dict()}

        self.min_freq = max(min_freq, 1)
        
        # ===== 新增：檢查是否使用 BERT =====
        self.use_bert = hasattr(config.embedding.token, 'type') and \
                        config.embedding.token.type == 'bert'
        
        # ===== 新增：如果使用 BERT，初始化 BERT tokenizer =====
        if self.use_bert:
            logger.info('Using BERT tokenizer...')
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                config.embedding.token.pretrained_model
            )
            # BERT 的特殊 token indices
            self.padding_index = self.bert_tokenizer.pad_token_id
            self.oov_index = self.bert_tokenizer.unk_token_id
            self.cls_index = self.bert_tokenizer.cls_token_id
            self.sep_index = self.bert_tokenizer.sep_token_id
        # ==========================================
        
        if not os.path.isdir(self.config.vocabulary.dir):
            os.system('mkdir ' + str(self.config.vocabulary.dir))
        token_dir = os.path.join(self.config.vocabulary.dir, self.config.vocabulary.vocab_dict)
        label_dir = os.path.join(self.config.vocabulary.dir, self.config.vocabulary.label_dict)
        vocab_dir = {'token': token_dir, 'label': label_dir}
        
        # ===== 修改：BERT 不需要載入 token vocabulary =====
        if os.path.isfile(label_dir) and (self.use_bert or os.path.isfile(token_dir)):
            logger.info('Loading Vocabulary from Cached Dictionary...')
            
            # BERT 不需要載入 token vocab
            if not self.use_bert:
                with open(token_dir, 'r') as f_in:
                    for i, line in enumerate(f_in):
                        data = line.rstrip().split('\t')
                        assert len(data) == 2
                        self.v2i['token'][data[0]] = i
                        self.i2v['token'][i] = data[0]
            else:
                logger.info('Using BERT vocabulary, skipping token dict loading...')
            
            # Label vocab 始終需要載入
            with open(label_dir, 'r') as f_in:
                for i, line in enumerate(f_in):
                    data = line.rstrip().split('\t')
                    assert len(data) == 2
                    self.v2i['label'][data[0]] = i
                    self.i2v['label'][i] = data[0]
            
            for vocab in self.v2i.keys():
                if vocab == 'token' and self.use_bert:
                    logger.info('Vocabulary of token: Using BERT vocab (size: {})'.format(
                        self.bert_tokenizer.vocab_size))
                else:
                    logger.info('Vocabulary of ' + vocab + ' ' + str(len(self.v2i[vocab])))
        # ================================================
        
        else:
            logger.info('Generating Vocabulary from Corpus...')
            
            # ===== 修改：BERT 不需要載入預訓練詞向量 =====
            if not self.use_bert:
                self._load_pretrained_embedding_vocab()
            else:
                logger.info('Using BERT, skipping pretrained embedding loading...')
            # ===========================================
            
            self._count_vocab_from_corpus()
            for vocab in self.freqs.keys():
                if vocab == 'token' and self.use_bert:
                    continue  # BERT 不需要 token frequency
                logger.info('Vocabulary of ' + vocab + ' ' + str(len(self.freqs[vocab])))

            # ===== 修改：BERT 不需要 shrink token vocab =====
            if not self.use_bert:
                self._shrink_vocab('token', max_size)
                for s_token in special_token:
                    self.freqs['token'][s_token] = self.min_freq
            # ===========================================

            for field in self.freqs.keys():
                # ===== 修改：BERT 跳過 token 處理 =====
                if field == 'token' and self.use_bert:
                    continue
                # =====================================
                
                temp_vocab_list = list(self.freqs[field].keys())
                for i, k in enumerate(temp_vocab_list):
                    self.v2i[field][k] = i
                    self.i2v[field][i] = k
                logger.info('Vocabulary of ' + field + ' with the size of ' + str(len(self.v2i[field].keys())))
                with open(vocab_dir[field], 'w') as f_out:
                    for k in list(self.v2i[field].keys()):
                        f_out.write(k + '\t' + str(self.freqs[field][k]) + '\n')
                logger.info('Save Vocabulary in ' + vocab_dir[field])
        
        # ===== 修改：只在非 BERT 模式設置這些 index =====
        if not self.use_bert:
            self.padding_index = self.v2i['token']['<PADDING>']
            self.oov_index = self.v2i['token']['<OOV>']
        # ============================================

    def _load_pretrained_embedding_vocab(self):
        """
        initialize counter for word in pre-trained word embedding
        """
        pretrained_file_dir = self.config.embedding.token.pretrained_file
        with open(pretrained_file_dir, 'r', encoding='utf8') as f_in:
            logger.info('Loading vocabulary from pretrained embedding...')
            for line in tqdm.tqdm(f_in):
                data = line.rstrip('\n').split(' ')
                if len(data) == 2:
                    # first line in pretrained embedding
                    continue
                v = data[0]
                self.freqs['token'][v] += self.min_freq + 1

    def _count_vocab_from_corpus(self):
        """
        count the frequency of tokens in the specified corpus
        """
        for corpus in self.corpus_files.keys():
            mode = 'ALL'
            with open(self.corpus_files[corpus], 'r') as f_in:
                logger.info('Loading ' + corpus + ' subset...')
                for line in tqdm.tqdm(f_in):
                    data = json.loads(line.rstrip())
                    self._count_vocab_from_sample(data, mode)

    def _count_vocab_from_sample(self, line_dict, mode='ALL'):
        """
        update the frequency from the current sample
        :param line_dict: Dict{'token': List[Str], 'label': List[Str]}
        """
        for k in self.freqs.keys():
            # ===== 修改：BERT 只計算 label 頻率 =====
            if k == 'token' and self.use_bert:
                continue
            # ======================================
            
            if mode == 'ALL':
                for t in line_dict[k]:
                    self.freqs[k][t] += 1
            else:
                if k == 'token':
                    for t in line_dict['token']:
                        self.freqs['token'][t] += 1

    def _shrink_vocab(self, k, max_size=None):
        """
        shrink the vocabulary
        :param k: Str, field <- 'token', 'label'
        :param max_size: int, the maximum number of vocabulary
        """
        logger.info('Shrinking Vocabulary...')
        tmp_dict = Counter()
        for v in self.freqs[k].keys():
            if self.freqs[k][v] >= self.min_freq:
                tmp_dict[v] = self.freqs[k][v]
        if max_size is not None:
            tmp_list_dict = tmp_dict.most_common(max_size)
            self.freqs[k] = Counter()
            for (t, v) in tmp_list_dict:
                self.freqs[k][t] = v
        logger.info('Shrinking Vocabulary of tokens: ' + str(len(self.freqs[k])))
    
    # ===== 新增：BERT tokenize 方法 =====
    def bert_tokenize(self, text, max_length=512):
        """
        使用 BERT tokenizer 對文本進行 tokenization
        :param text: str or List[str], 輸入文本
        :param max_length: int, 最大序列長度
        :return: dict with input_ids, attention_mask, token_type_ids
        """
        if not self.use_bert:
            raise ValueError("BERT tokenizer is not initialized. Set embedding.token.type to 'bert'")
        
        # 如果輸入是 token list，先轉成字符串
        if isinstance(text, list):
            text = ' '.join(text)
        
        # 使用 BERT tokenizer
        encoded = self.bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def bert_batch_tokenize(self, texts, max_length=512):
        """
        批量 tokenization
        :param texts: List[str] or List[List[str]], 批量文本
        :param max_length: int, 最大序列長度
        :return: dict with batched input_ids, attention_mask, token_type_ids
        """
        if not self.use_bert:
            raise ValueError("BERT tokenizer is not initialized")
        
        # 處理 token list 格式
        processed_texts = []
        for text in texts:
            if isinstance(text, list):
                processed_texts.append(' '.join(text))
            else:
                processed_texts.append(text)
        
        # 批量 tokenize
        encoded = self.bert_tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoded
    # ===================================