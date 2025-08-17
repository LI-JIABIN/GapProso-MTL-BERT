import json
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Union, Any  
import random
import numpy as np
import re
import os


class PoetryDataset(Dataset):

    def __init__(self,
                 file_path: str,
                 bert_vocab_path: str,
                 rhyme_list_path: str = r"D:\Users\lijiabin\PythonProject\char_predict\datasets\tone_df.xlsx",
                 max_seq_len: int = 500,
                 rhyme_features_path: str = "rhyme_features.pt"):  
        with open(file_path, 'r', encoding='utf-8') as f:
            data_raw = json.load(f)
        self.samples = []
        for sample in data_raw:
            if 'tokens' not in sample:
                print(sample)
                print(f"丢弃无效样本：{sample.get('id', '未知ID')}")
                continue
            self.samples.append(sample)

        self.vocab = self._load_bert_vocab(bert_vocab_path)

        self.rhyme_mapping = self._load_rhyme_system(rhyme_list_path)

        self.max_seq_len = max_seq_len
        self.pad_idx = self.vocab.get('[PAD]', 0)
        self.unk_idx = self.vocab.get('[UNK]', 1)
        self.valid_tones = {0, 1, 2, 3}  
        self.tone_pad_idx = len(self.valid_tones)

        if os.path.exists(rhyme_features_path):
            print(f"从文件加载预计算的韵部特征: {rhyme_features_path}")
            self.rhyme_features = torch.load(rhyme_features_path)
            print(f"加载了 {len(self.rhyme_features)} 个韵部特征")
        else:
            print(f"警告: 韵部特征文件 {rhyme_features_path} 不存在")
            self.rhyme_features = {}

        self.zero_vector = torch.zeros(768)
        self.embedding_cache = {}


        print(f"初始化完成: 词表大小={len(self.vocab)}，韵部映射大小={len(self.rhyme_mapping)}")

    def _load_bert_vocab(self, path: str) -> Dict[str, int]:
        with open(path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        return {token: idx for idx, token in enumerate(tokens)}

    def _load_rhyme_system(self, excel_path: str) -> Dict[str, int]:
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
            rhyme_list = df['rhyme'].unique().tolist()  # 保证唯一性且有序
            rhyme_list.append('未知')
            # print(len(rhyme_list))
            if 'rhyme' not in df.columns:
                raise ValueError("韵部表缺少'rhyme'列")
            return {rhyme: idx for idx, rhyme in enumerate(rhyme_list)}
        except Exception as e:
            raise RuntimeError(f"韵部表加载失败: {str(e)}")

    def _encode_sequence(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(t, self.unk_idx) for t in tokens]

    def _process_rhyme_char_with_bert(self, rhyme_char_list: List[str], mlm_mask: List[int],
                                      original_tokens: List[str]) -> Dict[str, torch.Tensor]:
        rhyme_embeddings = []

        for idx,rhyme_desc in enumerate(rhyme_char_list):
            if mlm_mask[idx]==-100:
                if rhyme_desc in self.rhyme_features:
                    rhyme_embeddings.append(self.rhyme_features[rhyme_desc])
                else:
                    rhyme_embeddings.append(self.zero_vector)
            else:
                rhyme_embeddings.append(self.zero_vector)

        rhyme_embeddings_tensor = torch.stack(rhyme_embeddings)  # [seq_len, 768]

        return {'rhyme_ids': rhyme_embeddings_tensor}
    def _process_rhyme_char(self, rhyme_char_list: List[str], mlm_mask: List[int], original_tokens: List[str]) -> Dict[
        str, torch.Tensor]:

        return self._process_rhyme_char_with_bert(rhyme_char_list, mlm_mask, original_tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # print(sample.keys())
        original_tokens = sample['tokens'][:self.max_seq_len]

        masked_tokens, mlm_labels = self._apply_dynamic_mask_2(original_tokens)
        # masked_tokens, mlm_labels = self._apply_NewSchool_mask(original_tokens)
        # print(masked_tokens)


        if 'tone_char' in sample:
            base_tones = [t - 1 for t in sample['tone_char'][:self.max_seq_len]]  # 原始标签1-4转0-3


        assert all(0 <= t <= 3 for t in base_tones), \
            f"非法平仄标签 {base_tones}"
        tone_ids, tone_labels = self._create_dual_tone_tags(base_tones, mlm_labels)

        rhyme_char_list = sample['rhyme_char'][:self.max_seq_len]
        rhyme_char_data = self._process_rhyme_char(rhyme_char_list, mlm_labels, original_tokens)

        rhyme_label_value = self.rhyme_mapping.get(sample['rhyme_label'], len(self.rhyme_mapping) - 1)
        rhyme_label_tensor = torch.tensor([[rhyme_label_value]], dtype=torch.long)  # 形状为[1, 1]

        return {
            'input_ids': torch.tensor(self._encode_sequence(masked_tokens), dtype=torch.long),
            'input_labels': torch.tensor(mlm_labels, dtype=torch.long),  # 原labels改名为input_labels
            'tone_ids': torch.tensor(tone_ids, dtype=torch.long),
            'tone_labels': torch.tensor(tone_labels, dtype=torch.long),
            'rhyme_ids': rhyme_char_data['rhyme_ids'],
            'rhyme_label': rhyme_label_tensor,
            'attention_mask': torch.tensor([1] * len(masked_tokens), dtype=torch.long)
        }

    def _create_dual_tone_tags(self, base_tones, mlm_mask):
        tone_ids = []
        tone_labels = []
        for t, m in zip(base_tones, mlm_mask):

            if t not in self.valid_tones:
                raise ValueError(f"非法平仄值 {t + 1}，请检查数据预处理")

            tone_ids.append(t if m == -100 else self.tone_pad_idx)  # 使用预定义的pad_idx

            tone_labels.append(t if m != -100 else -100)
        return tone_ids, tone_labels

    def _apply_dynamic_mask_1(self, tokens):
        mlm_labels = [-100] * len(tokens)
        mask_indices = sorted(random.sample(range(len(tokens)), k=int(0.2 * len(tokens))))
        # print(mask_indices)
        for idx in mask_indices:
            mlm_labels[idx] = self.vocab.get(tokens[idx], self.unk_idx)
            rand = random.random()
            if rand < 0.8:
                tokens = tokens[:idx] + ['[MASK]'] + tokens[idx + 1:]
            elif rand < 0.9:
                tokens = tokens[:idx] + [random.choice(list(self.vocab))] + tokens[idx + 1:]
            else:
                pass  # 无需修改tokens，但仍需记录label
        return tokens, mlm_labels
    def _apply_dynamic_mask_2(self, tokens):
        mlm_labels = [-100] * len(tokens)
        
        max_span_length = 5  
        mask_prob = 0.2  
        
        token_len = len(tokens)
        num_to_mask = int(token_len * mask_prob)
        remaining_to_mask = num_to_mask
        masked_indices = set()
        
        while remaining_to_mask > 0:
            if len(masked_indices) >= token_len * 0.8:  
                break
                
            start_idx = random.randint(0, token_len - 1)
            
            if start_idx in masked_indices:
                continue
                
            current_span_length = min(
                random.randint(1, max_span_length),  
                remaining_to_mask, 
                token_len - start_idx  
            )
            
            for i in range(start_idx, min(start_idx + current_span_length, token_len)):
                if i not in masked_indices:  
                    masked_indices.add(i)
                    mlm_labels[i] = self.vocab.get(tokens[i], self.unk_idx)
                    remaining_to_mask -= 1
                    if remaining_to_mask <= 0:
                        break
        
        masked_tokens = list(tokens)  
        for idx in masked_indices:
            rand = random.random()
            if rand < 0.8:
                masked_tokens[idx] = '[MASK]'
            elif rand < 0.9:
                masked_tokens[idx] = random.choice(list(self.vocab))
        
        return masked_tokens, mlm_labels
    def _apply_NewSchool_mask(self, tokens):
        mlm_labels = [-100] * len(tokens)
        text = "".join(tokens).replace(',', '，').replace('?', '？').replace('!', '！').replace(';', '；')
        pattern = re.compile(r'[^，。？、：*（）！；]+[，。？！；]')  # 脱字符 ^ 表示“取反”
        matches = list(pattern.finditer(text))
        target_idxs = []
        for idx, m in enumerate(matches):
            if idx % 2 == 1:  # 取 matches[1], matches[3], ...
                sent = m.group()
                char_pos_in_text = m.start() + (len(sent) - 2)
                target_idxs.append(char_pos_in_text)
        for mask_idx in target_idxs:
            original_token = tokens[mask_idx]
            mlm_labels[mask_idx] = self.vocab.get(original_token, self.unk_idx)
            tokens[mask_idx] = '[MASK]'
        return tokens, mlm_labels

    def _create_masked_tones(self, tones, mlm_labels):
        return [t if mlm_labels[i] == -100 else 4 for i, t in enumerate(tones)]


def collate_fn(batch):
    max_seq_len = max([d['input_ids'].size(0) for d in batch])

    padded_rhyme_ids = []

    for d in batch:
        curr_seq_len = d['rhyme_ids'].size(0)

        padded = torch.zeros((max_seq_len, 768), dtype=torch.float)

        padded[:curr_seq_len, :] = d['rhyme_ids']

        padded_rhyme_ids.append(padded)

    rhyme_labels = torch.cat([d['rhyme_label'] for d in batch], dim=0)

    batch_dict = {
        'input_ids': pad_sequence([d['input_ids'] for d in batch],
                                  batch_first=True, padding_value=0),
        'input_labels': pad_sequence([d['input_labels'] for d in batch],
                                     batch_first=True, padding_value=-100),
        'tone_ids': pad_sequence([d['tone_ids'] for d in batch],
                                 batch_first=True, padding_value=4),  
        'tone_labels': pad_sequence([d['tone_labels'] for d in batch],
                                    batch_first=True, padding_value=-100),
        'rhyme_ids': torch.stack(padded_rhyme_ids),
        'rhyme_label': rhyme_labels,  # 形状为[batch_size, 1]
        'attention_mask': pad_sequence([d['attention_mask'] for d in batch],
                                       batch_first=True, padding_value=0)
    }


    return batch_dict


if __name__ == "__main__":
    train_set = PoetryDataset(
        file_path=r"D:\Users\lijiabin\PythonProject\char_predict\split_data\第二版\train\train_1.json",
        bert_vocab_path=r"D:\Users\lijiabin\PythonProject\char_predict\bert-base-chinese\vocab.txt",
        rhyme_list_path="./datasets/tone_df.xlsx"
    )

    train_loader = DataLoader(train_set, batch_size=1, collate_fn=collate_fn, shuffle=False)

    sample = next(iter(train_loader))
    print("\n----- 数据批次示例 -----")
    print("输入张量形状:", sample['input_ids'].shape)
    print("输入标签形状:", sample['input_labels'].shape)
    # print("平仄输入形状:", sample['tone_ids'].shape)
    # print("平仄标签形状:", sample['tone_labels'].shape)
    # print("韵律输入形状:", sample['rhyme_ids'].shape)  
    # print("韵部标签形状:", sample['rhyme_label'].shape)  # 应该是[batch_size, 1]
    # print("注意力掩码形状:", sample['attention_mask'].shape)
    #
    print("input_ids:", sample['input_ids'])
    print("input_labels:", sample['input_labels'])
    # print("tone_ids:", sample['tone_ids'])
    # print("tone_labels:", sample['tone_labels'])
    # print("rhyme_ids:", sample['rhyme_ids']) 
    # print("rhyme_label:", sample['rhyme_label'])  
    # print("注意力掩码:", sample['attention_mask'])



