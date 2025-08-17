import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import pandas as pd
from model import MultiTaskBERT
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertPreTrainedModel, BertModel
from data_utils import PoetryDataset, collate_fn


class EnhancedToneAwareEvaluator:
    def __init__(self, topk_list, id2char, char2tone, char2rhyme, bert_tokenizer=None):
        self.topk_list = sorted(topk_list)
        self.max_k = max(topk_list)
        self.id2char = id2char
        self.char2tone = char2tone
        self.char2rhyme = char2rhyme
        self.bert_tokenizer = bert_tokenizer
        self.softmax = torch.nn.Softmax(dim=-1)
        self.metrics = {
            'char_total': 0,
            'tone_total': 0,
            'rhyme_total': 0,
            **{f'char_top{k}': 0 for k in topk_list},
            **{f'tone_top{k}': 0 for k in topk_list},
            **{f'rhyme_top{k}': 0 for k in topk_list}
        }

    def update(self, mlm_logits, input_labels, tone_labels, model_type):
        if model_type == 'proposed':
            self._update_proposed_metrics(mlm_logits, input_labels, tone_labels)
        else:
            raise ValueError("Invalid model type for this evaluator")

    def _update_proposed_metrics(self, logits, labels, tones):
        device = logits.device
        labels = labels.to(device)
        tones = tones.to(device)

        batch_size, seq_len, vocab_size = logits.shape
        mask = (labels != -100)
        num_masks = mask.sum().item()

        max_k = self.max_k
        topk_values, topk_indices = logits.topk(max_k, dim=-1)
        masked_topk_indices = topk_indices[mask]  # [num_masks, max_k]
        masked_labels = labels[mask]  # [num_masks]

        comparison_matrix = (masked_topk_indices == masked_labels.to(masked_topk_indices.device).unsqueeze(-1))

        for k in self.topk_list:
            correct = comparison_matrix[:, :k].any(dim=1)
            self.metrics[f'char_top{k}'] += correct.sum().item()


        all_candidate_tones = [
            [self.char2tone.get(self.id2char.get(cid.item(), ''), 3)
             for cid in candidates]
            for candidates in masked_topk_indices
        ]


        all_candidate_rhymes = []
        if self.char2rhyme: 
            for candidates in masked_topk_indices:
                candidate_rhymes = []
                for cid in candidates:
                    char = self.id2char.get(cid.item(), '')
                    rhyme_ids = self.char2rhyme.get(char, [])
                    candidate_rhymes.append(rhyme_ids)
                all_candidate_rhymes.append(candidate_rhymes)

            true_chars = [self.id2char.get(label.item(), '') for label in masked_labels]
            true_rhymes = []
            for char in true_chars:
                rhyme_ids = self.char2rhyme.get(char, [])
                true_rhymes.append(rhyme_ids)

        true_tones = tones[mask].cpu().numpy()
        for k in self.topk_list:
            for i in range(num_masks):
                if true_tones[i] in all_candidate_tones[i][:k]:
                    self.metrics[f'tone_top{k}'] += 1

                if self.char2rhyme:  
                    rhyme_correct = False
                    for j in range(min(k, len(all_candidate_rhymes[i]))):
                        pred_rhymes = all_candidate_rhymes[i][j]
                        if pred_rhymes and true_rhymes[i] and any(pr in true_rhymes[i] for pr in pred_rhymes):
                            rhyme_correct = True
                            break
                    if rhyme_correct:
                        self.metrics[f'rhyme_top{k}'] += 1

        self.metrics['char_total'] += num_masks
        self.metrics['tone_total'] += num_masks
        if self.char2rhyme: 
            self.metrics['rhyme_total'] += num_masks

    def get_results(self):
        results = {}
        for metric, value in self.metrics.items():
            if 'total' not in metric:
                prefix = metric.split('_top')[0]  
                total_metric = f"{prefix}_total"  
                total = self.metrics[total_metric]
                results[metric] = value / total if total > 0 else 0.0
        return results

    def baseline_model_predict(self, model, batch_data, mask_positions):
        input_ids = batch_data['input_ids'].to(model.device)
        attention_mask = batch_data['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        batch_preds = []
        for i, positions in enumerate(mask_positions):
            sample_preds = []
            for pos in positions:
                if pos >= outputs.logits.shape[1]:
                    continue
                # topk_ids = torch.topk(outputs.logits[i, pos], self.max_k).indices
                topk_ids = torch.topk(outputs.logits[i, pos], self.max_k).indices.tolist()  # 转为Python列表
                sample_preds.append({'position': pos, 'topk_tokens': topk_ids})
            batch_preds.append(sample_preds)
        return batch_preds


class DualModelInferenceRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_resources()
        self._load_models()

    def _init_resources(self):
        self.id2char = self._load_vocab(self.args.vocab_path)
        # print(self.id2char)
        self.char2tone = self._build_tone_mapping()
        self.char2rhyme = self._build_rhyme_mapping()  
        # print(self.char2tone)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.base_model)
        self.test_loader = self._create_dataloader()

    def _load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: token for idx, token in enumerate(tokens)}

    def _build_tone_mapping(self):
        tone_df = pd.read_excel(self.args.rhyme_table)
        char2tone = {}
        tone2pz = {"上平": 0, "下平": 0, "上声": 1, "去声": 1, "入声": 1}
        for _, row in tone_df.iterrows():
            char = row['char']
            tone = row['tone']
            if char not in char2tone.keys():
                char2tone[char] = []
                char2tone[char].append(tone2pz[tone])
            else:
                if tone2pz[tone] not in char2tone[char]:
                    char2tone[char].append(tone2pz[tone])
        char2tone_final = {}
        for k, v in char2tone.items():
            if len(list(set(v))) > 1:
                char2tone_final[k] = 2
            if len(list(set(v))) == 1:
                if v[0] == 1:
                    char2tone_final[k] = 1
                else:
                    char2tone_final[k] = 0
        return char2tone_final

    def _build_rhyme_mapping(self):
        tone_df = pd.read_excel(self.args.rhyme_table)
        rhyme_list = tone_df['rhyme'].unique().tolist()
        rhyme_list.append('未知')
        rhyme2id = {rhyme: idx for idx, rhyme in enumerate(rhyme_list)}

        char2rhyme = {}
        for _, row in tone_df.iterrows():
            char = row['char']
            rhyme = row['rhyme']
            rhyme_id = rhyme2id[rhyme]

            if char not in char2rhyme:
                char2rhyme[char] = []

            if rhyme_id not in char2rhyme[char]:
                char2rhyme[char].append(rhyme_id)

        return char2rhyme

    def _create_dataloader(self):
        return DataLoader(
            dataset=PoetryDataset(
                self.args.test_path,
                self.args.vocab_path,
                self.args.rhyme_table,
                self.args.max_len
            ),
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )

    def _load_models(self):
        self.models = {
            "proposed": MultiTaskBERT.from_pretrained(
                self.args.model_dir,
                use_mlm=self.args.use_mlm,
                use_tone=self.args.use_tone,
                use_rhyme=self.args.use_rhyme,
                shared_layers=self.args.shared_layers,
            ).to(self.device),
            "baseline": BertForMaskedLM.from_pretrained(self.args.base_model).to(self.device)
        }

    def _evaluate_model(self, model, model_type):
        evaluator = EnhancedToneAwareEvaluator(
            self.args.topk_list,
            self.id2char,
            self.char2tone,
            char2rhyme=self.char2rhyme,
            bert_tokenizer=self.bert_tokenizer if model_type == 'baseline' else None
        )

        model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Evaluating {model_type}"):
                if model_type == 'proposed':
                    self._process_proposed_model(model, batch, evaluator, self.args)
                else:
                    self._process_baseline_model(model, batch, evaluator)
        return evaluator.get_results()

    def _process_proposed_model(self, model, batch, evaluator, args):
        inputs = {
            'input_ids': batch['input_ids'].to(self.device, dtype=torch.long),
            'attention_mask': batch['attention_mask'].to(self.device, dtype=torch.long),
            'input_labels': batch['input_labels'].to(self.device, dtype=torch.long),
            'tone_ids': batch['tone_ids'].to(self.device, dtype=torch.long) if args.use_tone else None,
            'rhyme_ids': batch['rhyme_ids'].to(self.device, dtype=torch.float32) if args.use_rhyme else None,
            'rhyme_label': batch['rhyme_label'].to(self.device, dtype=torch.long) if args.use_rhyme else None,
            'tone_labels': batch['tone_labels'].to(self.device, dtype=torch.long) if args.use_tone else None
        }
        outputs = model(**inputs)
        evaluator.update(
            outputs['mlm_logits'],
            batch['input_labels'],
            batch['tone_labels'].to(self.device),
            'proposed'
        )

    def _process_baseline_model(self, model, batch, evaluator):
        input_ids, attention_mask, labels, mask_pos = self._reconstruct_input(batch)

        tone_labels = torch.full_like(labels, -100)
        
        valid_mask = (labels != -100)
        
        for i in range(labels.size(0)): 
            for j in range(labels.size(1)):  
                if labels[i, j] != -100:
                    char = self.bert_tokenizer.decode([labels[i, j].item()])
                    tone = self.char2tone.get(char, 3)  
                    tone_labels[i, j] = tone

        batch_data = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'input_labels': labels.to(self.device),
            'tone_labels': tone_labels.to(self.device)
        }

        preds = evaluator.baseline_model_predict(model, batch_data, mask_pos)
        self._update_baseline_metrics(preds, batch_data, evaluator)

    def _reconstruct_input(self, batch):
        processed_batch = []
        mask_labels_batch = []
        id2char = self.id2char
        pad_idx = self.bert_tokenizer.pad_token_id
        mask_token_id = self.bert_tokenizer.mask_token_id

        for seq_ids, seq_labels in zip(batch['input_ids'], batch['input_labels']):
            valid_mask = (seq_ids != pad_idx) & (seq_ids != 0) & (seq_ids != -100)
            seq_ids = seq_ids[valid_mask]
            seq_labels = seq_labels[valid_mask]
            tokens = []
            mask_labels = []
            for token_id, label in zip(seq_ids, seq_labels):
                if label != -100:
                    tokens.append('[MASK]')
                    ours_char = id2char.get(label.item(), '[UNK]')
                    bert_id = self.bert_tokenizer.convert_tokens_to_ids(ours_char)
                    mask_labels.append(bert_id)
                else:
                    tokens.append(id2char[token_id.item()])

            processed_batch.append(''.join(tokens))
            mask_labels_batch.append(mask_labels)
        encoded = self.bert_tokenizer(
            processed_batch,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        mask_positions = []
        for input_ids in encoded.input_ids:
            positions = [i for i, token_id in enumerate(input_ids) if token_id == mask_token_id]
            mask_positions.append(positions)

        mask_labels_tensor = torch.full_like(encoded.input_ids, -100)
        for i in range(len(mask_positions)):
            sample_positions = mask_positions[i]  
            sample_labels = mask_labels_batch[i]  
            if len(sample_positions) != len(sample_labels):
                print(processed_batch[i])
                raise ValueError(f"样本{i}的mask位置数({len(sample_positions)})与标签数({len(sample_labels)})不匹配")
            for pos, label in zip(sample_positions, sample_labels):
                mask_labels_tensor[i, pos] = label  
        # print('input_ids.shape',encoded.input_ids.shape)
        # print('attention_mask.shape',encoded.attention_mask.shape)
        # print('mask_labels',mask_labels_tensor.shape)
        return encoded.input_ids, encoded.attention_mask, mask_labels_tensor, mask_positions  

    def _convert_bert_id_to_custom(self, bert_id):
        bert_token = self.bert_tokenizer.convert_ids_to_tokens(bert_id)
        return list(self.id2char.keys())[
            list(self.id2char.values()).index(bert_token)] if bert_token in self.id2char.values() else -1
    def _update_baseline_metrics(self, preds, batch, evaluator):
        batch_size = len(preds)
        for i in range(batch_size):
            sample_preds = preds[i]
            true_labels = batch['input_labels'][i]
            true_tones = batch['tone_labels'][i]

            valid_mask = (true_labels != -100) & (true_tones != -100)  
            true_labels = true_labels[valid_mask]
            true_tones = true_tones[valid_mask]

            for k in evaluator.topk_list:
                char_correct = 0
                tone_correct = 0
                rhyme_correct = 0

                for pred, true_label, true_tone in zip(sample_preds, true_labels, true_tones):
                    topk_tokens = pred['topk_tokens'][:k]
                    
                    true_char = self.bert_tokenizer.decode([true_label.item()])
                    
                    pred_chars = [self.bert_tokenizer.decode([token_id]) for token_id in topk_tokens]
                    
                    char_correct += (true_char in pred_chars)
                    
                    pred_tones = [evaluator.char2tone.get(char, 3) for char in pred_chars]
                    tone_match = true_tone.item() in pred_tones
                    tone_correct += tone_match
                    
                    if evaluator.char2rhyme:  
                        true_rhymes = evaluator.char2rhyme.get(true_char, [])
                        
                        rhyme_match = False
                        for char in pred_chars:
                            pred_rhymes = evaluator.char2rhyme.get(char, [])
                            if pred_rhymes and true_rhymes and any(pr in true_rhymes for pr in pred_rhymes):
                                rhyme_match = True
                                break
                        
                        rhyme_correct += rhyme_match

                evaluator.metrics[f'char_top{k}'] += char_correct
                evaluator.metrics[f'tone_top{k}'] += tone_correct
                if evaluator.char2rhyme: 
                    evaluator.metrics[f'rhyme_top{k}'] += rhyme_correct

            total_valid = len(true_labels)
            evaluator.metrics['char_total'] += total_valid
            evaluator.metrics['tone_total'] += total_valid
            if evaluator.char2rhyme:  
                evaluator.metrics['rhyme_total'] += total_valid

    def generate_comparison_report(self, args):
        results = {}
        for model_name in ['proposed', 'baseline']:
            results[model_name] = self._evaluate_model(self.models[model_name], model_name)

        df = pd.DataFrame(results).T
        df = df.applymap(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
        md_path = os.path.join(args.comparison_report_dir, f"{str(args.model_dir).split('/')[-2]}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(tablefmt="pipe"))
        print(f"Markdown文件已保存至: {md_path}")
        print("\n模型对比结果:")
        print(df.to_markdown(tablefmt="grid"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="./datasets/test/test.json")  #Old
    parser.add_argument("--model_dir", default="./checkpoints/mlm_tone/best_model")  #mlm_tone_rhyme
    parser.add_argument("--base_model", default=r"./poem_bert")
    parser.add_argument("--vocab_path", default="./bert-base-chinese/vocab.txt")
    parser.add_argument("--comparison_report_dir", default="./results/all_testdata")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rhyme_table", default="./datasets/tone_df.xlsx")
    parser.add_argument("--topk_list", type=int, nargs='+', default=[1, 3, 5, 10, 30, 50, 100])  #
    parser.add_argument("--use_mlm", action='store_true', default=True)
    parser.add_argument("--use_tone", action='store_true', default=True)
    parser.add_argument("--use_rhyme", action='store_true', default=False)  #False
    parser.add_argument("--shared_layers", type=int, default=6,
                        help="共享投影层的堆叠层数")
    parser.add_argument("--cross_attn_heads", type=int, default=6,
                        help="跨任务注意力头数")
    return parser.parse_args()


if __name__ == "__main__":
    runner = DualModelInferenceRunner(get_args())
    runner.generate_comparison_report(get_args())  # 添加执行入口

