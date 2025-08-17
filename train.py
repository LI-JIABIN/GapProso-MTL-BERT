# -*- coding:utf-8 -*-
"""
作者：THUNDEROBOT-LI
日期：2025年05月26日
"""
# train.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
import glob
import json
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import MultiTaskBERT
from data_utils import PoetryDataset, collate_fn
import math



class TrainingState:
    def __init__(self, total_epochs, train_files):
        self.epoch = 0
        self.file_index = 0  
        self.total_epochs = total_epochs
        self.train_files = train_files  
        self.best_metric = 0.0
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.step_count = 0  
        self.active_tasks = ['mlm', 'tone', 'rhyme']  
        self.current_task_idx = 0  
        self.task_no_improve = {'mlm': 0, 'tone': 0, 'rhyme': 0}  
        self.task_best_metrics = {'mlm': 0.0, 'tone': 0.0, 'rhyme': 0.0}  
        self.task_max_epochs = {'mlm': total_epochs, 'tone': 5, 'rhyme': 5}  

    def save(self, output_dir, model, optimizer, scheduler):
        os.makedirs(output_dir, exist_ok=True)  
        state = {
            'epoch': self.epoch,
            'file_index': self.file_index,
            'best_metric': self.best_metric,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'active_tasks': self.active_tasks,
            'current_task_idx': self.current_task_idx,
            'task_no_improve': self.task_no_improve,
            'task_best_metrics': self.task_best_metrics,
            'task_max_epochs': self.task_max_epochs
        }
        torch.save(state, os.path.join(output_dir, f"training_state_{self.timestamp}.pt"))
        print(f" 训练状态已保存至 {output_dir}")

    @classmethod
    def load(cls, checkpoint_path, total_epochs, train_files):
        state = torch.load(checkpoint_path)
        instance = cls(total_epochs, train_files)
        instance.epoch = state['epoch']
        instance.file_index = state['file_index']
        instance.best_metric = state['best_metric']
        instance.timestamp = time.strftime("%Y%m%d-%H%M%S")  
        if 'active_tasks' in state:
            instance.active_tasks = state['active_tasks']
        if 'current_task_idx' in state:
            instance.current_task_idx = state['current_task_idx']
        if 'task_no_improve' in state:
            instance.task_no_improve = state['task_no_improve']
        if 'task_best_metrics' in state:
            instance.task_best_metrics = state['task_best_metrics']
        if 'task_max_epochs' in state:
            instance.task_max_epochs = state['task_max_epochs']
        return instance, state

    def get_current_task(self):
        if not self.active_tasks:
            return None  
        return self.active_tasks[self.current_task_idx % len(self.active_tasks)]
    
    def rotate_task(self):
        if self.active_tasks:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.active_tasks)
        
    def update_task_metrics(self, task, metric_value):
        if task not in self.active_tasks:
            return False
            
        if metric_value > self.task_best_metrics[task]:
            self.task_best_metrics[task] = metric_value
            self.task_no_improve[task] = 0
            return True
        else:
            self.task_no_improve[task] += 1
            if self.task_no_improve[task] >= 3:
                self.active_tasks.remove(task)
                print(f" 任务 {task} 连续3次未提升，停止训练该任务")
                if self.active_tasks:
                    self.current_task_idx = self.current_task_idx % len(self.active_tasks)
                return False
            return False
            
    def update_epoch_based_tasks(self):
        for task, max_epoch in self.task_max_epochs.items():
            if self.epoch >= max_epoch and task != 'mlm' and task in self.active_tasks:
                self.active_tasks.remove(task)
                print(f" 任务 {task} 已达到最大训练epoch {max_epoch}，停止训练该任务")
                
        if not self.active_tasks and 'mlm' not in self.active_tasks:
            self.active_tasks.append('mlm')
            print(" 所有任务都已达到最大epoch，恢复mlm任务以继续训练")
            
        if self.active_tasks:
            self.current_task_idx = self.current_task_idx % len(self.active_tasks)


class MetricTracker:
    def __init__(self, use_mlm, use_tone, use_rhyme):
        self.task_status = {
            'mlm': use_mlm,
            'tone': use_tone,
            'rhyme': use_rhyme
        }
        self.reset()

    def reset(self):
        self.metrics = {
            'mlm_correct': 0, 'mlm_total': 0,
            'tone_correct': 0, 'tone_total': 0,
            'rhyme_correct': 0, 'rhyme_total': 0,
            'total_loss': 0, 'step_count': 0
        }

    def update(self, outputs, labels):
        self.metrics['step_count'] += 1
        self.metrics['total_loss'] += outputs['loss'].item() if outputs['loss'] else 0

        if self.task_status['mlm'] and 'mlm_logits' in outputs and labels['mlm'] is not None:
            mask_pos = (labels['mlm'] != -100)
            preds = outputs['mlm_logits'].argmax(-1)
            self.metrics['mlm_correct'] += preds[mask_pos].eq(labels['mlm'][mask_pos]).sum().item()
            self.metrics['mlm_total'] += mask_pos.sum().item()

        if self.task_status['tone'] and 'tone_logits' in outputs and labels['tone'] is not None:
            tone_mask = (labels['tone'] != -100)
            self.metrics['tone_correct'] += outputs['tone_logits'].argmax(-1)[tone_mask].eq(
                labels['tone'][tone_mask]).sum().item()
            self.metrics['tone_total'] += tone_mask.sum().item()

        if self.task_status['rhyme'] and 'rhyme_logits' in outputs and labels['rhyme'] is not None:
            rhyme_preds = outputs['rhyme_logits'].argmax(-1)  # [B]
            rhyme_labels = labels['rhyme']  # 确保这里已经是[B]

            if rhyme_labels.dim() > 1:
                rhyme_labels = rhyme_labels.squeeze()

            self.metrics['rhyme_correct'] += rhyme_preds.eq(rhyme_labels).sum().item()
            self.metrics['rhyme_total'] += rhyme_labels.size(0)

    def get_metrics(self):
        metrics = {'loss': self.metrics['total_loss'] / self.metrics['step_count']}
        for task in ['mlm', 'tone', 'rhyme']:
            if self.task_status[task]:
                correct = self.metrics[f'{task}_correct']
                total = self.metrics[f'{task}_total'] or 1e-6
                metrics[f'{task}_acc'] = correct / total
        return metrics


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取所有训练文件
    if os.path.isdir(args.train_dir):
        train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.json")))
    else:
        train_files = args.train_dir.split(',')
    print(f" 共发现 {len(train_files)} 个训练文件")

    valid_set = PoetryDataset(args.valid_path, args.vocab_path, args.rhyme_table, args.max_len)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

    model = MultiTaskBERT.from_pretrained(
        args.bert_name,
        use_mlm=args.use_mlm,
        use_tone=args.use_tone,
        use_rhyme=args.use_rhyme,
        shared_layers=args.shared_layers,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    model.to(device)

    optimizer = AdamW([{"params": model.parameters(), "lr": args.lr}], weight_decay=0.01)
    total_steps = math.ceil(751574 / args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    print("\n=== 参数可训练状态 ===")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"[可训练] {name}")
        else:
            print(f"[冻结] {name}")
        total_params += param.numel()

    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params:.1%})")
    
    training_state = TrainingState(args.epochs, train_files)
    active_tasks = []
    if args.use_mlm:
        active_tasks.append('mlm')
    if args.use_tone:
        active_tasks.append('tone')
    if args.use_rhyme:
        active_tasks.append('rhyme')
    training_state.active_tasks = active_tasks
    
    if args.resume:
        try:
            training_state, state = TrainingState.load(args.resume, args.epochs, train_files)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            scheduler.load_state_dict(state['scheduler_state'])
            print(f" 恢复训练状态：Epoch {training_state.epoch + 1}，"
                  f"文件 {train_files[training_state.file_index]}，"
                  f"步数 {training_state.step_count}，"
                  f"活跃任务 {training_state.active_tasks}")
        except Exception as e:
            print(f" 恢复训练状态失败: {str(e)}")
            return

    no_improve = 0
    best_metric = 0.0
    evaluation_interval = args.eval_steps  

    for epoch in range(training_state.epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        training_state.epoch = epoch
        
        training_state.update_epoch_based_tasks()
        
        if not training_state.active_tasks:
            print("🏁 所有任务都已达到早停条件，训练结束！")
            break
            
        for file_idx in range(training_state.file_index, len(train_files)):
            model.train()  
            current_file = train_files[file_idx]
            print(f"\n 加载训练文件：{os.path.basename(current_file)}")

            try:
                train_set = PoetryDataset(current_file, args.vocab_path, args.rhyme_table, args.max_len)
                train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                          collate_fn=collate_fn, shuffle=True, num_workers=4)
            except Exception as e:
                print(f"文件加载失败: {str(e)}")
                continue
                
            metric = MetricTracker(args.use_mlm, args.use_tone, args.use_rhyme)
            model.train()

            pbar = tqdm(train_loader, desc=f"训练中 Epoch {epoch + 1} File {file_idx + 1}")

            for batch_idx, batch in enumerate(pbar):
                training_state.step_count += 1
                current_step = training_state.step_count
                
                current_task = training_state.get_current_task()
                if not current_task:
                    print(" 所有任务都已达到早停条件，训练结束！")
                    break
                
                inputs = {
                    'input_ids': batch['input_ids'].to(device, dtype=torch.long),
                    'tone_ids': batch['tone_ids'].to(device, dtype=torch.long),
                    'rhyme_ids': batch['rhyme_ids'].to(device, dtype=torch.float32),
                    'attention_mask': batch['attention_mask'].to(device, dtype=torch.long),
                }
                
                if current_task == 'mlm':
                    inputs['input_labels'] = batch['input_labels'].to(device, dtype=torch.long)
                elif current_task == 'tone':
                    # inputs['tone_ids'] = batch['tone_ids'].to(device, dtype=torch.long)
                    inputs['tone_labels'] = batch['tone_labels'].to(device, dtype=torch.long)
                elif current_task == 'rhyme':
                    # inputs['rhyme_ids'] = batch['rhyme_ids'].to(device, dtype=torch.float32)
                    inputs['rhyme_label'] = batch['rhyme_label'].to(device, dtype=torch.long)
                
                if 'rhyme_ids' in inputs:
                    assert inputs['rhyme_ids'].dtype == torch.float32
                assert inputs['input_ids'].dtype == torch.long
                
                optimizer.zero_grad()
                outputs = model(**inputs)
                
                if outputs['loss'] is not None:
                    outputs['loss'].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                
                labels = {
                    'mlm': batch['input_labels'].to(device) if current_task == 'mlm' else None,
                    'tone': batch['tone_labels'].to(device) if current_task == 'tone' else None,
                    'rhyme': batch['rhyme_label'].to(device).squeeze(1) if current_task == 'rhyme' else None
                }
                
                metric.update(outputs, labels)
                
                task_metrics = metric.get_metrics()
                current_task_metric = task_metrics.get(f'{current_task}_acc', 0)
                
                pbar.set_postfix({
                    "Task": current_task,
                    "Loss": f"{task_metrics['loss']:.3f}",
                    "Step": current_step,
                    f"{current_task.upper()}_ACC": f"{current_task_metric:.2%}",
                })
                
                if current_step % evaluation_interval == 0:
                    valid_metrics = evaluate(model, valid_loader, device, args)
                    model.train()  # 新增此行
                    
                    print(f"\n Step {current_step} 验证结果 >> Loss: {valid_metrics['loss']:.3f}")
                    for task in training_state.active_tasks:
                        task_acc = valid_metrics.get(f'{task}_acc', 0)
                        print(f"  - {task.upper()} Acc: {task_acc:.2%}")
                    
                    current_task_acc = valid_metrics.get(f'{current_task}_acc', 0)
                    training_state.update_task_metrics(current_task, current_task_acc)
                    
                    training_state.rotate_task()
                    
                    if not training_state.active_tasks:
                        print(" 所有任务都已达到早停条件，训练结束！")
                        break
                    
                    training_state.save(args.output_dir, model, optimizer, scheduler)
                    
                    current_metric = valid_metrics.get(args.monitor_metric, 0)
                    if current_metric > best_metric:
                        best_metric = current_metric
                        no_improve = 0
                        model.save_pretrained(os.path.join(args.output_dir, "best_model"), safe_serialization=False)
                        print(f" 发现最佳模型 | {args.monitor_metric}: {best_metric:.2%}")
                    else:
                        no_improve += 1
                        print(f" 连续 {no_improve}/{args.early_stop} 次验证未提升")
                    
                    # 全局早停判断
                    if no_improve >= args.early_stop:
                        print(f" 达到全局早停条件，终止训练！最终步数：{current_step}")
                        return
                else:
                    if batch_idx % args.task_rotation_interval == 0:
                        training_state.rotate_task()

            if not training_state.active_tasks:
                break
                
            training_state.file_index = file_idx + 1
            training_state.save(args.output_dir, model, optimizer, scheduler)

            del train_set, train_loader
            torch.cuda.empty_cache()

        if not training_state.active_tasks:
            break
            
        training_state.file_index = 0

        valid_metrics = evaluate(model, valid_loader, device, args)
        model.train()  # 新增此行
        print(f"\n Epoch {epoch + 1} 结束验证 >> Loss: {valid_metrics['loss']:.3f}")
        for task in training_state.active_tasks:
            task_acc = valid_metrics.get(f'{task}_acc', 0)
            print(f"  - {task.upper()} Acc: {task_acc:.2%}")


def evaluate(model, data_loader, device, args):
    model.eval()
    metric = MetricTracker(args.use_mlm, args.use_tone, args.use_rhyme)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'tone_ids': batch['tone_ids'].to(device) if args.use_tone else None,
                'rhyme_ids': batch['rhyme_ids'].to(device) if args.use_rhyme else None,
            }
            labels = {
                'mlm': batch['input_labels'].to(device),
                'tone': batch['tone_labels'].to(device),
                'rhyme': batch['rhyme_label'].to(device).squeeze(1)
            }

            outputs = model(**inputs)
            metric.update(outputs, labels)

    return metric.get_metrics()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=r"./datasets/train", help="训练数据路径，可以是目录或逗号分隔的文件列表")
    parser.add_argument("--resume", type=str, default=r"/root/autodl-tmp/char-predict/checkpoints/mlm_tone_rhyme/training_state_20250613-233223.pt",
                        help="恢复训练检查点路径")  # 新增参数
    parser.add_argument("--valid_path", type=str, default="./datasets/val/val.json")
    parser.add_argument("--vocab_path", type=str,
                        default=r"./bert-base-chinese/vocab.txt")
    parser.add_argument("--rhyme_table", type=str, default="./datasets/tone_df.xlsx")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mlm_tone_rhyme")  #_tone_rhyme

    parser.add_argument("--bert_name", type=str, default=r"./poem_bert")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--tone_dim", type=int, default=32)
    parser.add_argument("--rhyme_dim", type=int, default=32)
    parser.add_argument("--use_mlm", action='store_true', default=True)
    parser.add_argument("--use_tone", action='store_true', default=True)
    parser.add_argument("--use_rhyme", action='store_true', default=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)


    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=1)


    parser.add_argument("--early_stop", type=int, default=3,
                        help="连续多少个step无提升则停止")
    parser.add_argument("--eval_steps", type=int, default=600)
    parser.add_argument("--monitor_metric", type=str, default="mlm_acc",
                        choices=["mlm_acc", "tone_acc", "rhyme_acc"])
    parser.add_argument("--shared_layers", type=int, default=6,
                        help="共享投影层的堆叠层数")
    parser.add_argument("--task_rotation_interval", type=int, default=2,
                        help="每隔多少个batch轮换一次任务")
    return parser.parse_args()


if __name__ == "__main__":
    train()