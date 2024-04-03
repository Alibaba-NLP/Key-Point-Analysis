import argparse
import json
import os
from dataclasses import dataclass
from typing import List
import random

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer, T5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq


@dataclass(frozen=True)
class Feature:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class SftDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_len, ):
        super().__init__()
        source_texts, target_texts = self.load_data(data_file)
        self.features = self.tokenize(tokenizer=tokenizer, max_seq_len=max_seq_len, source_texts=source_texts, target_texts=target_texts)

    def load_data(self, data_file):
        datas = json.load(open(data_file, 'r', encoding='utf-8'))
        random.shuffle(datas)

        source_texts = []
        target_texts = []

        for data in datas:
            # if data["output"].startswith('No.'):
            #    continue

            source_texts.append(f'{data["input"]}')
            target_texts.append(f'{data["output"]}')

        return source_texts, target_texts

    def tokenize(self, tokenizer: PreTrainedTokenizer, max_seq_len, source_texts: List[str], target_texts: List[str]):
        inputs = tokenizer.__call__(source_texts, truncation=True, max_length=max_seq_len)
        labels = tokenizer.__call__(target_texts, truncation=True, max_length=max_seq_len).input_ids
        features = []
        for i in range(len(labels)):
            features.append(Feature(input_ids=inputs.input_ids[i], attention_mask=inputs.attention_mask[i], labels=labels[i]))
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index].__dict__


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--model_path', type=str, default='../models/flan-t5-xl')
parser.add_argument('--seed', type=int, default=20217)
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--report_steps', type=int, default=50)
parser.add_argument('--eval_steps', type=int, default=200)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--deepspeed', type=str)
parser.add_argument('--local_rank', type=int)
parser.add_argument('--warmup_ratio', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()

model = T5ForConditionalGeneration.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
train_set = SftDataset(data_file=args.input_file, tokenizer=tokenizer, max_seq_len=args.max_seq_len, )

os.makedirs(args.output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="no",
    learning_rate=args.lr,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    save_strategy='epoch',
    bf16=args.bf16,
    fp16=args.fp16,
    dataloader_pin_memory=False,
    dataloader_num_workers=4,
    logging_steps=args.report_steps,
    label_smoothing_factor=args.label_smoothing,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, max_length=args.max_seq_len, return_tensors='pt'),
)

if __name__ == '__main__':
    trainer.train()
