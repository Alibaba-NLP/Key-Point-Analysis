import math

import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer


class DataLoaderKPA:
    def __init__(self, ids, queries, tokenizer: PreTrainedTokenizer, batch_size, max_length=2048):
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=max_length, return_tensors='pt')
        self.ids = ids
        self.inputs = tokenizer.__call__(queries, max_length=max_length, truncation=True).input_ids

        truncation_count = len(list(filter(lambda x: len(x) == max_length, self.inputs)))
        print(f'{truncation_count=}')

        self.batch_size = batch_size
        self.total_steps = math.ceil(len(self.ids) / self.batch_size)

    def __len__(self):
        return self.total_steps

    def __call__(self):
        for step_idx in range(len(self)):
            inputs = [{'input_ids': inputs} for inputs in self.inputs[step_idx * self.batch_size:(step_idx + 1) * self.batch_size]]
            inputs = self.data_collator(inputs)
            yield self.ids[step_idx * self.batch_size:(step_idx + 1) * self.batch_size], inputs


def generate(model, tokenizer: PreTrainedTokenizer, ids, datas, batch_size, max_new_tokens, max_length):
    # only for T5/FlanT5 model
    model.eval()
    outputs = {}
    yes_token_id = tokenizer.convert_tokens_to_ids('▁Yes')
    no_token_id = tokenizer.convert_tokens_to_ids('▁No')
    dataloader = DataLoaderKPA(ids=ids, queries=datas, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)
    for item_ids, batch in tqdm(dataloader(), total=len(dataloader)):
        batch = batch.to(model.device)
        output = model.generate(**batch, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True)
        output_str = tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        first_output_token_scores = output.scores[0]

        # scores for token 'Yes/No'
        yes_token_scores_original = first_output_token_scores[:, yes_token_id]
        no_token_scores_original = first_output_token_scores[:, no_token_id]

        scores = torch.cat([yes_token_scores_original.unsqueeze(-1), no_token_scores_original.unsqueeze(-1)], dim=-1)  # batch_size * 2
        raw_scores = scores.detach().cpu().numpy().tolist()
        scores = torch.softmax(scores, dim=-1)  # batch_size * 2
        yes_token_scores = scores[:, 0].detach().cpu().numpy().tolist()

        max_token_idx = first_output_token_scores.argmax(dim=-1).detach().cpu().numpy().tolist()
        max_token = tokenizer.convert_ids_to_tokens(max_token_idx)

        outputs |= dict(zip(item_ids, [{'output': o, 'confidence_score': nts, 'max token': mt, 'raw_scores': rs} for o, nts, mt, rs in
                                       zip(output_str, yes_token_scores, max_token, raw_scores)]))
    return outputs
