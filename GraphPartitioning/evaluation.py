import argparse
import gc
import json
import os
import random

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from generate import generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cluster_model', type=str, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=20217)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()
    return args


def main(args):
    random.seed(args.seed)
    model, tokenizer = prepare_model_flant5(args)

    datas = json.load(open(args.test_file, 'r', encoding='utf-8'))

    prompts = []
    answers = []
    for data in datas:
        answer = data['output']
        prompt = data['input']
        prompts.append(prompt)
        answers.append(answer)

    print(f'sample prompt: {prompts[0]}\nsample answer: {answers[0]}')

    outputs = generate(model=model, tokenizer=tokenizer, ids=prompts, datas=prompts, max_new_tokens=64, batch_size=args.batch_size, max_length=args.max_length, )

    raw_outputs = [outputs[k]['output'] for k in prompts]
    confidence_score = [outputs[k]['confidence_score'] for k in prompts]
    max_tokens = [outputs[k]['max token'] for k in prompts]
    raw_scores = [outputs[k]['raw_scores'] for k in prompts]

    processed_outputs = list(map(lambda x: x.split('\n')[0], raw_outputs))

    for idx in random.sample(list(range(len(prompts))), 50):
        print(
            f'prompt: {prompts[idx]}\nanswer: {answers[idx]}\npred: {processed_outputs[idx]}\nconfidence_score: {confidence_score[idx]}\nmax token: {max_tokens[idx]}\nraw scores: {raw_scores[idx]}\n\n')

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    with open(args.output_file, 'w', encoding='utf-8') as output:
        datas = [
            {'prompt': _prompt, 'answer': _answer, 'pred': _pred, 'confidence_score': _confidence_score, 'max token': _max_token, 'raw_pred': _raw_pred, 'raw_scores': _raw_score}
            for _prompt, _answer, _pred, _confidence_score, _max_token, _raw_pred, _raw_score in
            zip(prompts, answers, processed_outputs, confidence_score, max_tokens, raw_outputs, raw_scores)]
        json.dump(datas, output, ensure_ascii=False, indent=4)


def prepare_model_flant5(args):
    os.system(f'cp /path/to/flan-t5-large/sp* {args.model_path}')
    os.system(f'cp /path/to/flan-t5-large/tokenize* {args.model_path}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).cuda()
    return model, tokenizer


if __name__ == '__main__':
    args = parse_args()
    print(f'{args=}')
    main(args)
