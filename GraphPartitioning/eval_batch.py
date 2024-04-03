import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--test_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

args.dev_file = args.test_file.replace('instructions_test', 'instructions_dev')

os.makedirs(args.output_dir, exist_ok=True)

os.makedirs(f'{args.output_dir}/test', exist_ok=True)
for ckp in os.listdir(args.model_dir):
    if not ckp.startswith('checkpoint-'):
        continue
    model_base_name = list(filter(lambda x: x.strip() != '', args.model_dir.split('/')))[-1]
    model_path = os.path.join(args.model_dir, ckp)
    cmd = f'python evaluation.py ' \
          f'--model_path {model_path} ' \
          f'--output_file {args.output_dir}/test/eval_{model_base_name}_{ckp}.json ' \
          f'--batch_size {args.batch_size} ' \
          '--max_length 512 ' \
          f'--test_file {args.test_file}'
    os.system(cmd)

os.makedirs(f'{args.output_dir}/dev', exist_ok=True)
for ckp in os.listdir(args.model_dir):
    if not ckp.startswith('checkpoint-'):
        continue
    model_base_name = list(filter(lambda x: x.strip() != '', args.model_dir.split('/')))[-1]
    model_path = os.path.join(args.model_dir, ckp)
    cmd = f'python evaluation.py ' \
          f'--model_path {model_path} ' \
          f'--output_file {args.output_dir}/dev/eval_{model_base_name}_{ckp}.json ' \
          f'--batch_size {args.batch_size} ' \
          '--max_length 512 ' \
          f'--test_file {args.dev_file}'
    os.system(cmd)
