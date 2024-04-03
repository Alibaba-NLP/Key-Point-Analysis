import argparse
import gc
import json
import math
import os
import random

import networkx as nx
import numpy as np
import torch.cuda
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bleurt import bleurt
from utils_clustering import embs_clustering
from rouge_score import rouge_scorer


def decompose_prompt(prompt):
    s = prompt.split('|')
    topic = s[0].strip()
    argument_a = s[1].split('.', 1)[1].strip()
    argument_b = s[2].split('.', 1)[1].strip()
    return topic, argument_a, argument_b


def graph_split_by_embs_clustering(graph: nx.Graph, embs_clustering_model, init_subgraph_count, seed=42):
    cluster_algorithm = 'kmeans'

    arguments = list(set(map(lambda x: x[1]['argument'], graph.nodes(data=True))))

    _, arg_clusters, _, _ = embs_clustering(model=embs_clustering_model, sentences=arguments, cluster_algorithm=cluster_algorithm, kmeans_n_clusters=init_subgraph_count, seed=seed)

    subgraphs = []

    for cluster in arg_clusters:
        nodes = list(filter(lambda x: graph.nodes[x]['argument'] in cluster, list(graph.nodes())))
        subgraphs.append(graph.subgraph(nodes).copy())

    return subgraphs


def clustering_soft(graph: nx.Graph, embs_clustering_model, init_subgraph_count=11, max_loop_count=100, seed=42):
    random.seed(seed)


    subgraphs = graph_split_by_embs_clustering(graph, embs_clustering_model, init_subgraph_count=init_subgraph_count, seed=seed)
    subgraphs = list(filter(lambda x: len(x) != 1, subgraphs))

    def graph_weight(g: nx.Graph):
        weights = [x[2]['weight'] for x in g.edges(data=True) if not x[2]['kp'].startswith('No')]
        return sum(weights) / len(g.edges) if len(g.edges) != 0 else 0

    move_count = 0
    for _ in tqdm(range(max_loop_count), total=max_loop_count, desc='running...'):
        if len(subgraphs) == 0:
            break
        if max(list(map(lambda x: len(x.nodes()), subgraphs))) == 1:
            break
        graph_idx_to_move_out_node = random.randint(0, len(subgraphs) - 1)
        while len(subgraphs[graph_idx_to_move_out_node].nodes()) == 1:
            graph_idx_to_move_out_node = random.randint(0, len(subgraphs) - 1)

        weight_change = []
        subgraph_to_move_out_node = subgraphs[graph_idx_to_move_out_node]

        subgraph_weight_to_move_out_node = graph_weight(subgraph_to_move_out_node)
        to_move_node_idx = random.randint(0, len(subgraph_to_move_out_node) - 1)
        to_move_node = list(subgraph_to_move_out_node.nodes())[to_move_node_idx]

        nodes_after_move_out_nodes = list(subgraph_to_move_out_node.nodes())
        del nodes_after_move_out_nodes[to_move_node_idx]

        subgraph_after_move_out_node = graph.subgraph(nodes_after_move_out_nodes).copy()
        subgraph_weight_after_move_out_node = graph_weight(subgraph_after_move_out_node)

        for subgraph_idx, subgraph in enumerate(subgraphs):
            if subgraph_idx == graph_idx_to_move_out_node:
                weight_change.append(0)
                continue
            if subgraph.has_node(to_move_node):
                weight_change.append(0)
                continue

            subgraph_after_move_in_node = graph.subgraph(list(subgraph.nodes()) + [to_move_node]).copy()
            subgraph_weight_after_move_in_node = graph_weight(subgraph_after_move_in_node)

            if subgraph_after_move_in_node.number_of_edges() == subgraph.number_of_edges():
                # to_move_node has no edges with all nodes in subgraph
                weight_change.append(0)
                continue

            weight_change.append(subgraph_weight_after_move_in_node - graph_weight(subgraph))

        if max(weight_change) <= 0:
            continue
        max_graph_idx = weight_change.index(max(weight_change))

        subgraph_after_move_in_node = graph.subgraph(list(subgraphs[max_graph_idx].nodes()) + [to_move_node]).copy()

        subgraphs.append(subgraph_after_move_in_node)
        subgraphs[max_graph_idx] = None

        if subgraph_weight_after_move_out_node - subgraph_weight_to_move_out_node > -args.th:
            subgraphs.append(subgraph_after_move_out_node)
            subgraphs[graph_idx_to_move_out_node] = None

        subgraphs = list(filter(lambda x: x is not None, subgraphs))

        move_count += 1

    subgraphs = list(filter(lambda x: len(x) > 1, subgraphs))
    print(f'{move_count=}')
    return subgraphs


def load_kp_pair_score_map():
    filename = '.bleurt_kp_pair_score_map.json'
    print(f'==={filename=}===')
    retry_count = 0
    try:
        print(f'load from {filename}')
        if os.path.exists(filename):
            return json.load(open(filename, 'r', encoding='utf-8'))
    except Exception as e:
        retry_count += 1
        os.system('sleep 5')
        if retry_count > 5:
            return {}
    return {}


def dump_kp_pair_score_map(kp_pair_score_map):
    m = load_kp_pair_score_map()
    if len(m) == len(kp_pair_score_map):
        return

    filename = '.bleurt_kp_pair_score_map.json'
    print(f'==={filename=}===')
    with open(filename, 'w', encoding='utf-8') as output:
        json.dump(kp_pair_score_map, output, ensure_ascii=False, indent=4)


def get_topic_argskp_map(p):
    datas = json.load(open(p))
    s = {}
    for line in datas:
        topic = line['topic']
        if topic[-1] != '.':
            topic += '.'
        if topic not in s:
            s[topic] = {}
        args = line['argument']
        if args not in s[topic]:
            s[topic][args] = []
        for kp in line['kps']:
            s[topic][args].append(kp['key_point'])
    return s


def graph_clustering(outputs, embs_clustering_model):
    topic_argskp_map = get_topic_argskp_map(f'../Datasets/{args.dataset}Data/{args.split}.json')

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    topic_graph_map = {}
    topic_subgraphs_map = {}
    topic_arguments_map = {}

    all_rouge1, all_rouge2, all_rougel = 0, 0, 0
    all_sp_avgw, all_sr_avgw, all_sf1_avgw = 0, 0, 0

    kp_pair_score_map = load_kp_pair_score_map()

    for data in outputs:
        prompt = data['prompt']
        topic, argument_a, argument_b = decompose_prompt(prompt)
        if topic not in topic_graph_map:
            topic_graph_map[topic] = nx.Graph()
            topic_arguments_map[topic] = []
        if argument_a not in topic_arguments_map[topic]:
            topic_arguments_map[topic].append(argument_a)
        if argument_b not in topic_arguments_map[topic]:
            topic_arguments_map[topic].append(argument_b)

        node_idx_a = topic_arguments_map[topic].index(argument_a)
        node_idx_b = topic_arguments_map[topic].index(argument_b)

        if not topic_graph_map[topic].has_node(node_idx_a):
            topic_graph_map[topic].add_node(node_idx_a, argument=argument_a)
        if not topic_graph_map[topic].has_node(node_idx_b):
            topic_graph_map[topic].add_node(node_idx_b, argument=argument_b)

        if not topic_graph_map[topic].has_edge(node_idx_a, node_idx_b):
            scores = data['raw_scores']
            weight = math.e ** (scores[0] / args.t) / (math.e ** (scores[0] / args.t) + math.e ** (scores[1] / args.t))

            topic_graph_map[topic].add_edge(node_idx_a, node_idx_b,
                                            weight=weight,
                                            kp=data['pred'].replace('Yes.', '', 1).strip(),
                                            answer=data['answer'].replace('Yes.', '', 1).strip(), )

    for topic in topic_graph_map:
        answer_set = list(set(sum(topic_argskp_map[topic].values(), [])))

        print('-' * 20)
        print(topic)
        print(f'{len(topic_graph_map[topic])=}')

        graph = topic_graph_map[topic]
        n_nodes_before_clustering = len(graph.nodes())

        topic_subgraphs_map[topic] = clustering_soft(graph=graph,
                                                     init_subgraph_count=len(answer_set),
                                                     max_loop_count=args.max_loop_count,
                                                     embs_clustering_model=embs_clustering_model,
                                                     seed=args.seed)
        n_nodes_after_clustering = sum(list(map(lambda x: len(x.nodes()), topic_subgraphs_map[topic])))
        print(f'topic: {topic}\ndropped nodes count:{n_nodes_before_clustering - n_nodes_after_clustering}')

    if embs_clustering_model is not None:
        embs_clustering_model.cpu()
        del embs_clustering_model
        gc.collect()
        torch.cuda.empty_cache()

    count = 0
    argument_div_kps = 0
    for topic in topic_graph_map:
        kp_avg_weight_map = []
        for subgraph in topic_subgraphs_map[topic]:
            this_kp_avg_weight_map = {}
            for _, _, d in subgraph.edges(data=True):
                kp = d['kp']
                weight = d['weight']
                if kp not in this_kp_avg_weight_map:
                    this_kp_avg_weight_map[kp] = []
                this_kp_avg_weight_map[kp].append(weight)
            for kp in this_kp_avg_weight_map:
                this_kp_avg_weight_map[kp] = sum(this_kp_avg_weight_map[kp]) / len(this_kp_avg_weight_map[kp])
            kp_avg_weight_map.append(this_kp_avg_weight_map)


        kps_set_avgw = set()
        for idx, graph in enumerate(sorted(topic_subgraphs_map[topic], key=lambda x: x.number_of_nodes())):
            edges_wo_no = list(filter(lambda x: x[2]['kp'] != 'No.', list(graph.edges(data=True))))
            if len(edges_wo_no) == 0:
                continue
            kps_sorted = list(map(lambda x: x[2]['kp'], sorted(edges_wo_no, key=lambda x: x[2]['weight'], reverse=True)))
            for kp in kps_sorted:
                if kp not in kps_set_avgw:
                    kps_set_avgw.add(kp)
                    break

        answer_set = list(set(sum(topic_argskp_map[topic].values(), [])))
        kps_set_avgw = list(kps_set_avgw)
        if len(answer_set) == 0:
            print(f'error! empty answer_set, topic: {topic}')
            continue
        sp_avgw, sr_avgw, sf1_avgw, score_matrix_avgw = cal_final_score_bleurt(answer_set=answer_set, kps_set=kps_set_avgw, kp_pair_score_map=kp_pair_score_map)

        rouge1 = rouge.score(' '.join(kps_set_avgw), ' '.join(answer_set))['rouge1'].fmeasure
        rouge2 = rouge.score(' '.join(kps_set_avgw), ' '.join(answer_set))['rouge2'].fmeasure
        rougel = rouge.score(' '.join(kps_set_avgw), ' '.join(answer_set))['rougeL'].fmeasure

        print(f'kps:')
        print('\nanswers:')
        print('\n'.join(answer_set))

        all_rouge1 += rouge1
        all_rouge2 += rouge2
        all_rougel += rougel

        all_sp_avgw += sp_avgw
        all_sr_avgw += sr_avgw
        all_sf1_avgw += sf1_avgw

        count += 1
        argument_div_kps += topic_graph_map[topic].number_of_nodes() / len(answer_set)

        print(topic)
        print(f'#argument: {topic_graph_map[topic].number_of_nodes()}')
        print(f'{answer_set=}')
        print(f'{len(answer_set)=}')
        print(f'{sp_avgw=}, {sr_avgw=}, {sf1_avgw=}')

        print('-' * 20)

        dump_kp_pair_score_map(kp_pair_score_map)

    res = {
        'sp_avgw': all_sp_avgw / count,
        'sr_avgw': all_sr_avgw / count,
        'sf1_avgw': all_sf1_avgw / count,
        'rouge1': all_rouge1 / count,
        'rouge2': all_rouge2 / count,
        'rougel': all_rougel / count,
    }
    for key in res:
        res[key] = round(res[key], 4)
    return res


def get_single_score(kp, a, kp_pair_score_map):
    if f'{kp}-{a}' not in kp_pair_score_map:
        kp_pair_score_map[f'{kp}-{a}'] = bleurt(candidates=[kp], references=[a])[0]
    return kp_pair_score_map[f'{kp}-{a}']


def cal_final_score_bleurt(answer_set, kps_set, kp_pair_score_map):
    ak_pairs = [[k, a] for k in kps_set for a in answer_set if f'{k}-{a}' not in kp_pair_score_map]
    print(f'bleurt_ak_pairs: {len(ak_pairs)}')
    if len(ak_pairs) != 0:
        scores = bleurt(candidates=list(map(lambda x: x[0], ak_pairs)), references=list(map(lambda x: x[1], ak_pairs)))
        for (k, a), score in zip(ak_pairs, scores):
            kp_pair_score_map[f'{k}-{a}'] = score
    score_matrix = [[kp_pair_score_map[f'{k}-{a}'] for a in answer_set] for k in kps_set]
    score_matrix = np.array(score_matrix).reshape(len(kps_set), len(answer_set))
    sp = score_matrix.max(axis=-1).mean().tolist()
    sr = score_matrix.max(axis=0).mean().tolist()

    return sp, sr, 2 * sp * sr / (sp + sr), score_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--max_loop_count', type=int, default=200)
    parser.add_argument('--embs_clustering_model', type=str, default='/path/to/BAAI-bge-large-en-v1.5')
    parser.add_argument('--th', type=float, default=0.008)
    parser.add_argument('--dataset', type=str, choices=['QAM', 'ArgKP'], default=None)
    parser.add_argument('--split', type=str, choices=['dev', 'test'], default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t', type=float, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset is None:
        if 'argkp' in args.output_file.lower():
            args.dataset = 'ArgKP'
        elif 'qam' in args.output_file.lower():
            args.dataset = 'QAM'
        else:
            raise NotImplementedError(f'not recognized dataset from {args.output_file}')

    outputs = json.load(open(args.output_file, 'r', encoding='utf-8'))
    results = graph_clustering(outputs, embs_clustering_model=SentenceTransformer(args.embs_clustering_model))

    print(f'{results=}')
