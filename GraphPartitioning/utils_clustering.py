import os

import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def find_core_sample(embeddings: np.array, core_emb=None):
    if core_emb is None:
        core_emb = np.mean(embeddings, axis=0)
    distances = list(map(lambda x: np.linalg.norm(core_emb - x), embeddings))
    return np.argmin(distances), distances


def embs_clustering(model, sentences, cluster_algorithm, seed, **kwargs):
    embs = model.encode(sentences, batch_size=32)
    if cluster_algorithm == 'dbscan':
        dbscan_eps = float(os.environ.get('dbscan_eps'))
        dbscan_min_samples = int(os.environ.get('dbscan_min_samples'))
        print(f'{dbscan_eps=}')
        print(f'{dbscan_min_samples=}')
        results = clustering_impl_dbscan(embs=embs, sentences=sentences,
                                         dbscan_eps=dbscan_eps, seed=seed,
                                         dbscan_min_samples=dbscan_min_samples)
    elif cluster_algorithm == 'kmeans':
        results = clustering_impl_kmeans(embs=embs, sentences=sentences, seed=seed,
                                         kmeans_n_clusters=kwargs.get('kmeans_n_clusters'))
    else:
        raise NotImplementedError(f'not support cluster algorithm: {cluster_algorithm}')
    return results


def clustering_impl_kmeans(embs, sentences, seed, **kwargs):
    clustering = KMeans(n_clusters=min(kwargs.get('kmeans_n_clusters'), len(sentences)), n_init=500, random_state=seed).fit(embs)
    labels = clustering.labels_
    cluster_centers = clustering.cluster_centers_
    clusters = []
    for idx in range(max(labels) + 1):
        idx_clus = np.where(labels == idx)[0]
        clus_emb = embs[idx_clus]
        core_sample_idx, distances = find_core_sample(clus_emb, core_emb=cluster_centers[idx])
        cluster_sentences = np.array(sentences)[idx_clus].tolist()
        assert len(cluster_sentences) == len(distances)
        sorted_cluster_sentences = list(map(lambda x: x[0], sorted(list(zip(cluster_sentences, distances)), key=lambda x: x[1])))
        clusters.append({'core kp': sentences[idx_clus[core_sample_idx]], 'cluster': sorted_cluster_sentences})
    return [c['core kp'] for c in clusters], [c['cluster'] for c in clusters], None, {'total_count': len(sentences), f'noise sentences count': 0,
                                                                                      f'cluster_num': max(labels).tolist() + 1}


def clustering_impl_dbscan(embs, sentences, **kwargs):
    clustering = DBSCAN(eps=kwargs.get('dbscan_eps'), min_samples=kwargs.get('dbscan_min_samples')).fit(embs)
    labels = clustering.labels_
    clusters = []
    for idx in range(max(labels) + 1):
        idx_clus = np.where(labels == idx)[0]
        clus_emb = embs[idx_clus]
        core_sample_idx, _ = find_core_sample(clus_emb)
        clusters.append({'core kp': sentences[idx_clus[core_sample_idx]], 'cluster': np.array(sentences)[idx_clus].tolist()})
    noise_cluster = np.array(sentences)[np.where(labels == -1)].tolist()
    m1_count = len(list(filter(lambda x: x == -1, labels)))
    return [c['core kp'] for c in clusters], [c['cluster'] for c in clusters], noise_cluster, {'total_count': len(sentences), f'noise sentences count': m1_count,
                                                                                               f'cluster_num': max(labels).tolist() + 1}
