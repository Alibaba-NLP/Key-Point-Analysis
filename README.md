## Exploring Key Point Analysis with Pairwise Generation and Graph Partitioning

### Requirements
- torch 2.0.1
- transformers 4.30.2
- networkx
- sentence_transformers
- rouge_score
- scikit-learn

### Key Point Generation

- ArgKP with Flan-T5-base: `scirpts/run_argkp_base_muc5.sh`
- ArgKP with Flan-T5-large: `scirpts/run_argkp_large_muc5.sh`
- QAM with Flan-T5-large: `scirpts/run_QAM_base_muc5.sh`
- QAM with Flan-T5-large: `scirpts/run_QAM_large_muc5.sh`


### Graph Partitioning
``cd GraphPartitioning && python graph_clustering --dataset ArgKP/QAM --output_file {file path to generated key point output file, under GraphPartitioning/eval_outputs/}``


### Note
We use BLEURT to score the relevance between two sentences. However, unlike our code, which is based on PyTorch, BLEURT requires a TensorFlow environment. To avoid package dependency conflicts, we install an additional TensorFlow environment and then invoke BLEURT through HTTP requests. You can run `bleurt_app.py` in the TensorFlow environment.