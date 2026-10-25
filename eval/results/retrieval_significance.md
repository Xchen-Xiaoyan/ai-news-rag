Paired bootstrap (10,000 resamples, n = 59 answerable questions). Baseline: bm25 | w200 | bge-small-en-v1.5.

| metric   | config                                             |   mean | 95% CI         |   diff vs bm25 | diff 95% CI      | significant   |
|:---------|:---------------------------------------------------|-------:|:---------------|---------------:|:-----------------|:--------------|
| hit@1    | bm25 | w200 | bge-small-en-v1.5                    |  0.729 | [0.610, 0.831] |          0     | [0.000, 0.000]   | no            |
| hit@1    | bm25+rerank | w200 | bge-small-en-v1.5             |  0.797 | [0.695, 0.898] |          0.068 | [-0.051, 0.186]  | no            |
| hit@1    | dense | w200 | bge-base-en-v1.5                    |  0.627 | [0.508, 0.746] |         -0.102 | [-0.220, 0.017]  | no            |
| hit@1    | dense | w200 | bge-small-en-v1.5                   |  0.661 | [0.542, 0.780] |         -0.068 | [-0.186, 0.051]  | no            |
| hit@1    | dense+rerank | w200 | bge-small-en-v1.5            |  0.763 | [0.661, 0.864] |          0.034 | [-0.102, 0.169]  | no            |
| hit@1    | hybrid | w200 | bge-small-en-v1.5                  |  0.695 | [0.576, 0.814] |         -0.034 | [-0.136, 0.068]  | no            |
| hit@1    | hybrid+rerank | w100 | bge-small-en-v1.5           |  0.831 | [0.729, 0.915] |          0.102 | [0.000, 0.220]   | no            |
| hit@1    | hybrid+rerank | w200 | bge-base-en-v1.5            |  0.797 | [0.695, 0.898] |          0.068 | [-0.051, 0.186]  | no            |
| hit@1    | hybrid+rerank | w200 | bge-small-en-v1.5           |  0.797 | [0.695, 0.898] |          0.068 | [-0.051, 0.186]  | no            |
| hit@1    | hybrid+rerank | w350 | bge-small-en-v1.5           |  0.797 | [0.695, 0.898] |          0.068 | [-0.068, 0.203]  | no            |
| hit@1    | hybrid+rerank+decompose | w200 | bge-small-en-v1.5 |  0.797 | [0.695, 0.898] |          0.068 | [-0.051, 0.186]  | no            |
| hit@10   | bm25 | w200 | bge-small-en-v1.5                    |  0.966 | [0.915, 1.000] |          0     | [0.000, 0.000]   | no            |
| hit@10   | bm25+rerank | w200 | bge-small-en-v1.5             |  0.983 | [0.949, 1.000] |          0.017 | [0.000, 0.051]   | no            |
| hit@10   | dense | w200 | bge-base-en-v1.5                    |  0.949 | [0.881, 1.000] |         -0.017 | [-0.085, 0.051]  | no            |
| hit@10   | dense | w200 | bge-small-en-v1.5                   |  0.881 | [0.797, 0.949] |         -0.085 | [-0.153, -0.017] | yes           |
| hit@10   | dense+rerank | w200 | bge-small-en-v1.5            |  0.932 | [0.864, 0.983] |         -0.034 | [-0.102, 0.034]  | no            |
| hit@10   | hybrid | w200 | bge-small-en-v1.5                  |  0.983 | [0.949, 1.000] |          0.017 | [0.000, 0.051]   | no            |
| hit@10   | hybrid+rerank | w100 | bge-small-en-v1.5           |  0.983 | [0.949, 1.000] |          0.017 | [0.000, 0.051]   | no            |
| hit@10   | hybrid+rerank | w200 | bge-base-en-v1.5            |  0.966 | [0.915, 1.000] |          0     | [-0.051, 0.051]  | no            |
| hit@10   | hybrid+rerank | w200 | bge-small-en-v1.5           |  0.983 | [0.949, 1.000] |          0.017 | [0.000, 0.051]   | no            |
| hit@10   | hybrid+rerank | w350 | bge-small-en-v1.5           |  0.949 | [0.881, 1.000] |         -0.017 | [-0.085, 0.034]  | no            |
| hit@10   | hybrid+rerank+decompose | w200 | bge-small-en-v1.5 |  0.983 | [0.949, 1.000] |          0.017 | [0.000, 0.051]   | no            |
| mrr      | bm25 | w200 | bge-small-en-v1.5                    |  0.824 | [0.743, 0.898] |          0     | [0.000, 0.000]   | no            |
| mrr      | bm25+rerank | w200 | bge-small-en-v1.5             |  0.866 | [0.792, 0.932] |          0.042 | [-0.025, 0.111]  | no            |
| mrr      | dense | w200 | bge-base-en-v1.5                    |  0.739 | [0.648, 0.826] |         -0.085 | [-0.169, -0.003] | yes           |
| mrr      | dense | w200 | bge-small-en-v1.5                   |  0.743 | [0.644, 0.835] |         -0.081 | [-0.171, 0.006]  | no            |
| mrr      | dense+rerank | w200 | bge-small-en-v1.5            |  0.83  | [0.746, 0.907] |          0.006 | [-0.082, 0.093]  | no            |
| mrr      | hybrid | w200 | bge-small-en-v1.5                  |  0.807 | [0.728, 0.882] |         -0.016 | [-0.084, 0.050]  | no            |
| mrr      | hybrid+rerank | w100 | bge-small-en-v1.5           |  0.89  | [0.823, 0.950] |          0.067 | [-0.001, 0.139]  | no            |
| mrr      | hybrid+rerank | w200 | bge-base-en-v1.5            |  0.864 | [0.789, 0.932] |          0.04  | [-0.035, 0.115]  | no            |
| mrr      | hybrid+rerank | w200 | bge-small-en-v1.5           |  0.866 | [0.793, 0.932] |          0.043 | [-0.030, 0.116]  | no            |
| mrr      | hybrid+rerank | w350 | bge-small-en-v1.5           |  0.852 | [0.768, 0.926] |          0.028 | [-0.059, 0.111]  | no            |
| mrr      | hybrid+rerank+decompose | w200 | bge-small-en-v1.5 |  0.864 | [0.790, 0.932] |          0.04  | [-0.034, 0.115]  | no            |