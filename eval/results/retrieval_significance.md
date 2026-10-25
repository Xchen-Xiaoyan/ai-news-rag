Paired bootstrap (10,000 resamples, n = 76 answerable questions). Baseline: bm25 | w200 | bge-small-en-v1.5.

| metric   | config                                   |   mean | 95% CI         |   diff vs bm25 | diff 95% CI     | significant   |
|:---------|:-----------------------------------------|-------:|:---------------|---------------:|:----------------|:--------------|
| hit@1    | bm25 | w200 | bge-small-en-v1.5          |  0.711 | [0.605, 0.803] |          0     | [0.000, 0.000]  | no            |
| hit@1    | bm25+rerank | w200 | bge-small-en-v1.5   |  0.776 | [0.684, 0.868] |          0.066 | [-0.039, 0.171] | no            |
| hit@1    | dense | w200 | bge-base-en-v1.5          |  0.605 | [0.500, 0.711] |         -0.105 | [-0.211, 0.013] | no            |
| hit@1    | dense | w200 | bge-small-en-v1.5         |  0.645 | [0.539, 0.750] |         -0.066 | [-0.171, 0.039] | no            |
| hit@1    | dense+rerank | w200 | bge-small-en-v1.5  |  0.75  | [0.645, 0.842] |          0.039 | [-0.079, 0.158] | no            |
| hit@1    | hybrid | w200 | bge-small-en-v1.5        |  0.711 | [0.605, 0.803] |          0     | [-0.092, 0.105] | no            |
| hit@1    | hybrid+rerank | w100 | bge-small-en-v1.5 |  0.803 | [0.711, 0.882] |          0.092 | [-0.013, 0.197] | no            |
| hit@1    | hybrid+rerank | w200 | bge-base-en-v1.5  |  0.776 | [0.671, 0.868] |          0.066 | [-0.039, 0.184] | no            |
| hit@1    | hybrid+rerank | w200 | bge-small-en-v1.5 |  0.776 | [0.671, 0.868] |          0.066 | [-0.039, 0.184] | no            |
| hit@1    | hybrid+rerank | w350 | bge-small-en-v1.5 |  0.737 | [0.632, 0.829] |          0.026 | [-0.092, 0.145] | no            |
| hit@10   | bm25 | w200 | bge-small-en-v1.5          |  0.908 | [0.842, 0.961] |          0     | [0.000, 0.000]  | no            |
| hit@10   | bm25+rerank | w200 | bge-small-en-v1.5   |  0.947 | [0.895, 0.987] |          0.039 | [0.000, 0.092]  | no            |
| hit@10   | dense | w200 | bge-base-en-v1.5          |  0.908 | [0.842, 0.974] |          0     | [-0.079, 0.079] | no            |
| hit@10   | dense | w200 | bge-small-en-v1.5         |  0.855 | [0.763, 0.934] |         -0.053 | [-0.132, 0.013] | no            |
| hit@10   | dense+rerank | w200 | bge-small-en-v1.5  |  0.895 | [0.816, 0.961] |         -0.013 | [-0.079, 0.053] | no            |
| hit@10   | hybrid | w200 | bge-small-en-v1.5        |  0.961 | [0.908, 1.000] |          0.053 | [0.013, 0.105]  | yes           |
| hit@10   | hybrid+rerank | w100 | bge-small-en-v1.5 |  0.947 | [0.895, 0.987] |          0.039 | [0.000, 0.092]  | no            |
| hit@10   | hybrid+rerank | w200 | bge-base-en-v1.5  |  0.934 | [0.868, 0.987] |          0.026 | [-0.026, 0.079] | no            |
| hit@10   | hybrid+rerank | w200 | bge-small-en-v1.5 |  0.947 | [0.895, 0.987] |          0.039 | [0.000, 0.092]  | no            |
| hit@10   | hybrid+rerank | w350 | bge-small-en-v1.5 |  0.921 | [0.855, 0.974] |          0.013 | [-0.039, 0.066] | no            |
| mrr      | bm25 | w200 | bge-small-en-v1.5          |  0.791 | [0.708, 0.864] |          0     | [0.000, 0.000]  | no            |
| mrr      | bm25+rerank | w200 | bge-small-en-v1.5   |  0.839 | [0.764, 0.905] |          0.049 | [-0.015, 0.113] | no            |
| mrr      | dense | w200 | bge-base-en-v1.5          |  0.709 | [0.623, 0.790] |         -0.082 | [-0.168, 0.006] | no            |
| mrr      | dense | w200 | bge-small-en-v1.5         |  0.719 | [0.628, 0.803] |         -0.072 | [-0.154, 0.009] | no            |
| mrr      | dense+rerank | w200 | bge-small-en-v1.5  |  0.805 | [0.721, 0.881] |          0.015 | [-0.068, 0.093] | no            |
| mrr      | hybrid | w200 | bge-small-en-v1.5        |  0.804 | [0.727, 0.872] |          0.013 | [-0.053, 0.080] | no            |
| mrr      | hybrid+rerank | w100 | bge-small-en-v1.5 |  0.858 | [0.786, 0.921] |          0.067 | [-0.001, 0.139] | no            |
| mrr      | hybrid+rerank | w200 | bge-base-en-v1.5  |  0.838 | [0.762, 0.906] |          0.048 | [-0.021, 0.116] | no            |
| mrr      | hybrid+rerank | w200 | bge-small-en-v1.5 |  0.839 | [0.763, 0.906] |          0.049 | [-0.017, 0.116] | no            |
| mrr      | hybrid+rerank | w350 | bge-small-en-v1.5 |  0.804 | [0.722, 0.877] |          0.013 | [-0.063, 0.086] | no            |