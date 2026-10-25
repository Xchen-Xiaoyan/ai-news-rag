Answerable questions: 76

| experiment   | config                                   |   hit@1 |   hit@5 |   hit@10 |   mrr |   multi all@10 |   latency_s |
|:-------------|:-----------------------------------------|--------:|--------:|---------:|------:|---------------:|------------:|
| method       | bm25 | w200 | bge-small-en-v1.5          |   0.711 |   0.895 |    0.908 | 0.791 |          0.5   |       0.004 |
| method       | dense | w200 | bge-small-en-v1.5         |   0.645 |   0.855 |    0.855 | 0.719 |          0.5   |       0.028 |
| method       | hybrid | w200 | bge-small-en-v1.5        |   0.711 |   0.934 |    0.961 | 0.804 |          0.438 |       0.019 |
| method       | bm25+rerank | w200 | bge-small-en-v1.5   |   0.776 |   0.908 |    0.947 | 0.839 |          0.5   |       0.228 |
| method       | dense+rerank | w200 | bge-small-en-v1.5  |   0.75  |   0.895 |    0.895 | 0.805 |          0.5   |       0.24  |
| method       | hybrid+rerank | w200 | bge-small-en-v1.5 |   0.776 |   0.908 |    0.947 | 0.839 |          0.5   |       0.243 |
| chunk size   | hybrid+rerank | w100 | bge-small-en-v1.5 |   0.803 |   0.921 |    0.947 | 0.858 |          0.438 |       0.179 |
| chunk size   | hybrid+rerank | w350 | bge-small-en-v1.5 |   0.737 |   0.895 |    0.921 | 0.804 |          0.5   |       0.274 |
| embedder     | dense | w200 | bge-base-en-v1.5          |   0.605 |   0.868 |    0.908 | 0.709 |          0.375 |       0.02  |
| embedder     | hybrid+rerank | w200 | bge-base-en-v1.5  |   0.776 |   0.908 |    0.934 | 0.838 |          0.438 |       0.245 |