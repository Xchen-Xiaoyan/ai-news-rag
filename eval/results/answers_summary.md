Answerer: ollama/qwen2.5:7b | Judge: ollama/qwen2.5:7b

|                                            |   baseline |   full |   full_premise |
|:-------------------------------------------|-----------:|-------:|---------------:|
| n                                          |     75     | 75     |         75     |
| accuracy (correct)                         |      1     |  0.949 |          0.898 |
| accuracy (partial=0.5)                     |      1     |  0.975 |          0.949 |
| faithfulness (claims supported)            |      0.988 |  0.994 |          0.967 |
| hallucination rate (>=1 unsupported claim) |      0.034 |  0.017 |          0.085 |
| false abstention (answerable)              |      0     |  0     |          0     |
| correct abstention (unanswerable)          |      0.875 |  0.812 |          1     |
| single acc                                 |      1     |  0.98  |          0.96  |
| multi acc                                  |      1     |  0.778 |          0.556 |
| errors                                     |      0     |  0     |          0     |