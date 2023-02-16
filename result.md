| \              | DeepLog | LogAnomaly | UniLog | 
| HDFS(session)  |  0.935  |   0.910    | 0.931  |
|  BGL(session)  |  0.991  |   0.952    |        |
| BGL(timestamp) |  0.322  |   0.273    |        |

| DeepLog | BGL (session) |
| topk=20 |     0.964     |
| topk=30 |     0.967     |
| topk=40 |     0.982     |
| topk=45 |     0.975     |
| topk=50 |     0.991     |

|    DeepLog    | BGL (timestamp) |
| 100l, topk=30 |      0.321      |
| 100l, topk=40 |      0.322      |

| LogAnomaly | BGL (session) |
|  topk=20   |    0.943      |
|  topk=25   |    0.952      |
|  topk=30   |    0.948      |
|  topk=35   |    0.941      |
|  topk=40   |    0.932      |
|  topk=50   |    0.711      |

| LogAnomaly | BGL (timestamp) |
|  topk=30   |     0.273       |