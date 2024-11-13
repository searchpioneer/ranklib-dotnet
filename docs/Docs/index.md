# RankLib for .NET

Ranklib for .NET is a hardened open source port to .NET of [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/),
a popular open source learning to rank library written in Java. It maintains compatibility with input and output
files of RankLib, allowing it to be used to integrate with systems that use RankLib, such as the
[Elasticsearch Learning to Rank plugin](http://github.com/o19s/elasticsearch-learning-to-rank).

RankLib is available as both a command line tool for training and evaluating rankers, as well as a library for
incorporating into solutions.

## What is Learning to Rank (LTR)?

Learning to Rank (LTR) is a technique in machine learning that trains models to optimize the
ranking order of items in a list based on relevance to a specific query or user intent.
The goal is to improve the quality of search results, recommendations, and other ranked
lists by understanding and modeling what users find most relevant or useful. LTR is widely
used in search engines, recommendation systems, and information retrieval to enhance user
satisfaction and engagement.

## Algorithms and Metrics

The following ranking algorithms are implemented

1. [MART (Multiple Additive Regression Trees, a.k.a. Gradient Boosted Decision Trees (GBDT))](https://jerryfriedman.su.domains/ftp/trebst.pdf)
2. [RankNet](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf)
3. [RankBoost](https://www.jmlr.org/papers/volume4/freund03a/freund03a.pdf)
4. [AdaRank](https://dl.acm.org/doi/10.1145/1277741.1277809)
5. [Coordinate Ascent](https://link.springer.com/content/pdf/10.1007/s10791-006-9019-z.pdf)
6. [LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf)
7. [ListNet](https://dl.acm.org/doi/10.1145/1273496.1273513)
8. [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

The following metrics are implemented to measure the effectiveness of ranking

1. Average Precision (`MAP`)
2. Best at K (`Best@K`)
3. Discounted Cumulative Gain (`DCG@K`)
4. Expected Reciprocal Rank (`ERR@K`)
5. Normalized Discounted Cumulative Gain (`NDCG@K`)
6. Precision at K (`P@K`)
7. Reciprocal Rank (`RR`)
