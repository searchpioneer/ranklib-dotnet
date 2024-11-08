# RankLib for .NET

Ranklib for .NET is a hardened port to .NET of [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/),
a popular open source learning to rank library written in Java. It maintains compatibility with input and output
files of RankLib, allowing it to be used to integrate with systems that use RankLib, such as the
[Elasticsearch Learning to Rank plugin](http://github.com/o19s/elasticsearch-learning-to-rank).

RankLib is available as both a command line tool for training and evaluating rankers, as well as a library for
incorporating into solutions.

## Installation

### Library

To add as a library to an existing project

```sh
dotnet add package RankLib
```

### Command Line Tool

To add as a global .NET command line tool

```sh
dotnet tool install -g RankLib.Cli
```

To see all the commands supported by the command line tool

```sh
dotnet-ranklib --help
```

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

## What is Learning to Rank (LTR)?

Learning to Rank (LTR) is a technique in machine learning that trains models to optimize the
ranking order of items in a list based on relevance to a specific query or user intent.
The goal is to improve the quality of search results, recommendations, and other ranked
lists by understanding and modeling what users find most relevant or useful. LTR is widely
used in search engines, recommendation systems, and information retrieval to enhance user
satisfaction and engagement.

## File formats

The file format for the training, testing, and validation data is the same as for
[SVM-Rank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) and
[LETOR](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) datasets.

Each line in a training file represents one training example, and uses the following format:

```text
<line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
<target> .=. <positive integer>
<qid> .=. <positive integer>
<feature> .=. <positive integer>
<value> .=. <float>
<info> .=. <string>
```

`<info>` typically takes the form of `<document id> <query>`.

The following example has data for three different queries, where each example has two features:

```text
4 qid:1 1:12.318474 2:10.573917 # 7555 rambo
3 qid:1 1:10.357876 2:11.95039  # 1370 rambo
3 qid:1 1:7.0105133 2:11.220095 # 1369 rambo
3 qid:1 1:0.0 2:11.220095 # 1368 rambo
0 qid:1 1:0.0 2:0.0 # 136278 rambo
0 qid:1 1:0.0 2:0.0 # 102947 rambo
0 qid:1 1:0.0 2:0.0 # 13969 rambo
0 qid:1 1:0.0 2:0.0 # 61645 rambo
0 qid:1 1:0.0 2:0.0 # 14423 rambo
0 qid:1 1:0.0 2:0.0 # 54156 rambo
4 qid:2 1:10.686391 2:8.814846 # 1366 rocky
3 qid:2 1:8.985554 2:9.984511 # 1246 rocky
3 qid:2 1:8.985554 2:8.067703 # 60375 rocky
3 qid:2 1:8.985554 2:5.66055 # 1371 rocky
3 qid:2 1:8.985554 2:7.300773 # 1375 rocky
3 qid:2 1:8.985554 2:8.814846 # 1374 rocky
0 qid:2 1:6.815921 2:0.0 # 110123 rocky
0 qid:2 1:6.081685 2:8.725065 # 17711 rocky
0 qid:2 1:6.081685 2:5.9764786 # 36685 rocky
4 qid:3 1:7.672084 2:12.72242 # 17711 bullwinkle
0 qid:3 1:0.0 2:0.0 # 1246 bullwinkle
0 qid:3 1:0.0 2:0.0 # 60375 bullwinkle
0 qid:3 1:0.0 2:0.0 # 1371 bullwinkle
0 qid:3 1:0.0 2:0.0 # 1375 bullwinkle
0 qid:3 1:0.0 2:0.0 # 1374 bullwinkle
```
