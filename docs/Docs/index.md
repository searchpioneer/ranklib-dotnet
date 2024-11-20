# RankLib for .NET

[![NuGet Release][nuget image]][nuget url]
[![Build Status](https://github.com/searchpioneer/ranklib-dotnet/actions/workflows/dotnet.yml/badge.svg)](https://github.com/searchpioneer/ranklib-dotnet/actions/workflows/dotnet.yml)
[![license badge][license badge]][license url]

Ranklib for .NET is a hardened open source port to .NET of [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/),
a popular open source learning to rank library written in Java. It maintains compatibility with input and output
files of RankLib, allowing it to be used to integrate with systems that use RankLib, such as the
[Elasticsearch Learning to Rank](http://github.com/o19s/elasticsearch-learning-to-rank) and
[OpenSearch Learning to Rank](https://opensearch.org/docs/latest/search-plugins/ltr/index/) plugins.

RankLib for .NET is available as both a command line tool for training and evaluating rankers, as well as a library for
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
3. [LambdaRank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)
4. [RankBoost](https://www.jmlr.org/papers/volume4/freund03a/freund03a.pdf)
5. [AdaRank](https://dl.acm.org/doi/10.1145/1277741.1277809)
6. [Coordinate Ascent](https://link.springer.com/content/pdf/10.1007/s10791-006-9019-z.pdf)
7. [LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf)
8. [ListNet](https://dl.acm.org/doi/10.1145/1273496.1273513)
9. [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

The following metrics are implemented to measure the effectiveness of ranking

1. Average Precision (`MAP`)
2. Best at K (`Best@K`)
3. Discounted Cumulative Gain (`DCG@K`)
4. Expected Reciprocal Rank (`ERR@K`)
5. Normalized Discounted Cumulative Gain (`NDCG@K`)
6. Precision at K (`P@K`)
7. Reciprocal Rank (`RR`)

[nuget url]: https://www.nuget.org/packages/SearchPioneer.RankLib/
[nuget image]: https://img.shields.io/nuget/v/SearchPioneer.RankLib.svg
[license badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license url]: https://www.apache.org/licenses/LICENSE-2.0
