# Relevance Judgment File Format

A query relevance judgment file is used in Information Retrieval (IR) tasks to provide relevance judgments for
evaluating the effectiveness of search systems. It specifies which documents are relevant or non-relevant
for a set of queries. This is particularly useful for metrics like [`MAP`](../../api/RankLib.Metric.APScorer.yml)
and [`NDCG`](../../api/RankLib.Metric.NDCGScorer.yml) to provide _ideal_ judgments for scoring.

The file follows the format used by [TREC Query Relevance files (**qrels**)](https://trec.nist.gov/). Each line 
represents a judgment for a document and query pair and includes fields separated by whitespace, using the following format:

```text
<line> .=. <qid> <iteration> <docid> <judgment>
<qid> .=. <positive integer>
<iteration> .=. <positive integer>
<docid> .=. <positive integer>
<judgment> .=. <positive integer>
```

where

- `<qid>`

  An identifier for the query

- `<iteration>`

  A placeholder field for the iteration, often 0, which is ignored.

- `<docid>`

  The unique identifier for the document.

- `<judgment>`

  Relevance judgement indicating the document's relevance to the query. Common values use binary relevance:

    - `0`: Not relevant
    - `1`: Relevant

  or a graded relevance judgment from 0 to 4:

    - `0`: Not relevant
    - `1`: ...
    - `2`: ...
    - `3`: ...
    - `4`: Most relevant

## Example

The following example has judgments for two different queries:

```text
101 0 DOC001 1
101 0 DOC002 0
101 0 DOC003 1
102 0 DOC045 2
102 0 DOC046 1
102 0 DOC047 0
```