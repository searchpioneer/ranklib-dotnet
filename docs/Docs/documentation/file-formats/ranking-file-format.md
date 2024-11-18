# Ranking File Format

A ranking file is a ranking output format commonly used by the
[Indri search engine, part of the Lemur Project](https://sourceforge.net/p/lemur/wiki/Indri/)
that provides rankings for queries based on their relevance to a set of documents.

The file follows the format used by [TREC submissions](https://trec.nist.gov/). Each line represents a single
document's rank for a specific query and includes fields separated by whitespace, using the following format:

```text
<line> .=. <qid> Q0 <docid> <rank> <score> <runid>
<qid> .=. <positive integer>
<docid> .=. <positive integer>
<rank> .=. <positive integer>
<score> .=. <float>
<runid> .=. <string>
```

where

- `<qid>`

  An identifier for the query

- `<docid>`

  The unique identifier for the document.

- `<rank>`

  The rank order of the document for the query

- `<score>`

  Relevance score given to the document for each query

- `<runid>`

  Label for the experiment, run, or method used

## Example

The following example has data for two different queries, with the ranks and scores of three documents in each:

```text
101 Q0 DOC001 1 12.34 run1
101 Q0 DOC002 2 11.87 run1
101 Q0 DOC003 3 11.50 run1
102 Q0 DOC045 1 15.00 run2
102 Q0 DOC046 2 14.70 run2
102 Q0 DOC047 3 14.50 run2
```