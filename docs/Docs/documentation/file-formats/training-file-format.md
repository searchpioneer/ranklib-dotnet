# Training File Format

The file format for training, testing, and validation data is the same as for
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

where

- `<target>`

  the relevance label. This is typically a value in the range `[0, 1, 2, 3, 4]` where `0` is not relevant,
  and `4` is perfect relevance, or in the range `[0, 1]` where `0` is not relevant and `1` is relevant.

- `<qid>`

  An identifier for the query. A common approach is to assign a numeric identifier to each distinct query text,
  or group related query text under an identifier.

- `<feature>:<value>`

  An identifier for a feature and its value. Features and their values are used to learn a ranking function.

  > [!IMPORTANT]
  >
  > Features **must** be ordered by ascending feature identifier on each line.

- `<info>` 

  Comments for the example that are not used in learning, but help in identifying examples and aid in readability.
  Typically takes the form of `<document id> <query text>`.

## Example

The following example has data for three different queries, where each example has two features.

```text
4 qid:1 1:12.318474 2:10.573917 # 7555 rambo
3 qid:1 1:10.357876 2:11.95039  # 1370 rambo
3 qid:1 1:7.0105133 2:11.220095 # 1369 rambo
3 qid:1 1:0.0       2:11.220095 # 1368 rambo
0 qid:1 1:0.0       2:0.0       # 136278 rambo
0 qid:1 1:0.0       2:0.0       # 102947 rambo
0 qid:1 1:0.0       2:0.0       # 13969 rambo
0 qid:1 1:0.0       2:0.0       # 61645 rambo
0 qid:1 1:0.0       2:0.0       # 14423 rambo
0 qid:1 1:0.0       2:0.0       # 54156 rambo
4 qid:2 1:10.686391 2:8.814846  # 1366 rocky
3 qid:2 1:8.985554  2:9.984511  # 1246 rocky
3 qid:2 1:8.985554  2:8.067703  # 60375 rocky
3 qid:2 1:8.985554  2:5.66055   # 1371 rocky
3 qid:2 1:8.985554  2:7.300773  # 1375 rocky
3 qid:2 1:8.985554  2:8.814846  # 1374 rocky
0 qid:2 1:6.815921  2:0.0       # 110123 rocky
0 qid:2 1:6.081685  2:8.725065  # 17711 rocky
0 qid:2 1:6.081685  2:5.9764786 # 36685 rocky
4 qid:3 1:7.672084  2:12.72242  # 17711 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1246 bullwinkle
0 qid:3 1:0.0       2:0.0       # 60375 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1371 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1375 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1374 bullwinkle
```

The example includes additional whitespace to make it easier to read, but a single whitespace character between
each piece of data on each line is sufficient.
