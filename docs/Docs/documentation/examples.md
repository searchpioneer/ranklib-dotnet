# Examples

> The following examples are an amended version of the 
> [RankLib examples documentation](https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/).

Download one of the [Microsoft Learning to Rank Datasets](https://www.microsoft.com/en-us/research/project/mslr/). 

For these examples, we'll use the MSLR-WEB10K dataset. Unzip it to a directory, and navigate to the directory on
the command line.

## Training on held-out data

Run the following on the command line

```sh
dotnet-ranklib eval -train Fold1/train.txt -test Fold1/test.txt -validate Fold1/vali.txt -ranker LambdaMART --train-metric NDCG@10 --test-metric ERR@10 -save mymodel.txt
```

This command trains a [LambdaMART](../api/RankLib.Learning.Tree.LambdaMART.yml) ranker on the training data and
records the model that performs best on the validation data. The training metric is NDCG@10. After training is
completed, the trained model is evaluated on the test data using ERR@10. Finally, the model is saved to a file
named mymodel.txt in the current directory.

The parameter `-validate` is _optional_, but it often leads to better models. In particular, `-validate` is very important
for [RankNet](../api/RankLib.Learning.NeuralNet.RankNet.yml), [MART](../api/RankLib.Learning.Tree.MART.yml), 
and [LambdaMART](../api/RankLib.Learning.Tree.LambdaMART.yml). [Coordinate Ascent](../api/RankLib.Learning.CoordinateAscent.yml),
on the other hand, works pretty well without validation data.

> [!IMPORTANT]
> `--train-metric` only applies to _list-wise_ algorithms 
> ([AdaRank](../api/RankLib.Learning.Boosting.AdaRank.yml),
> [Coordinate Ascent](../api/RankLib.Learning.CoordinateAscent.yml) and
> [LambdaMART](../api/RankLib.Learning.Tree.LambdaMART.yml)). _Point-wise_ and _pair-wise_ techniques 
> ([MART](../api/RankLib.Learning.Tree.MART.yml), [RankNet](../api/RankLib.Learning.NeuralNet.RankNet.yml),
> [RankBoost](../api/RankLib.Learning.Boosting.RankBoost.yml)),
> due to their nature, always use their internal RMSE / pair-wise loss as the optimization criteria.
> Thus, `--train-metric` has no effect on them. [ListNet](../api/RankLib.Learning.NeuralNet.ListNet.yml) is a special case.
> Despite being a list-wise algorithm,
> it has its own optimization criteria as well. Therefore, `--train-metric` also has no effect on ListNet.

> [!TIP]
> Instead of using a separate validation dataset, you can use the `-tvs` option to perform a split of the training
> data into train and validation datasets.

> [!TIP]
> Instead of using a separate test dataset, you can use the `-tts` option to perform a split of the training data
> into train and test datasets.

## k-Fold Cross Validation

Although the MSLR-WEB10K dataset comes with train, test, and validation data, let's pretend that we
had MSLR-WEB10K/Fold1/train.txt as the only dataset. We now want to do 5-fold cross validation experiment.

### Sequential partition

```sh
dotnet-ranklib eval -train Fold1/train.txt -ranker CoordinateAscent -kcv 5 -kcvmd models/ -kcvmn ca --train-metric NDCG@10 --test-metric ERR@10
```

The command above will sequentially split the training data into 5 chunks of roughly equal size. 
The `i`-th chunk is used as the test data for the `i`-th fold. The training data for each fold consists of the test data
from all other folds.

This example trains a Coordinate Ascent model on each fold that optimizes for NDCG@10.
This model is then evaluated on the corresponding test data for the current fold using ERR@10.
After the training process completes, the overall performance across all folds is reported and all
5 models (one for each fold) are saved to the specified models directory, using the naming convention:
f1.ca, f2.ca, f3.ca, f4.ca, and f5.ca.

> [!TIP]
> You can use the `-tvs` option to reserve a portion of training data in each fold for validation

### Randomized partition

Let's say in the training data, rank lists are ordered in a certain way such that sequentially
partitioning them might introduce some bias. We want `k` partitions such that each partition contains a random
portion of the input data. We can do this simply by shuffling the order of rank lists in the training data:

```sh
dotnet-ranklib prepare -input Fold1/train.txt -output mydata -shuffle
```

The command above will create the shuffled copy of the training data called "train.txt.shuffled" which is saved to 
the `mydata` directory. Cross validation can then be done using the shuffled data

```sh
dotnet-ranklib eval -train Fold1/mydata/train.txt.shuffled -ranker CoordinateAscent -kcv 5 -kcvmd models -kcvmn ca --train-metric NDCG@10 --test-metric ERR@10
```

### How do I obtain the data used in each fold?

You can obtain the data used in each fold with `ranklib prepare`

```sh
dotnet-ranklib prepare -input MQ2008/Fold1/train.txt.shuffled -output mydata/ -k 5
```

This will extract and store the train/test data used in each fold, which is exactly the same as the in-memory
partitions used for learning (partitioning is always done sequentially).

> [!TIP]
> See [dotnet-ranklib prepare](cli/dotnet-ranklib-prepare.md) for more details, or run
> 
> ```sh
> dotnet-ranklib prepare --help
> ```

## Evaluating previously trained models

```sh
dotnet-ranklib eval -load mymodel.txt -test Fold1/test.txt --test-metric ERR@10
```

This will evaluate the pre-trained model stored in mymodel.txt on the specified test data using ERR@10.

## Comparing models

Let's assume our test data test.txt contains a set of queries, and lists of documents (or more precisely,
their feature vector) retrieved for each of the queries using BM25. Let's say we have trained two models:
ca.model.txt (a Coordinate Ascent model) and lm.model.txt (a LambdaMART modeL) from the same training set.
The task is to see if using the Coordinate Ascent model and the LambdaMART model to re-rank these BM25 ranked
lists will improve retrieval effectiveness (NDCG@10).

It goes like this:

```sh
dotnet-ranklib eval -test Fold1/test.txt --test-metric NDCG@10 -idv output/baseline.ndcg.txt
dotnet-ranklib eval -load ca.model.txt -test Fold1/test.txt --test-metric NDCG@10 -idv output/ca.ndcg.txt
dotnet-ranklib eval -load lm.model.txt -test Fold1/test.txt --test-metric NDCG@10 -idv output/lm.ndcg.txt
```

Each of the output files (specified with -idv) provides the ndcg@10 each system achieves for each of the test queries.
These files are stored in the output/ directory. Note that these commands are different from the one used in
Section 2.3 above, which only reports the average measure (e.g. ndcg@10) across all queries. These 3 commands,
on the other hand, report ndcg@10 on each of the queries, not just the average.

Here's an example of an output file showing individual and all query performance levels (in terms of the selected metric):

```
NDCG@10   170   0.0
NDCG@10   176   0.6722390270733757
NDCG@10   177   0.4772656487866462
NDCG@10   178   0.539003131276382
NDCG@10   185   0.6131471927654585
NDCG@10   189   1.0
NDCG@10   191   0.6309297535714574
NDCG@10   192   1.0
NDCG@10   194   0.2532778777010656
NDCG@10   197   1.0
NDCG@10   200   0.6131471927654585
NDCG@10   204   0.4772656487866462
NDCG@10   207   0.0
NDCG@10   209   0.123151194370365
NDCG@10   221   0.39038004999210174
NDCG@10   all   0.5193204478059303
```    
    
Now to compare them, do this:

```sh
dotnet-ranklib analyze -all output -base baseline.ndcg.txt > analysis.txt
```

The output file analysis.txt is tab separated. Copy and paste it into any spreadsheet program for easy viewing.
Everything should be self-explanatory. It looks like this:

```text
Overall comparison
------------------------------------------------------------------------
System  Performance     Improvement     Win     Loss    p-value
baseline_ndcg.txt [baseline]    0.093
LM_ndcg.txt     0.2863  +0.1933 (+207.8%)       9       1       0.03
CA_ndcg.txt     0.5193  +0.4263 (+458.26%)      12      0       0.0

Detailed break down
------------------------------------------------------------------------
            [ < -100%)  [-100%,-75%)  [-75%,-50%)  [-50%,-25%)  [-25%,0%)  (0%,+25%]  (+25%,+50%]  (+50%,+75%]  (+75%,+100%]  ( > +100%]
LM_ndcg.txt    0           0            1             0            0         4            2            2            1            0
CA_ndcg.txt    0           0            0             0            0         1            6            2            3            0
```

This output shows performance comparisons of two saved models (CoordinateAscent and LambdaMART) against a baseline.
The table shows performance differences and percent improvements between each saved model and the baseline.
Also numbers of queries that were better or worse than baseline and P-value for statistical confidence in the better
model. Note, only queries where performance metrics showed different values are listed in the win/loss columns.

The final part of the output is a simple histogram of performance differences over percent change intervals.
Again, only queries that provided differences in metric values are listed and the percent values actually represent
difference values between base and test in chosen metric X 100.

> [!TIP]
> See [dotnet-ranklib analyze](cli/dotnet-ranklib-analyze.md) for more details, or run
>
> ```sh
> dotnet-ranklib analyze --help
> ```

## Using trained models to do re-ranking

Instead of using a model to re-rank documents in some test data and examining the effectiveness of the final rankings
(as shown in [Evaluating Previously Trained models](#evaluating-previously-trained-models)), we may want to view 
the re-rankings themselves (i.e. these rankings might serve as input to
some other systems) as produced by a saved model. This is achieved by loading a saved model to use for re-scoring an
input result list (or result lists from a set of queries) and producing an output file that contains the new scores
for documents in the input list. The output file will have to be sorted by score within query IDs to get the actual
ranked list for each query. The input data has the same format as the training/validation/test data used to produce
a model.

This can be achieved using the following RankLib command:

```sh
dotnet-ranklib eval -load mymodel.txt -rank myResultLists.txt -score myScoreFile.txt
```

The output file myScoreFile.txt provides the score that the ranker assigns to each of the documents as presented in
the input list. This will not be a ranked list per se. It is merely a re-scoring of the documents in the presented
input data. One would need to sort the scores within query ID to get the new ranking for each query. The re-scored
output in myScoreFile.txt would appear as follows:

```text
1   0   -7.528650760650635
1   1   2.9022061824798584
1   2   -0.700125515460968
1   3   2.376657485961914
1   4   -0.29666265845298767
1   5   -2.038628101348877
1   6   -5.267711162567139
1   7   -2.022146463394165
1   8   0.6741248369216919
...
```

Note that the output does not specify document IDs. RankLib has no knowledge of document IDs; only the order of
documents in the input data result list for each query, and uses that ordering as a sort of ID.

If one wished to know the actual indexed document IDs of the re-scored documents, one would have to present that
information in the optional description field for the data input list. The description field is all the text on a
feature input file at the end of each line starting with a pound character. The description can include document
ID and any other information of interest to the user for the particular document. An example input data file shown
below includes document ID and a couple further information items (inc and prob) in the description. Note the input
file must still contain a relevance label (the first field on a line), but it's not used since one is re-ranking a
list from an already defined model, not developing the model itself or doing an evaluation.

```text
0 qid:1 1:0.000000 2:0.000000 3:0.000000 4:0.000000 5:0.000000 #docid=GX000-00-0000000 inc=1 prob=0.0246906
1 qid:1 1:0.031310 2:0.666667 3:0.500000 4:0.166667 5:0.033206 #docid=GX000-24-12369390 inc=0.60031 prob=0.416367
1 qid:1 1:0.078682 2:0.166667 3:0.500000 4:0.333333 5:0.080022 #docid=GX000-62-7863450 inc=1 prob=0.56895
1 qid:1 1:0.019058 2:1.000000 3:1.000000 4:0.500000 5:0.022591 #docid=GX016-48-5543459 inc=1 prob=0.775913
...
```

To obtain a direct re-ranking of the input data using a saved model, one must print the output using the `-indri` option,
which will include the description field of the input data. If this data includes a document ID, it will be in the
display. If not, one must do one's own processing to figure out the original document IDs for the input data.

The following example re-ranks the input data using the `-indri` option. One can see the description field is printed
between the Q0 and document rank fields. This format is identical to the format used for TREC evaluations if the
description contains only a document ID, although the indri label may not be what is desired if doing an actual
TREC submission.

```sh
dotnet-ranklib eval -rank myResultList.txt -load myModel.txt -indri myNewRankedLists.txt
```

```
1 Q0 docid=GX236-70-4445188 inc=0.600318836372593 prob=0.170566 1 3.21606 indri
1 Q0 docid=GX000-24-12369390 inc=0.60031 prob=0.416367 2 2.90221 indri
1 Q0 docid=GX016-48-5543459 inc=1 prob=0.775913 3 2.37666 indri
1 Q0 docid=GX272-12-14599061 inc=0.600318836372593 prob=0.236838 4 0.94184 indri
1 Q0 docid=GX225-79-9870332 inc=0.568969759822883 prob=0.517692 5 0.7082 indri
1 Q0 docid=GX068-48-12934837 inc=1 prob=0.659932 6 0.67412 indri
1 Q0 docid=GX265-53-7328411 inc=1 prob=0.416294 7 0.08022 indri
1 Q0 docid=GX261-90-2024545 inc=0.600318836372593 prob=0.451932 8 -0.20906 indri
```

## Generating Model Feature Statistics

The Feature Manager can generate feature use statistics from saved model files that make use of only a subset of 
the defined features used to generate the model. The purpose of these statistics are to aid in determination of what
features of a model might be discarded allowing substitution of new features. If a model does not even make use of a
number of features, then those features are not contributing towards development of an effective model, and other,
new, features might be substituted to improve the model. As always, experimentation is needed to confirm whether
specific features aid or hinder effective model creation.

The `dotnet-ranklib stats` command along with the path to a saved LTR model, generates
feature use frequencies as well as minimum, maximum, mean and mode freature frequency values along with frequency
variation and standard deviation. Feature statistics are only generated for models that do not use all the defined
features in creating a model, so these statistics are not available for Coordinate Ascent, LambdaRank,
Linear Regression, ListNet and RankNet models.

Print out feature use statistics for a saved RankLib MART model. This model is tree based and does not use all
features that were defined for it.

```sh
dotnet-ranklib stats models/mart_model.txt
```

```
Model File: models/mart_model.txt
Algorithm : MART

Feature frequencies :
Feature[1] :     226
Feature[2] :     156
Feature[3] :      42
Feature[4] :      28
Feature[5] :     213
Feature[11] :     208
Feature[12] :     281
Feature[13] :     223
Feature[14] :     121
Feature[15] :     259
Feature[16] :     188
Feature[17] :     255
Feature[18] :     246
Feature[19] :     315
Feature[20] :     136
Feature[21] :     294
Feature[22] :     332
Feature[23] :     303
Feature[24] :     303
Feature[25] :     135
Feature[26] :     203
Feature[27] :     263
Feature[28] :     261
Feature[29] :     159
Feature[30] :     187
Feature[31] :     228
Feature[32] :     188
Feature[33] :     205
Feature[34] :     153
Feature[35] :     144
Feature[36] :     131
Feature[37] :     296
Feature[38] :     365
Feature[39] :     302
Feature[40] :     341
Feature[41] :     147
Feature[42] :     336
Feature[43] :     205
Feature[44] :     235
Feature[45] :     380
Feature[46] :       7

Total Features Used: 41

Min frequency    :       7.00
Max frequency    :     380.00
Median frequency :     223.00
Avg frequency    :     219.51
Variance         :    7811.06
STD              :      88.38
```

The training data used to create this MART model defined a total of 46 features. Note however that only 41 of the
features were actually used. In the feature frequency distributions, there is no output for features 6-10 indicating
0 frequency. The minimum feature frequency was for feature 46, which was used only 7 times in the model.
The maximum feature occurrence was for feature 45 which occurred 380 times in the model. Given the mean and
standard deviation values for the frequency distributions, one might consider removing features 3, 4 and 46,
as well as those features not present at all in the model to see if model performance changes significantly.
The removal of these features could make room for the addition of new (and hopefully better) features to produce
the model.
