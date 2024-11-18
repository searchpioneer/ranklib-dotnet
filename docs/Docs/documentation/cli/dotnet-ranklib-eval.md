# dotnet-ranklib eval

Trains and evaluates a ranker, or evaluates a previously saved ranker model.

## Usage

```sh
dotnet-ranklib eval [options]
```

## Options

- **-train | --train-input-file `<train-input-file>`**  
  Training data file in [SVMRank / LETOR format](../file-formats/training-file-format.md)

- **-ranker | --ranker `<ranker>`**  
  Ranking algorithm to use. The default is `CoordinateAscent`

  - `AdaRank`
  - `CoordinateAscent`
  - `LambdaMART`
  - `LambdaRank`
  - `LinearRegression`
  - `ListNet`
  - `MART`
  - `RandomForests`
  - `RankBoost`
  - `RankNet`

- **-feature | --feature-description-input-file `<feature-description-input-file>`**  
  Feature description file. List features to be considered by the learner, each on a separate line. 
  If not specified, all features will be used.

- **-metric2t | --train-metric `<train-metric>`**  
  Metric to optimize on the training data. The default is `ERR@10`

  - `MAP` 
  - `NDCG@k`
  - `DCG@k`
  - `P@k`
  - `RR@k` 
  - `ERR@k`

  > [!IMPORTANT]
  > Use `--train-metric` as the current implementation does not case-sensitive

- **-metric2T | --test-metric `<test-metric>`**  
  Metric to evaluate on the test data. The default is same as **-metric2t | --train-metric**

  > [!IMPORTANT]
  > Use `--test-metric` as the current implementation does not case-sensitive

- **-gmax | --max-label `<max-label>`**  
  Highest judged relevance label. It affects the calculation of ERR. The default is `4` i.e. 
  5-point scale `[0,1,2,3,4]` where value used is $2^{gmax}$

- **-qrel | --query-relevance-input-file `<query-relevance-input-file>`**  
  [TREC-style relevance judgment file](../file-formats/relevance-judgment-file-format.md)

- **-missingZero | --missing-zero**  
  Substitute zero for missing feature values rather than throwing an exception.

- **-validate | --validate-file `<validate-file>`**  
  Specify if you want to tune your system on the validation data

- **-tvs | --train-validation-split `<train-validation-split>`**  
  If you don't have separate validation data, use this to set train-validation split to be. **Must** be between 0 and 1,
  where `train=<train-validation-split>` and `validate=(1.0 - <train-validation-split>)`

- **-test | --test-input-files `<test-input-files>`**  
  Specify if you want to evaluate the trained model on this data

- **-tts | --train-test-split `<train-test-split>`**  
  Set train-test split. **Must** be between 0 and 1, where `train=<train-test-split>` and `test=(1.0 - <train-test-split>)`.
  Overrides **-tvs | --train-validation-split**

- **-save | --model-output-file `<model-output-file>`**  
  Save the model learned. The default does not save the model.

- **-norm | --norm `<normalizer>`**  
  Type of normalizer to use to normalize all feature vectors. the default is no normalization. 
 
  - `Linear`
  - `Sum`
  - `ZScore`

- **-kcv | --cross-validation-folds `<cross-validation-folds>`**  
  Specify how many folds to perform for k-fold cross validation using the specified training data. 
  The default is k-fold cross validation.

- **-kcvmd | --cross-validation-output-directory `<cross-validation-output-directory>`**  
  Directory for models trained via cross-validation

- **-kcvmn | --cross-validation-model-name `<cross-validation-model-name>`**  
  Name for model learned in each fold. It will be prefixed with the fold-number

- **-load | --model-input-files `<model-input-files>`**  
  Load saved model file for evaluation

- **-thread | --thread `<thread>`**  
  Number of threads to use. The performance of some algorithms can be improved by parallelizing computation.
  The default is to use all available processors

- **-rank | --rank-input-file `<rank-input-file>`**  
  Rank the samples in the specified file. Specify either this or **-test | --test-input-files**, but not both

- **-indri | --rank-output-file `<indri-ranking-output-file>`**  
  [Indri ranking file](../file-formats/ranking-file-format.md) with ranking outputs.

- **-sparse | --use-sparse-representation**  
  Use data points with sparse representation. Default is `false` which uses dense data points.

- **-idv | --individual-ranklist-performance-output-file `<individual-ranklist-performance-output-file>`**  
  Individual rank list model performance (in test metric). Only used with **-test | --test-input-files**

- **-score | --score `<score>`**  
  Store ranker's score for each object being ranked. Only used with **-rank | --rank-input-file**

- **-L2 | --l2 `<l2>`**  
  L2-norm regularization parameter. Defaults to `1E-10`

- **-hr | --must-have-relevant-docs**  
  Whether to ignore ranked lists without any relevant document. Defaults to `false`

- **-? | -h | --help**  
  Show help and usage information

### RankNet specific parameters

The following are parameters specific to using `RankNet` as the ranker.

- **-epoch | --epoch `<epoch>`**  
  The number of epochs to train

- **-layer | --layer `<layer>`**  
  The number of hidden layers

- **-node | --node `<node>`**  
  The number of hidden nodes per layer

- **-lr | --learning-rate `<learning-rate>`**  
  Learning rate

### RankBoost / MART / LambdaMART specific parameters

The following are parameters specific to using `RankBoost`, `MART`, or `LambdaMART` as the ranker.

- **-tc | --threshold-candidates `<threshold-candidates>`**  
  Number of threshold candidates to search or for tree splitting. -1 to use all feature values

### RankBoost / AdaRank specific parameters

The following are parameters specific to using `RankBoost` or `AdaRank` as the ranker.

- **-round | --round `<round>`**  
  The number of rounds to train

### AdaRank specific parameters

The following are parameters specific to using `AdaRank` as the ranker.

- **-noeq | --no-train-enqueue**  
  Train without enqueuing too-strong features. Defaults to `true`

- **-max | --max-selections `<max-selections>`**  
  The maximum number of times can a feature be consecutively selected without changing performance

### AdaRank / CoordinateAscent specific parameters

The following are parameters specific to using `AdaRank` or `CoordinateAscent` as the ranker.

- **-tolerance | --tolerance `<tolerance>`**  
  Tolerance between two consecutive rounds of learning. Defaults to `0.002` for AdaRank and `0.001` for CoordinateAscent

### CoordinateAscent specific parameters

The following are parameters specific to using `CoordinateAscent` as the ranker.

- **-r | --random-restarts `<random-restarts>`**  
  The number of random restarts. The default is `5`

- **-i | --iterations `<iterations>`**  
  The number of iterations to search in each dimension. the default is `25`

- **-reg | --regularization `<regularization>`**  
  Regularization parameter. The default is no regularization

### MART / LambdaMART specific parameters

The following are parameters specific to using `MART` or `LambdaMART` as the ranker.

- **-tree | --tree `<tree>`**  
  Number of trees. Default is `1000`

- **-leaf | --leaf `<leaf>`**  
  Number of leaves for each tree. Default is `10`

- **-shrinkage `<shrinkage>`**  
  Shrinkage, or learning rate. The default is `0.1`

- **-mls | --minimum-leaf-support `<minimum-leaf-support>`**  
  Minimum leaf support. Minimum number of samples each leaf has to contain. The default is `1`.

- **-estop | --early-stop `<early-stop>`**  
  Stop early when no improvement is observed on validation data in e consecutive rounds. The default is `100`.

### Random Forests specific parameters

The following are parameters specific to using `RandomForests` as the ranker.

- **-bag `<bag>`**  
  Number of bags [default: 300]

- **-srate | --sub-sampling-rate `<sub-sampling-rate>`**  
  Sub-sampling rate. Must be between 0 and 1. The default is `1`

- **-frate | --feature-sampling-rate `<feature-sampling-rate>`**  
  Feature sampling rate. Must be between 0 and 1. The default is `0.3`

- **-rtype | --random-forests-ranker `<LambdaMART|MART>`**  
  Ranker type to bag. Random Forests only support MART/LambdaMART. The default is `MART`

## Examples

Train a ranker using LambdaMART on the train data, using NDCG@10 as the metric score. Model is tested against
the test data and validated against the validation data. Finally, the model is saved to file:

```sh
dotnet-ranklib eval -ranker LambdaMART -train train.txt -test test.txt -validate validate.txt --train-metric NDCG@10 -save lambdamart_model.txt
```