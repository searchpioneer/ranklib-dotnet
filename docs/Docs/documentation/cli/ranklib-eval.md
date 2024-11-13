# ranklib eval

Trains and evaluates a ranker, or evaluates a previously saved ranker model.

## Usage

```sh
ranklib eval [options]
```

## Options

- **-train | --train-input-file `<train-input-file>`**  
  Training data file

- **-ranker `<ranker>`**  
  Ranking algorithm to use 

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


  [default: CoordinateAscent]

- **-feature | --feature-description-input-file `<feature-description-input-file>`**  
  Feature description file. List features to be considered by the learner, each on a separate line. If not specified, all features will be used.

- **-metric2t | --train-metric `<train-metric>`**  
  Metric to optimize on the training data. Supports MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k. [default: ERR@10]

- **-gmax | --max-label `<max-label>`**  
  Highest judged relevance label. It affects the calculation of ERR (default=4, i.e. 5-point scale [0,1,2,3,4] where value used is 2^gmax)

- **-qrel | --query-relevance-input-file `<query-relevance-input-file>`**  
  TREC-style relevance judgment file

- **-missingZero | --missing-zero**  
  Substitute zero for missing feature values rather than throwing an exception.

- **-validate | --validate-file `<validate-file>`**  
  Specify if you want to tune your system on the validation data

- **-tvs | --train-validation-split `<train-validation-split>`**  
  If you don't have separate validation data, use this to set train-validation split to be. Must be between 0 and 1, where train=<value> and validate=(1.0 - <value>)

- **-save | --model-output-file `<model-output-file>`**  
  Save the model learned [default: no save]

- **-test | --test-input-files `<test-input-files>`**  
  Specify if you want to evaluate the trained model on this data

- **-tts | --train-test-split `<train-test-split>`**  
  Set train-test split. Must be between 0 and 1, where train=<value> and test=(1.0 - <value>). Overrides --train-validation-split (-tvs)

- **-metric2T | --test-metric `<test-metric>`**  
  Metric to evaluate on the test data [default: same as -metric2t]

- **-norm `<normalizer>`**  
  Type of normalizer to use to normalize all feature vectors 
 
  - `Linear`
  - `Sum`
  - `ZScore`


  [default: no normalization]

- **-kcv | --cross-validation-folds `<cross-validation-folds>`**  
  Specify how many folds to perform for k-fold cross validation using the specified training data. Defaults to no k-fold cross validation. [default: -1]

- **-kcvmd | --cross-validation-output-directory `<cross-validation-output-directory>`**  
  Directory for models trained via cross-validation

- **-kcvmn | --cross-validation-model-name `<cross-validation-model-name>`**  
  Name for model learned in each fold. It will be prefixed with the fold-number

- **-load | --model-input-files `<model-input-files>`**  
  Load saved model file for evaluation

- **-thread `<thread>`**  
  Number of threads to use. [default: all available processors]

- **-rank | --rank-input-file `<rank-input-file>`**  
  Rank the samples in the specified file (specify either this or -test but not both)

- **-indri | --indri-ranking-output-file `<indri-ranking-output-file>`**  
  Indri ranking file

- **-sparse | --use-sparse-representation**  
  Use data points with sparse representation

- **-idv | --individual-ranklist-performance-output-file `<individual-ranklist-performance-output-file>`**  
  Individual rank list model performance (in test metric). Has to be used with -test

- **-score `<score>`**  
  Store ranker's score for each object being ranked. Has to be used with -rank

- **-L2 | --l2 `<l2>`**  
  L2-norm regularization parameter [default: 1E-10]

- **-hr | --must-have-relevant-docs**  
  Whether to ignore ranked list without any relevant document [default: False]

- **-? | -h | --help**  
  Show help and usage information

### RankNet specific parameters

- **-epoch `<epoch>`**  
  The number of epochs to train

- **-layer `<layer>`**  
  The number of hidden layers

- **-node `<node>`**  
  The number of hidden nodes per layer

- **-lr | --learning-rate `<learning-rate>`**  
  Learning rate

### RankBoost / MART / LambdaMART specific parameters

- **-tc | --threshold-candidates `<threshold-candidates>`**  
  Number of threshold candidates to search or for tree splitting. -1 to use all feature values

### RankBoost / AdaRank specific parameters

- **-round `<round>`**  
  The number of rounds to train

### AdaRank specific parameters

- **-noeq | --no-train-enqueue**  
  Train without enqueuing too-strong features. Defaults to true

- **-max | --max-selections `<max-selections>`**  
  The maximum number of times can a feature be consecutively selected without changing performance

### AdaRank / CoordinateAscent specific parameters

- **-tolerance `<tolerance>`**  
  Tolerance between two consecutive rounds of learning. Defaults to 0.002 for AdaRank and 0.001 for CoordinateAscent

### CoordinateAscent specific parameters

- **-r | --random-restarts `<random-restarts>`**  
  The number of random restarts [default: 5]

- **-i | --iterations `<iterations>`**  
  The number of iterations to search in each dimension [default: 25]

- **-reg | --regularization `<regularization>`**  
  Regularization parameter [default: no regularization]

### MART / LambdaMART specific parameters

- **-tree `<tree>`**  
  Number of trees [default: 1000]

- **-leaf `<leaf>`**  
  Number of leaves for each tree [default: 10]

- **-shrinkage `<shrinkage>`**  
  Shrinkage, or learning rate [default: 0.1]

- **-mls | --minimum-leaf-support `<minimum-leaf-support>`**  
  Minimum leaf support. Minimum number of samples each leaf has to contain [default: 1]

- **-estop | --early-stop `<early-stop>`**  
  Stop early when no improvement is observed on validation data in e consecutive rounds [default: 100]

### Random Forests specific parameters

- **-bag `<bag>`**  
  Number of bags [default: 300]

- **-srate | --sub-sampling-rate `<sub-sampling-rate>`**  
  Sub-sampling rate [default: 1]

- **-frate | --feature-sampling-rate <feature-sampling-rate>**  
  Feature sampling rate [default: 0.3]

- **-rtype | --random-forests-ranker <LambdaMART|MART>**  
  Ranker type to bag. Random Forests only support MART/LambdaMART [default: MART]
