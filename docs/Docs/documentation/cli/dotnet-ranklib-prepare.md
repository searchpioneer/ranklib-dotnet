# dotnet-ranklib prepare

Splits the input sample set into k chunks (folds) of roughly equal size and create train/ test data for each fold.

## Usage

```sh
dotnet-ranklib prepare [options]
```

## Options

- **--input `<input>`** (REQUIRED)  
  Source data (ranked lists)

- **--output `<output>`** (REQUIRED)  
  The output directory

- **--shuffle**  
  Create a copy of the input file in which the ordering of all ranked lists (e.g., queries) is randomized.

- **--tvs `<tvs>`**  
  Train-validation split ratio (x)(1.0 - x)

- **--tts `<tts>`**  
  Train-test split ratio (x)(1.0 - x)

- **--k `<k>`**  
  The number of folds

- **-? | -h | --help**  
  Show help and usage information
