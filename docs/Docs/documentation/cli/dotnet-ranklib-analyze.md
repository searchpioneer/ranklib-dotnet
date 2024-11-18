# dotnet-ranklib analyze

Analyzes the performance of saved models by comparing them against a baseline.

## Usage

```sh
dotnet-ranklib analyze [options]
```

## Options

- **--all `<all>`**  
  Directory of performance files (one per system)

- **--base `<base>`**  
  Performance file for the baseline. **Must** be in the same directory as the other performance files.

- **--np `<np>`**  
  Number of permutations (Fisher randomization test) [default: 10000]

- **-? | -h | --help**  
  Show help and usage information