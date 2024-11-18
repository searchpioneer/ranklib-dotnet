# Overview

`dotnet-ranklib` is used to train and evaluate rankers using Learning to Rank (LTR). It has the ability to

1. Train and evaluate rankers, allowing ranker models to be saved, and subsequently loaded into applications
   to perform ranking.
2. Analyze and compare rankers against a baseline, which is useful for understanding improvements.
3. Display the feature statistics for saved ranker models, to understand their relative impact on scoring.

## Installation

It's recommended to add as a global .NET command line tool

```sh
dotnet tool install -g RankLib.Cli
```

After installation, the tool can be run with

```sh
dotnet-ranklib
```

to see the help documentation and available commands.

## `dotnet-ranklib` commands

<div class="commands">

| Command                               | Description                                                                                                      |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [dotnet-ranklib eval](dotnet-ranklib-eval.md)       | Trains and evaluates a ranker, or evaluates a previously saved ranker model.                                     |
| [dotnet-ranklib analyze](dotnet-ranklib-analyze.md) | Analyze performance comparison of saved models against a baseline.                                               |
| [dotnet-ranklib combine](dotnet-ranklib-combine.md) | Combines ensembles from files in a directory into one file.                                                      |
| [dotnet-ranklib prepare](dotnet-ranklib-prepare.md) | Split the input sample set into k chunks (folds) of roughly equal size and create train/test data for each fold. |
| [dotnet-ranklib stats](dotnet-ranklib-stats.md)     | Feature statistics for the given model.                                                                          |

</div>

## Tab Completion

Tab completion can be enabled for the ranklib CLI by following the 
[System.CommandLine instructions](https://learn.microsoft.com/en-us/dotnet/standard/commandline/tab-completion#enable-tab-completion):

1. Install the [dotnet-suggest](https://nuget.org/packages/dotnet-suggest) global tool.
2. Add the appropriate shim script to your shell profile. You may have to create a shell profile file. 
   The shim script forwards completion requests from your shell to the dotnet-suggest tool, 
   which delegates to the appropriate ranklib CLI app.

    1. For bash, add the contents of [dotnet-suggest-shim.bash](https://github.com/dotnet/command-line-api/blob/main/src/System.CommandLine.Suggest/dotnet-suggest-shim.bash) to `~/.bash_profile`.
    2. For zsh, add the contents of [dotnet-suggest-shim.zsh](https://github.com/dotnet/command-line-api/blob/main/src/System.CommandLine.Suggest/dotnet-suggest-shim.zsh) to `~/.zshrc`.
    3. For PowerShell, add the contents of [dotnet-suggest-shim.ps1](https://github.com/dotnet/command-line-api/blob/main/src/System.CommandLine.Suggest/dotnet-suggest-shim.ps1) 
       to your PowerShell profile. You can find the expected path to your PowerShell profile by running the following command in your console:

        ```powershell
        echo $PROFILE
        ```

