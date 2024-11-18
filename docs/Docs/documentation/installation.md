# Installation

RankLib for .NET is available as both a command line tool for training and evaluating rankers, as well as a library for
incorporating into solutions.

## Library

To add as a library to an existing project

```sh
dotnet add package SearchPioneer.RankLib --prerelease
```

## Command Line Tool

To add as a global .NET command line tool

```sh
dotnet tool install -g SearchPioneer.RankLib.Cli --prerelease
```

To see all the commands supported by the command line tool

```sh
dotnet-ranklib --help
```