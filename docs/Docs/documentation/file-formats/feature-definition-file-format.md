# Feature Definition File Format

The feature definition file defines the features used in the 
ranking model. The file format is as follows:

```text
<line> .=. <featureid> <description>
<featureid> .=. <positive integer>
<description> .=. <string>
```

where

- `<featureid>`

  The feature's unique identifier.

  > [!NOTE]
  > Feature identifiers should match the identifiers referenced in the training data file.

- `<description>`

  A short description or explanation of the feature.

A feature file can contain comments by starting a line with `#`.

## Example

The following example has five features:

```text
#   Movies data set features - revision 1
1   BM25 score for the title field
2   BM25 score for the untokenized title field
3   BM25 score for the actors field
4   BM25 score for the untokenized actors field
5   Popularity score derived from click data
```