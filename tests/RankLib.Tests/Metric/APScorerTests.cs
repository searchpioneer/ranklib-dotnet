using Xunit;
using RankLib.Metric;
using RankLib.Learning;

namespace RankLib.Tests.Metric;

public class APScorerTests
{
    private readonly APScorer _scorer;

    public APScorerTests() => _scorer = new APScorer();

    [Fact]
    public void Name_ReturnsCorrectValue() => Assert.Equal("MAP", _scorer.Name);

    [Theory]
    [InlineData(new[] {
        "1 qid:1 1:0.7 2:0.3 #doc1",  // Relevant
        "0 qid:1 1:0.3 2:0.7 #doc2",  // Not relevant
        "1 qid:1 1:0.1 2:0.9 #doc3"   // Relevant
    }, 0.833)] // (1/1 + 2/3) / 2 ≈ 0.833
    [InlineData(new[] {
        "0 qid:1 1:0.7 2:0.3 #doc1",  // Not relevant
        "1 qid:1 1:0.3 2:0.7 #doc2",  // Relevant
        "1 qid:1 1:0.1 2:0.9 #doc3"   // Relevant
    }, 0.583)] // (1/2 + 2/3) / 2 ≈ 0.583
    [InlineData(new[] {
        "0 qid:1 1:0.7 2:0.3 #doc1",  // Not relevant
        "0 qid:1 1:0.3 2:0.7 #doc2",  // Not relevant
        "0 qid:1 1:0.1 2:0.9 #doc3"   // Not relevant
    }, 0.0)]  // No relevant documents
    public void Score_CalculatesCorrectly(string[] documents, double expectedScore, double delta = 0.001)
    {
        var rankList = CreateRankList(documents);
        var score = _scorer.Score(rankList);
        Assert.Equal(expectedScore, score, delta);
    }

    [Fact]
    public void Score_EmptyList_ReturnsZero()
    {
        var rankList = new RankList(new List<DataPoint>());
        var score = _scorer.Score(rankList);
        Assert.Equal(0.0, score);
    }

    [Fact]
    public void LoadExternalRelevanceJudgment_ValidFile_LoadsCorrectly()
    {
        using var tempFile = new TempFile();
        using (var writer = tempFile.GetWriter())
        {
	         // 3 relevant documents for qid:1
	         // qid <unused> document judgment
	         writer.WriteLine("1 qid:1 1 2");
	         writer.WriteLine("1 qid:1 3 1");
	         writer.WriteLine("1 qid:1 4 1");
	         writer.WriteLine("1 qid:1 5 0");
	         writer.WriteLine("2 qid:2 4 1");
	         writer.WriteLine("2 qid:2 5 0");
        }

        _scorer.LoadExternalRelevanceJudgment(tempFile.Path);

        // Test with a rank list that should match the external judgments
        var documents = new[]
        {
            "2 qid:1 1:0.7 2:0.3 #doc1",  // Relevant
            "0 qid:1 1:0.3 2:0.7 #doc2",  // Not relevant
            "1 qid:1 1:0.1 2:0.9 #doc3"   // Relevant
        };
        var rankList = CreateRankList(documents);
        var score = _scorer.Score(rankList);

        // Expected: (1/1 + 2/3) / 3 ≈ 0.556 (because external file shows 3 relevant docs for qid:1)
        Assert.Equal(0.556, score, 0.001);
    }

    [Fact]
    public void LoadExternalRelevanceJudgment_InvalidFile_ThrowsException() =>
	    Assert.Throws<Utilities.RankLibException>(() =>
		    _scorer.LoadExternalRelevanceJudgment("nonexistent_file.txt"));

    [Fact]
    public void SwapChange_CalculatesCorrectly()
    {
        var documents = new[]
        {
            "1 qid:1 1:0.7 2:0.3 #doc1",  // Relevant
            "0 qid:1 1:0.3 2:0.7 #doc2",  // Not relevant
            "1 qid:1 1:0.1 2:0.9 #doc3"   // Relevant
        };
        var rankList = CreateRankList(documents);
        var changes = _scorer.SwapChange(rankList);

        // Swapping positions 0 and 1 (relevant with non-relevant)
        // Original: (1/1 + 2/3) / 2 = 0.833
        // After swap: (1/2 + 2/3) / 2 = 0.5833
        // Change should be: 0.5833 - 0.833 = -0.25
        Assert.Equal(-0.25, changes[0][1], 0.001);
        Assert.Equal(-0.25, changes[1][0], 0.001);
    }

    [Fact]
    public void SwapChange_NoRelevantDocs_ReturnsZeroChanges()
    {
        var documents = new[]
        {
            "0 qid:1 1:0.7 2:0.3 #doc1",
            "0 qid:1 1:0.3 2:0.7 #doc2",
            "0 qid:1 1:0.1 2:0.9 #doc3"
        };
        var rankList = CreateRankList(documents);
        var changes = _scorer.SwapChange(rankList);

        for (var i = 0; i < changes.Length; i++)
            for (var j = 0; j < changes[i].Length; j++)
                Assert.Equal(0.0, changes[i][j]);
    }

    [Fact]
    public void SwapChange_AllRelevantDocs_CalculatesCorrectly()
    {
        var documents = new[]
        {
            "1 qid:1 1:0.7 2:0.3 #doc1",
            "2 qid:1 1:0.3 2:0.7 #doc2",
            "1 qid:1 1:0.1 2:0.9 #doc3"
        };
        var rankList = CreateRankList(documents);
        var changes = _scorer.SwapChange(rankList);

        // Since all documents are relevant, swapping should have no impact
        for (var i = 0; i < changes.Length; i++)
        {
	        for (var j = 0; j < changes[i].Length; j++)
		        Assert.Equal(0.0, changes[i][j]);
        }
    }

    [Fact]
    public void Score_WithExternalJudgments_HandlesMissingQuery()
    {
        using var tempFile = new TempFile();
        using (var writer = tempFile.GetWriter())
	        writer.WriteLine("1 qid:1 docid:1 1");

        _scorer.LoadExternalRelevanceJudgment(tempFile.Path);

        var documents = new[]
        {
            "1 qid:999 1:0.7 2:0.3 #doc1"  // Query not in external judgments
        };
        var rankList = CreateRankList(documents);
        var score = _scorer.Score(rankList);

        // Should fall back to counting relevant docs in the actual list
        Assert.Equal(1.0, score);
    }

    private static RankList CreateRankList(string[] documents)
    {
        var dataPoints = documents.Select(DataPoint (doc) => new DenseDataPoint(doc)).ToList();
        return  new RankList(dataPoints);
    }
}
