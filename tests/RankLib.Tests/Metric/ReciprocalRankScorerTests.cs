using RankLib.Learning;
using RankLib.Metric;
using Xunit;

namespace RankLib.Tests.Metric;

public class ReciprocalRankScorerTests
{
	private readonly ReciprocalRankScorer _scorer;

	public ReciprocalRankScorerTests() => _scorer = new ReciprocalRankScorer();

	[Fact]
	public void Name_ReturnsCorrectFormat() => Assert.Equal("RR@0", _scorer.Name);

	[Theory]
	[InlineData(new[] {
		"3 qid:1 1:0.7 2:0.3 # doc1",  // Relevant (label > 0)
        "0 qid:1 1:0.3 2:0.7 # doc2",  // Not relevant
        "0 qid:1 1:0.1 2:0.9 # doc3"   // Not relevant
    }, 1.0)] // First position is relevant -> RR = 1/1
	[InlineData(new[] {
		"0 qid:1 1:0.7 2:0.3 # doc1",  // Not relevant
        "2 qid:1 1:0.3 2:0.7 # doc2",  // Relevant
        "0 qid:1 1:0.1 2:0.9 # doc3"   // Not relevant
    }, 0.5)] // Second position is relevant -> RR = 1/2
	[InlineData(new[] {
		"0 qid:1 1:0.7 2:0.3 # doc1",  // Not relevant
        "0 qid:1 1:0.3 2:0.7 # doc2",  // Not relevant
        "1 qid:1 1:0.1 2:0.9 # doc3"   // Relevant
    }, 0.333, 0.001)] // Third position is relevant -> RR = 1/3
	[InlineData(new[] {
		"0 qid:1 1:0.7 2:0.3 # doc1",  // Not relevant
        "0 qid:1 1:0.3 2:0.7 # doc2",  // Not relevant
        "0 qid:1 1:0.1 2:0.9 # doc3"   // Not relevant
    }, 0.0)] // No relevant docs -> RR = 0
	public void Score_CalculatesCorrectly(string[] documents, double expectedScore, double delta = 0.0)
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
	public void SwapChange_FirstPositionRelevant_CalculatesCorrectChanges()
	{
		var documents = new[]
		{
			"2 qid:1 1:0.7 2:0.3 # doc1",  // Relevant
            "0 qid:1 1:0.3 2:0.7 # doc2",  // Not relevant
            "0 qid:1 1:0.1 2:0.9 # doc3"   // Not relevant
        };
		var rankList = CreateRankList(documents);
		var changes = _scorer.SwapChange(rankList);

		// Swapping first (relevant) with second (non-relevant)
		Assert.Equal(-0.5, changes[0][1], 3); // 1.0 -> 0.5
		Assert.Equal(-0.5, changes[1][0], 3);

		// Swapping first (relevant) with third (non-relevant)
		Assert.Equal(-0.667, changes[0][2], 3); // 1.0 -> 0.333
		Assert.Equal(-0.667, changes[2][0], 3);
	}

	[Fact]
	public void SwapChange_SecondPositionRelevant_CalculatesCorrectChanges()
	{
		var documents = new[]
		{
			"0 qid:1 1:0.7 2:0.3 # doc1",  // Not relevant
            "3 qid:1 1:0.3 2:0.7 # doc2",  // Relevant
            "0 qid:1 1:0.1 2:0.9 # doc3"   // Not relevant
        };
		var rankList = CreateRankList(documents);
		var changes = _scorer.SwapChange(rankList);

		// Swapping first (non-relevant) with second (relevant)
		Assert.Equal(0.5, changes[0][1], 3); // 0.5 -> 1.0
		Assert.Equal(0.5, changes[1][0], 3);

		// Swapping second (relevant) with third (non-relevant)
		Assert.Equal(-0.167, changes[1][2], 3); // 0.5 -> 0.333
		Assert.Equal(-0.167, changes[2][1], 3);
	}

	[Fact]
	public void SwapChange_NoRelevantDocs_ReturnsZeroChanges()
	{
		var documents = new[]
		{
			"0 qid:1 1:0.7 2:0.3 # doc1",  // Not relevant
            "0 qid:1 1:0.3 2:0.7 # doc2",  // Not relevant
            "0 qid:1 1:0.1 2:0.9 # doc3"   // Not relevant
        };
		var rankList = CreateRankList(documents);
		var changes = _scorer.SwapChange(rankList);

		for (var i = 0; i < changes.Length; i++)
		{
			for (var j = 0; j < changes[i].Length; j++)
				Assert.Equal(0.0, changes[i][j]);
		}
	}

	[Fact]
	public void SwapChange_MultipleRelevantDocs_HandlesCorrectly()
	{
		var documents = new[]
		{
			"0 qid:1 1:0.7 2:0.3 # doc1",  // Not relevant
            "2 qid:1 1:0.3 2:0.7 # doc2",  // Relevant
            "1 qid:1 1:0.1 2:0.9 # doc3",  // Also relevant
            "0 qid:1 1:0.2 2:0.8 # doc4"   // Not relevant
        };
		var rankList = CreateRankList(documents);
		var changes = _scorer.SwapChange(rankList);

		// Swapping first (non-relevant) with second (first relevant) changes
		Assert.Equal(0.5, changes[0][1], 3); // 0.5 -> 1.0
		Assert.Equal(0.5, changes[1][0], 3);

		// Second relevant document shouldn't affect changes when swapping with first relevant
		Assert.Equal(0, changes[1][2], 3);
		Assert.Equal(0, changes[2][1], 3);

		// Second relevant document swapped with second non-relevant, decreases
		Assert.Equal(-0.1667, changes[1][3], 3);
		Assert.Equal(-0.1667, changes[3][1], 3);
	}

	[Fact]
	public void Score_WithDifferentFeatureValues_MaintainsCorrectRanking()
	{
		var documents = new[]
		{
			"0 qid:1 1:0.9 2:0.8 3:0.7 # high_features_not_relevant",
			"2 qid:1 1:0.3 2:0.2 3:0.1 # low_features_but_relevant",
			"0 qid:1 1:0.6 2:0.5 3:0.4 # medium_features_not_relevant"
		};
		var rankList = CreateRankList(documents);
		var score = _scorer.Score(rankList);

		// Even though first document has higher feature values,
		// RR should only care about relevance label (which is at position 2)
		Assert.Equal(0.5, score);
	}

	[Fact]
	public void Score_WithSameRelevanceLabel_ConsidersFirst()
	{
		var documents = new[]
		{
			"2 qid:1 1:0.3 2:0.7 # relevant1",
			"2 qid:1 1:0.8 2:0.9 # relevant2",
			"0 qid:1 1:0.1 2:0.2 # not_relevant"
		};
		var rankList = CreateRankList(documents);
		var score = _scorer.Score(rankList);

		// Should return 1.0 because first relevant document is at position 1,
		// even though there's another document with same relevance
		Assert.Equal(1.0, score);
	}

	private static RankList CreateRankList(string[] documents)
	{
		var dataPoints = documents
			.Select(DataPoint (doc) => new DenseDataPoint(doc))
			.ToList();
		return new RankList(dataPoints);
	}
}
