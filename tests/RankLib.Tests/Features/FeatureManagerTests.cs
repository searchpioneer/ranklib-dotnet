using RankLib.Features;
using Xunit;

namespace RankLib.Tests.Features;

public class FeatureManagerTests
{
	[Fact]
	public void CanReadInput()
	{
		var featureManager = new FeatureManager();
		var rankLists = featureManager.ReadInput("sample_judgments_with_features.txt");

		Assert.NotEmpty(rankLists);
		Assert.Equal(3, rankLists.Count);

		Assert.Collection(rankLists,
			list1 =>
			{
				Assert.Equal("1", list1.Id);
				Assert.Equal(10, list1.Count);
				Assert.Equal(2, list1.FeatureCount);
				Assert.Collection(list1,
					point1 =>
					{
						Assert.Equal(4, point1.Label);
						Assert.Equal(2, point1.FeatureCount);
						Assert.Equal("# 7555\trambo", point1.Description);
						Assert.Equal(12.318474f, point1.GetFeatureValue(1));
						Assert.Equal(10.573917f, point1.GetFeatureValue(2));
					},
					point2 =>
					{
						Assert.Equal(3, point2.Label);
						Assert.Equal(2, point2.FeatureCount);
						Assert.Equal("# 1370\trambo", point2.Description);
						Assert.Equal(10.357876f, point2.GetFeatureValue(1));
						Assert.Equal(11.95039f, point2.GetFeatureValue(2));
					},
					point3 =>
					{
						Assert.Equal(3, point3.Label);
						Assert.Equal(2, point3.FeatureCount);
						Assert.Equal("# 1369\trambo", point3.Description);
						Assert.Equal(7.0105133f, point3.GetFeatureValue(1));
						Assert.Equal(11.220095f, point3.GetFeatureValue(2));
					},
					point4 => { },
					point5 => { },
					point6 => { },
					point7 => { },
					point8 => { },
					point9 => { },
					point10 => { });
			},
			list2 => { },
			list3 => { });
	}
}
