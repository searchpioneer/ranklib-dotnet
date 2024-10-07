namespace RankLib.Features;

internal static class ListExtensions
{
	private static readonly Random Random = Random.Shared;

	public static void Shuffle<T>(this IList<T> list)
	{
		var n = list.Count;
		while (n > 1)
		{
			n--;
			var k = Random.Next(n + 1);
			(list[k], list[n]) = (list[n], list[k]);
		}
	}
}
