using System.Runtime.CompilerServices;

namespace RankLib.Utilities;

internal static class ListExtensions
{
	private static readonly Random Random = ThreadsafeSeedableRandom.Shared;

	/// <summary>
	/// Shuffles a list using Fisher-Yates shuffle
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
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
