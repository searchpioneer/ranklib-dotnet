namespace RankLib.Utilities;

internal static class Partitioner
{
	public static IEnumerable<Range> PartitionEnumerable(int listSize, int partitionCount)
	{
		partitionCount = Math.Min(listSize, partitionCount);
		var partitionSize = listSize / partitionCount;
		var mod = listSize % partitionCount;
		var current = 0;

		for (var i = 0; i < partitionCount; i++)
		{
			var size = partitionSize + (i < mod ? 1 : 0);
			var end = current + size;
			yield return new Range(current, end - 1);
			current = end;
		}
	}

	public static int[] Partition(int listSize, int partitionCount)
	{
		partitionCount = Math.Min(listSize, partitionCount);
		var partitionSize = listSize / partitionCount;
		var mod = listSize % partitionCount;
		var partition = new int[partitionCount + 1];
		partition[0] = 0;
		for (var i = 1; i <= partitionCount; i++)
			partition[i] = partition[i - 1] + partitionSize + (i <= mod ? 1 : 0);

		return partition;
	}
}
