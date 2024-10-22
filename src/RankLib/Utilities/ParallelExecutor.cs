namespace RankLib.Utilities;

public static class ParallelExecutor
{
	public static async Task<TWorker[]> ExecuteAsync<TWorker>(TWorker worker, int nTasks, int maxDegreeOfParallelism = -1, CancellationToken cancellationToken = default)
		where TWorker : WorkerThread
	{
		if (maxDegreeOfParallelism <= 0)
			maxDegreeOfParallelism = Environment.ProcessorCount;

		var partition = Partition(nTasks, maxDegreeOfParallelism);
		var workers = Enumerable.Range(0, partition.Length - 1)
			.Select(i =>
			{
				var w = (TWorker)worker.Clone();
				w.Set(partition[i], partition[i + 1] - 1);
				return w;
			})
			.ToList();

		await Parallel.ForEachAsync(workers,
			new ParallelOptions
			{
				MaxDegreeOfParallelism = maxDegreeOfParallelism,
				CancellationToken = cancellationToken
			},
			async (w, ct) =>
			{
				await w.RunAsync().ConfigureAwait(false);
			});

		return workers.ToArray();
	}

	public static async Task ExecuteAsync(IEnumerable<RunnableTask> tasks, int maxDegreeOfParallelism = -1, CancellationToken cancellationToken = default)
	{
		if (maxDegreeOfParallelism <= 0)
			maxDegreeOfParallelism = Environment.ProcessorCount;

		await Parallel.ForEachAsync(tasks,
			new ParallelOptions
			{
				MaxDegreeOfParallelism = maxDegreeOfParallelism,
				CancellationToken = cancellationToken
			},
			async (task, ct) =>
			{
				await task.RunAsync().ConfigureAwait(false);
			});
	}

	public static IEnumerable<Range> PartitionEnumerable(int listSize, int nChunks)
	{
		nChunks = Math.Min(listSize, nChunks);
		var chunkSize = listSize / nChunks;
		var mod = listSize % nChunks;
		var current = 0;

		for (var i = 0; i < nChunks; i++)
		{
			var size = chunkSize + (i < mod ? 1 : 0);
			var end = current + size;
			yield return new Range(current, end - 1);
			current = end;
		}
	}

	public static int[] Partition(int listSize, int nChunks)
	{
		nChunks = Math.Min(listSize, nChunks);
		var chunkSize = listSize / nChunks;
		var mod = listSize % nChunks;
		var partition = new int[nChunks + 1];
		partition[0] = 0;
		for (var i = 1; i <= nChunks; i++)
		{
			partition[i] = partition[i - 1] + chunkSize + (i <= mod ? 1 : 0);
		}
		return partition;
	}
}
