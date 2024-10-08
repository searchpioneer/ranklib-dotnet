namespace RankLib.Utilities;
public class MyThreadPool
{
	private readonly SemaphoreSlim semaphore;
	private readonly int size;
	private static MyThreadPool? singleton;
	private static readonly object lockObj = new();

	private MyThreadPool(int size)
	{
		this.size = size;
		semaphore = new SemaphoreSlim(size, size);
	}

	public static MyThreadPool GetInstance()
	{
		if (singleton == null)
		{
			lock (lockObj)
			{
				if (singleton == null)
				{
					Init(Environment.ProcessorCount);
				}
			}
		}
		return singleton!;
	}

	public static void Init(int poolSize) => singleton = new MyThreadPool(poolSize);

	public int Size() => size;

	public WorkerThread[] Execute(WorkerThread worker, int nTasks)
	{
		var p = GetInstance();
		var partition = p.Partition(nTasks);
		var workers = new WorkerThread[partition.Length - 1];

		for (var i = 0; i < partition.Length - 1; i++)
		{
			var w = worker.Clone();
			w.Set(partition[i], partition[i + 1] - 1);
			workers[i] = w;
			p.Execute(w);
		}

		Await();
		return workers;
	}

	public void Await()
	{
		for (var i = 0; i < size; i++)
		{
			try
			{
				semaphore.Wait();
			}
			catch (Exception ex)
			{
				throw RankLibException.Create("Error in MyThreadPool.Await(): ", ex);
			}
		}

		semaphore.Release(size);
	}

	public int[] Partition(int listSize)
	{
		var nChunks = Math.Min(listSize, size);
		var chunkSize = listSize / nChunks;
		var mod = listSize % nChunks;

		var partition = new int[nChunks + 1];
		partition[0] = 0;

		for (var i = 1; i <= nChunks; i++)
		{
			partition[i] = partition[i - 1] + chunkSize + ((i <= mod) ? 1 : 0);
		}

		return partition;
	}

	public void Execute(RunnableTask task)
	{
		try
		{
			semaphore.Wait();
			Task.Run(() =>
			{
				try
				{
					task.Run();
				}
				finally
				{
					semaphore.Release();
				}
			});
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in MyThreadPool.Execute(): ", ex);
		}
	}
}
