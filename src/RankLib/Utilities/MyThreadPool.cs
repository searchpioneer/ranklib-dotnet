namespace RankLib.Utilities;
public class MyThreadPool
{
	private readonly SemaphoreSlim _semaphore;
	private readonly int _size;
	private static MyThreadPool? Singleton;
	private static readonly object LockObj = new();

	private MyThreadPool(int size)
	{
		_size = size;
		_semaphore = new SemaphoreSlim(size, size);
	}

	public static MyThreadPool Instance
	{
		get
		{
			if (Singleton == null)
			{
				lock (LockObj)
				{
					if (Singleton == null)
					{
						Init(Environment.ProcessorCount);
					}
				}
			}

			return Singleton!;
		}
	}

	public static void Init(int poolSize) => Singleton = new MyThreadPool(poolSize);

	public int Size() => _size;

	public WorkerThread[] Execute(WorkerThread worker, int nTasks)
	{
		var p = Instance;
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
		for (var i = 0; i < _size; i++)
		{
			try
			{
				_semaphore.Wait();
			}
			catch (Exception ex)
			{
				throw RankLibException.Create("Error in MyThreadPool.Await(): ", ex);
			}
		}

		_semaphore.Release(_size);
	}

	public int[] Partition(int listSize)
	{
		var nChunks = Math.Min(listSize, _size);
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
			_semaphore.Wait();
			Task.Run(() =>
			{
				try
				{
					task.Run();
				}
				finally
				{
					_semaphore.Release();
				}
			});
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in MyThreadPool.Execute(): ", ex);
		}
	}
}
