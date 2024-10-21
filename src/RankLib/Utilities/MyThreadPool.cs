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

}
