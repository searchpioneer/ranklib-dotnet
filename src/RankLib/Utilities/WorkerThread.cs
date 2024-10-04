namespace RankLib.Utilities;

public abstract class WorkerThread : RunnableTask
{
	protected int start = -1;
	protected int end = -1;

	public void Set(int start, int end)
	{
		this.start = start;
		this.end = end;
	}

	public abstract WorkerThread Clone();
}

public abstract class RunnableTask
{
	public abstract void Run();
}
