namespace RankLib.Utilities;

using System;

internal class ThreadsafeSeedableRandom : Random
{
	private static readonly Lazy<Random> LazyRandom =
		new(() => Seed == null ? Random.Shared : new ThreadsafeSeedableRandom(Seed.Value));

	public static int? Seed { get; set; }

	public static new Random Shared => LazyRandom.Value;

	private readonly object _locker = new();

	internal ThreadsafeSeedableRandom(int seed)
		: base(seed)
	{
	}

	public override int Next()
	{
		lock (_locker)
			return base.Next();
	}

	public override int Next(int maxValue)
	{
		lock (_locker)
			return base.Next(maxValue);
	}

	public override int Next(int minValue, int maxValue)
	{
		lock (_locker)
			return base.Next(minValue, maxValue);
	}

	public override void NextBytes(byte[] buffer)
	{
		lock (_locker)
			base.NextBytes(buffer);
	}

	public override void NextBytes(Span<byte> buffer)
	{
		lock (_locker)
			base.NextBytes(buffer);
	}

	public override double NextDouble()
	{
		lock (_locker)
			return base.NextDouble();
	}

	public override long NextInt64()
	{
		lock (_locker)
			return base.NextInt64();
	}

	public override long NextInt64(long maxValue)
	{
		lock (_locker)
			return base.NextInt64(maxValue);
	}

	public override long NextInt64(long minValue, long maxValue)
	{
		lock (_locker)
			return base.NextInt64(minValue, maxValue);
	}

	public override float NextSingle()
	{
		lock (_locker)
			return base.NextSingle();
	}

	protected override double Sample()
	{
		lock (_locker)
			return base.Sample();
	}

}
