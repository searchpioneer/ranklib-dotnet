namespace RankLib.Utilities;

public static class Sorter
{
	/// <summary>
	/// Sort a double array using Interchange sort.
	/// </summary>
	/// <param name="sortVal">The double array to be sorted.</param>
	/// <param name="asc">TRUE to sort ascendingly, FALSE to sort descendingly.</param>
	/// <returns>The sorted indexes.</returns>
	public static int[] Sort(double[] sortVal, bool asc)
	{
		var freqIdx = new int[sortVal.Length];
		for (var i = 0; i < sortVal.Length; i++)
			freqIdx[i] = i;

		for (var i = 0; i < sortVal.Length - 1; i++)
		{
			var max = i;
			for (var j = i + 1; j < sortVal.Length; j++)
			{
				if (asc)
				{
					if (sortVal[freqIdx[max]] > sortVal[freqIdx[j]])
						max = j;
				}
				else
				{
					if (sortVal[freqIdx[max]] < sortVal[freqIdx[j]])
						max = j;
				}
			}
			// Swap
			(freqIdx[i], freqIdx[max]) = (freqIdx[max], freqIdx[i]);
		}
		return freqIdx;
	}

	public static int[] Sort(float[] sortVal, bool asc)
	{
		var freqIdx = new int[sortVal.Length];
		for (var i = 0; i < sortVal.Length; i++)
			freqIdx[i] = i;

		for (var i = 0; i < sortVal.Length - 1; i++)
		{
			var max = i;
			for (var j = i + 1; j < sortVal.Length; j++)
			{
				if (asc)
				{
					if (sortVal[freqIdx[max]] > sortVal[freqIdx[j]])
						max = j;
				}
				else
				{
					if (sortVal[freqIdx[max]] < sortVal[freqIdx[j]])
						max = j;
				}
			}
			// Swap
			(freqIdx[i], freqIdx[max]) = (freqIdx[max], freqIdx[i]);
		}
		return freqIdx;
	}

	/// <summary>
	/// Sort an integer array using Quick Sort.
	/// </summary>
	/// <param name="sortVal">The integer array to be sorted.</param>
	/// <param name="asc">TRUE to sort ascendingly, FALSE to sort descendingly.</param>
	/// <returns>The sorted indexes.</returns>
	public static int[] Sort(int[] sortVal, bool asc) => QSort(sortVal, asc);

	public static int[] Sort(List<int> sortVal, bool asc) => QSort(sortVal, asc);

	public static int[] SortString(List<string> sortVal, bool asc) => QSortString(sortVal, asc);

	public static int[] SortLong(List<long> sortVal, bool asc) => QSortLong(sortVal, asc);

	public static int[] SortDesc(List<double> sortVal) => QSortDouble(sortVal, false);

	private static int[] QSort(List<int> l, bool asc)
	{
		var idx = new int[l.Count];
		var idxList = new List<int>();
		for (var i = 0; i < l.Count; i++)
		{
			idxList.Add(i);
		}

		idxList = QSort(l, idxList, asc);
		for (var i = 0; i < l.Count; i++)
		{
			idx[i] = idxList[i];
		}
		return idx;
	}

	private static int[] QSortString(List<string> l, bool asc)
	{
		var idx = new int[l.Count];
		var idxList = new List<int>();
		for (var i = 0; i < l.Count; i++)
		{
			idxList.Add(i);
		}

		idxList = QSortString(l, idxList, asc);
		for (var i = 0; i < l.Count; i++)
		{
			idx[i] = idxList[i];
		}
		return idx;
	}

	private static int[] QSortLong(List<long> l, bool asc)
	{
		var idx = new int[l.Count];
		var idxList = new List<int>();
		for (var i = 0; i < l.Count; i++)
		{
			idxList.Add(i);
		}

		idxList = QSortLong(l, idxList, asc);
		for (var i = 0; i < l.Count; i++)
		{
			idx[i] = idxList[i];
		}
		return idx;
	}

	private static int[] QSortDouble(List<double> l, bool asc)
	{
		var idx = new int[l.Count];
		var idxList = new List<int>();
		for (var i = 0; i < l.Count; i++)
		{
			idxList.Add(i);
		}

		idxList = QSortDouble(l, idxList, asc);
		for (var i = 0; i < l.Count; i++)
		{
			idx[i] = idxList[i];
		}
		return idx;
	}

	private static int[] QSort(int[] l, bool asc)
	{
		var idx = new int[l.Length];
		var idxList = new List<int>();
		for (var i = 0; i < l.Length; i++)
		{
			idxList.Add(i);
		}

		idxList = QSort(l, idxList, asc);
		for (var i = 0; i < l.Length; i++)
		{
			idx[i] = idxList[i];
		}
		return idx;
	}

	private static List<int> QSort(List<int> l, List<int> idxList, bool asc)
	{
		var mid = idxList.Count / 2;
		var left = new List<int>();
		var right = new List<int>();
		var pivot = new List<int>();

		for (var i = 0; i < idxList.Count; i++)
		{
			if (l[idxList[i]] > l[idxList[mid]])
			{
				if (asc)
					right.Add(idxList[i]);
				else
					left.Add(idxList[i]);
			}
			else if (l[idxList[i]] < l[idxList[mid]])
			{
				if (asc)
					left.Add(idxList[i]);
				else
					right.Add(idxList[i]);
			}
			else
			{
				pivot.Add(idxList[i]);
			}
		}

		if (left.Count > 1)
		{
			left = QSort(l, left, asc);
		}
		if (right.Count > 1)
		{
			right = QSort(l, right, asc);
		}

		var newIdx = new List<int>();
		newIdx.AddRange(left);
		newIdx.AddRange(pivot);
		newIdx.AddRange(right);

		return newIdx;
	}

	private static List<int> QSortString(List<string> l, List<int> idxList, bool asc)
	{
		var mid = idxList.Count / 2;
		var left = new List<int>();
		var right = new List<int>();
		var pivot = new List<int>();

		for (var i = 0; i < idxList.Count; i++)
		{
			if (string.Compare(l[idxList[i]], l[idxList[mid]], StringComparison.Ordinal) > 0)
			{
				if (asc)
					right.Add(idxList[i]);
				else
					left.Add(idxList[i]);
			}
			else if (string.Compare(l[idxList[i]], l[idxList[mid]], StringComparison.Ordinal) < 0)
			{
				if (asc)
					left.Add(idxList[i]);
				else
					right.Add(idxList[i]);
			}
			else
			{
				pivot.Add(idxList[i]);
			}
		}

		if (left.Count > 1)
		{
			left = QSortString(l, left, asc);
		}
		if (right.Count > 1)
		{
			right = QSortString(l, right, asc);
		}

		var newIdx = new List<int>();
		newIdx.AddRange(left);
		newIdx.AddRange(pivot);
		newIdx.AddRange(right);

		return newIdx;
	}

	private static List<int> QSort(int[] l, List<int> idxList, bool asc)
	{
		var mid = idxList.Count / 2;
		var left = new List<int>();
		var right = new List<int>();
		var pivot = new List<int>();

		for (var i = 0; i < idxList.Count; i++)
		{
			if (l[idxList[i]] > l[idxList[mid]])
			{
				if (asc)
					right.Add(idxList[i]);
				else
					left.Add(idxList[i]);
			}
			else if (l[idxList[i]] < l[idxList[mid]])
			{
				if (asc)
					left.Add(idxList[i]);
				else
					right.Add(idxList[i]);
			}
			else
			{
				pivot.Add(idxList[i]);
			}
		}

		if (left.Count > 1)
		{
			left = QSort(l, left, asc);
		}
		if (right.Count > 1)
		{
			right = QSort(l, right, asc);
		}

		var newIdx = new List<int>();
		newIdx.AddRange(left);
		newIdx.AddRange(pivot);
		newIdx.AddRange(right);

		return newIdx;
	}

	private static List<int> QSortDouble(List<double> l, List<int> idxList, bool asc)
	{
		var mid = idxList.Count / 2;
		var left = new List<int>();
		var right = new List<int>();
		var pivot = new List<int>();

		for (var i = 0; i < idxList.Count; i++)
		{
			if (l[idxList[i]] > l[idxList[mid]])
			{
				if (asc)
					right.Add(idxList[i]);
				else
					left.Add(idxList[i]);
			}
			else if (l[idxList[i]] < l[idxList[mid]])
			{
				if (asc)
					left.Add(idxList[i]);
				else
					right.Add(idxList[i]);
			}
			else
			{
				pivot.Add(idxList[i]);
			}
		}

		if (left.Count > 1)
		{
			left = QSortDouble(l, left, asc);
		}
		if (right.Count > 1)
		{
			right = QSortDouble(l, right, asc);
		}

		var newIdx = new List<int>();
		newIdx.AddRange(left);
		newIdx.AddRange(pivot);
		newIdx.AddRange(right);

		return newIdx;
	}

	private static List<int> QSortLong(List<long> l, List<int> idxList, bool asc)
	{
		var mid = idxList.Count / 2;
		var left = new List<int>();
		var right = new List<int>();
		var pivot = new List<int>();

		for (var i = 0; i < idxList.Count; i++)
		{
			if (l[idxList[i]] > l[idxList[mid]])
			{
				if (asc)
					right.Add(idxList[i]);
				else
					left.Add(idxList[i]);
			}
			else if (l[idxList[i]] < l[idxList[mid]])
			{
				if (asc)
					left.Add(idxList[i]);
				else
					right.Add(idxList[i]);
			}
			else
			{
				pivot.Add(idxList[i]);
			}
		}

		if (left.Count > 1)
		{
			left = QSortLong(l, left, asc);
		}
		if (right.Count > 1)
		{
			right = QSortLong(l, right, asc);
		}

		var newIdx = new List<int>();
		newIdx.AddRange(left);
		newIdx.AddRange(pivot);
		newIdx.AddRange(right);

		return newIdx;
	}
}
