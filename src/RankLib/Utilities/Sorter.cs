namespace RankLib.Utilities;

public static class Sorter
{
	/// <summary>
	/// Sort a double array using Interchange sort.
	/// </summary>
	/// <param name="sortVal">The double array to be sorted.</param>
	/// <param name="asc"><c>true</c> to sort ascending, <c>false</c> to sort descending.</param>
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

	private static int[] QSort(int[] l, bool asc)
	{
		var idx = new int[l.Length];
		var idxList = new List<int>();
		for (var i = 0; i < l.Length; i++)
			idxList.Add(i);

		idxList = QSort(l, idxList, asc);
		for (var i = 0; i < l.Length; i++)
			idx[i] = idxList[i];

		return idx;
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
				pivot.Add(idxList[i]);
		}

		if (left.Count > 1)
			left = QSort(l, left, asc);
		if (right.Count > 1)
			right = QSort(l, right, asc);

		var newIdx = new List<int>(left.Count + pivot.Count + right.Count);
		newIdx.AddRange(left);
		newIdx.AddRange(pivot);
		newIdx.AddRange(right);
		return newIdx;
	}
}
