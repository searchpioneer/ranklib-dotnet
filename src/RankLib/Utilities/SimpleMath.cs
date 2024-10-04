namespace RankLib.Utilities;

public static class SimpleMath
{
    private static readonly double Log2 = Math.Log(2);
    private static readonly double Log10 = Math.Log(10);
    private static readonly double Loge = Math.Log(Math.E);
    
    public static double LogBase2(double value)
    {
        return Math.Log(value) / Log2;
    }

    public static double LogBase10(double value)
    {
        return Math.Log(value) / Log10;
    }

    public static double Ln(double value)
    {
        return Math.Log(value) / Loge;
    }

    public static int Min(int a, int b)
    {
        return a > b ? b : a;
    }

    public static double P(long count, long total)
    {
        return (count + 0.5) / (total + 1);
    }

    public static double Round(double val)
    {
        int precision = 10000; // Keep 4 digits
        return Math.Floor(val * precision + 0.5) / precision;
    }

    public static double Round(float val)
    {
        int precision = 10000; // Keep 4 digits
        return Math.Floor(val * precision + 0.5) / precision;
    }

    public static double Round(double val, int n)
    {
        int precision = 1;
        for (int i = 0; i < n; i++)
        {
            precision *= 10;
        }
        return Math.Floor(val * precision + 0.5) / precision;
    }

    public static float Round(float val, int n)
    {
        int precision = 1;
        for (int i = 0; i < n; i++)
        {
            precision *= 10;
        }
        return (float)(Math.Floor(val * precision + 0.5) / precision);
    }
}