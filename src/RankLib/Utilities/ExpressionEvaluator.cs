using System;
using System.Collections.Generic;

public class ExpressionEvaluator
{
    private static readonly string[] operators = { "+", "-", "*", "/", "^" };
    private static readonly string[] functions = { "log", "ln", "log2", "exp", "sqrt", "neg" };
    private static Dictionary<string, int>? priority;

    public ExpressionEvaluator()
    {
        if (priority == null)
        {
            priority = new Dictionary<string, int>
            {
                { "+", 2 },
                { "-", 2 },
                { "*", 3 },
                { "/", 3 },
                { "^", 4 },
                { "neg", 5 },
                { "log", 6 },
                { "ln", 6 },
                { "sqrt", 6 }
            };
        }
    }

    public static void Main(string[] args)
    {
        var ev = new ExpressionEvaluator();
        var exp = "sqrt(16)/exp(4^2)";
        Console.WriteLine(ev.GetRPN(exp));
        Console.WriteLine(ev.Eval(exp));
    }

    class Queue
    {
        private readonly List<string> _list = new();

        public void Enqueue(string s)
        {
            _list.Add(s);
        }

        public string Dequeue()
        {
            if (_list.Count == 0)
                return "";
            var s = _list[0];
            _list.RemoveAt(0);
            return s;
        }

        public int Size => _list.Count;

        public override string ToString()
        {
            return string.Join(" ", _list);
        }
    }

    class Stack
    {
        private readonly List<string> _list = new();

        public void Push(string s)
        {
            _list.Add(s);
        }

        public string Pop()
        {
            if (_list.Count == 0)
                return "";
            var s = _list[^1];
            _list.RemoveAt(_list.Count - 1);
            return s;
        }

        public int Size => _list.Count;

        public override string ToString()
        {
            return string.Join(" ", _list.ToArray());
        }
    }

    private bool IsOperator(string token)
    {
        foreach (var op in operators)
        {
            if (token == op)
                return true;
        }
        return false;
    }

    private bool IsFunction(string token)
    {
        foreach (var func in functions)
        {
            if (token == func)
                return true;
        }
        return false;
    }

    private Queue ToPostFix(string expression)
    {
        expression = expression.Replace(" ", "");
        var output = new Queue();
        var op = new Stack();
        string lastReadToken = "";
        for (int i = 0; i < expression.Length; i++)
        {
            var token = expression[i].ToString();
            if (token == "(")
            {
                op.Push(token);
            }
            else if (token == ")")
            {
                bool foundOpen = false;
                while (op.Size > 0 && !foundOpen)
                {
                    var last = op.Pop();
                    if (last != "(")
                        output.Enqueue(last);
                    else
                        foundOpen = true;
                }
                if (!foundOpen)
                    throw new Exception($"Error: Invalid expression: \"{expression}\". Parentheses mismatched.");
            }
            else if (IsOperator(token))
            {
                if (lastReadToken == "(" || IsOperator(lastReadToken))
                {
                    if (token == "-")
                        op.Push("neg");
                }
                else
                {
                    if (op.Size > 0)
                    {
                        var last = op.Pop();
                        if (last == "(")
                            op.Push(last);
                        else if (priority[token] > priority[last])
                            op.Push(last);
                        else if (priority[token] < priority[last])
                            output.Enqueue(last);
                        else
                        {
                            if (token == "^")
                                op.Push(last);
                            else
                                output.Enqueue(last);
                        }
                    }
                    op.Push(token);
                }
            }
            else
            {
                int j = i + 1;
                while (j < expression.Length && !IsOperator(expression[j].ToString()) && expression[j] != '(' && expression[j] != ')')
                {
                    token += expression[j];
                    j++;
                }
                i = j - 1;

                if (IsFunction(token))
                {
                    if (j == expression.Length || expression[j] != '(')
                        throw new Exception($"Error: Invalid expression: \"{expression}\". Function specification requires parentheses.");
                    op.Push(token);
                }
                else
                {
                    if (!double.TryParse(token, out _))
                        throw new Exception($"Error: \"{token}\" is not a valid token.");
                    output.Enqueue(token);
                }
            }
            lastReadToken = token;
        }

        while (op.Size > 0)
        {
            var last = op.Pop();
            if (last == "(")
                throw new Exception($"Error: Invalid expression: \"{expression}\". Parentheses mismatched.");
            output.Enqueue(last);
        }

        return output;
    }

    public string GetRPN(string expression)
    {
        return ToPostFix(expression).ToString();
    }

    public double Eval(string expression)
    {
        var output = ToPostFix(expression);
        var eval = new double[output.Size];
        int cp = 0;

        try
        {
            while (output.Size > 0)
            {
                var token = output.Dequeue();
                double v = 0;

                if (IsOperator(token))
                {
                    if (token == "+")
                        v = eval[cp - 2] + eval[cp - 1];
                    else if (token == "-")
                        v = eval[cp - 2] - eval[cp - 1];
                    else if (token == "*")
                        v = eval[cp - 2] * eval[cp - 1];
                    else if (token == "/")
                        v = eval[cp - 2] / eval[cp - 1];
                    else if (token == "^")
                        v = Math.Pow(eval[cp - 2], eval[cp - 1]);

                    eval[cp - 2] = v;
                    cp--;
                }
                else if (IsFunction(token))
                {
                    if (token == "log")
                    {
                        if (eval[cp - 1] <= 0)
                            throw new Exception($"Error: expression {expression} involves taking log of a non-positive number");
                        v = Math.Log10(eval[cp - 1]);
                    }
                    else if (token == "ln")
                    {
                        if (eval[cp - 1] <= 0)
                            throw new Exception($"Error: expression {expression} involves taking log of a non-positive number");
                        v = Math.Log(eval[cp - 1]);
                    }
                    else if (token == "log2")
                    {
                        if (eval[cp - 1] <= 0)
                            throw new Exception($"Error: expression {expression} involves taking log of a non-positive number");
                        v = Math.Log(eval[cp - 1]) / Math.Log(2);
                    }
                    else if (token == "exp")
                        v = Math.Exp(eval[cp - 1]);
                    else if (token == "sqrt")
                    {
                        if (eval[cp - 1] < 0)
                            throw new Exception($"Error: expression {expression} involves taking square root of a negative number");
                        v = Math.Sqrt(eval[cp - 1]);
                    }
                    else if (token == "neg")
                        v = -eval[cp - 1];

                    eval[cp - 1] = v;
                }
                else
                {
                    eval[cp++] = double.Parse(token);
                }
            }

            if (cp != 1)
                throw new Exception($"Error: invalid expression: {expression}");
        }
        catch (Exception ex)
        {
            throw new Exception($"Unknown error in ExpressionEvaluator::eval() with \"{expression}\"", ex);
        }

        return eval[cp - 1];
    }
}
