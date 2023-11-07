# DP
def knapsack_dp(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i-1])
            else:
                dp[i][w] = dp[i - 1][w]

    max_value = dp[n][capacity]
    items_in_knapsack = []
    w = capacity

    for i in range(n, 0, -1):
        if max_value <= 0:
            break
        if max_value != dp[i - 1][w]:
            items_in_knapsack.append(i - 1)
            max_value -= values[i - 1]
            w -= weights[i - 1]

    items_in_knapsack.reverse()
    return dp[n][capacity], items_in_knapsack

# Example usage
# values = [60, 100, 120]
# weights = [10, 20, 30]
# capacity = 50
values=[1,2,5,6]
weigths=[2,3,4,5]
capacity=8

max_value, selected_items = knapsack_dp(values, weights, capacity)

# print(f"Maximum value using dynamic programming: {max_value}")
# print("Items selected:")
# for item in selected_items:
#     print(f"Value: {values[item]}, Weight: {weights[item]}")
print("maximum value= ",max_value)
print("Items selected in knapsack: ")
for items in selected_items:
    print("value: ",values[items]," Weight: ",weights[items])

#BB

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.value_per_weight = value / weight

def knapsack_bb(values, weights, capacity):
    n = len(values)
    items = [Item(weights[i], values[i]) for i in range(n)]
    def bound(node, cur_weight, cur_value, remaining):
        if cur_weight >= capacity:
            return 0
        result = cur_value
        j = node
        while j < n and cur_weight + items[j].weight <= capacity:
            result += items[j].value
            cur_weight += items[j].weight
            j += 1
        if j < n:
            result += remaining * items[j].value_per_weight
        return result
    def knapsack_recursive(node, cur_weight, cur_value, remaining):
        if cur_weight <= capacity and cur_value > max_value[0]:
            max_value[0] = cur_value
        if node >= n or cur_weight >= capacity or bound(node, cur_weight, cur_value, remaining) <= max_value[0]:
            return
        knapsack_recursive(node + 1, cur_weight, cur_value, remaining - items[node].weight)
        knapsack_recursive(node + 1, cur_weight + items[node].weight, cur_value + items[node].value, remaining - items[node].weight)
    max_value = [0]
    knapsack_recursive(0, 0, 0, sum(values))
    return max_value[0]

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

max_value = knapsack_bb(values, weights, capacity)

print(f"Maximum value using branch and bound: {max_value}")


#DP
# Time Complexity Analysis:
# - Filling the dp table: O(n*capacity), where n is the number of items and capacity is the knapsack capacity.

# Space Complexity Analysis:
# - Space required for the dp table: O(n*capacity), where n is the number of items and capacity is the knapsack capacity.

"""
The 0/1 Knapsack Problem is a classic optimization problem in computer science and mathematics. It is a problem of combinatorial optimization where the goal is to select a combination of items, each with a given weight and value, to maximize the total value while not exceeding a given weight limit (the capacity of the knapsack). The term "0/1" indicates that for each item, you can either choose to include it (1) or exclude it (0) in the knapsack, meaning you can't take a fractional part of an item. In other words, it's a binary choice for each item.

Here's a formal description of the problem:

- You have a set of n items, each with a specific weight (w_i) and value (v_i).
- You have a knapsack with a maximum weight capacity (W).
- You can choose to include or exclude each item in the knapsack (0 or 1).
- The goal is to find the combination of items to include in the knapsack, such that the total weight of the selected items does not exceed W, and the total value is maximized.

Mathematically, the problem can be defined as follows:

Maximize Σ(v_i * x_i) for i = 1 to n
Subject to Σ(w_i * x_i) for i = 1 to n ≤ W
x_i = 0 or 1 for all i

Where:
- v_i is the value of item i.
- w_i is the weight of item i.
- x_i is a binary variable (0 or 1) indicating whether item i is included (1) or excluded (0) from the knapsack.
- W is the maximum weight capacity of the knapsack.

The 0/1 Knapsack Problem is known to be an NP-hard problem, meaning there is no known algorithm to solve it optimally in polynomial time for all inputs. However, dynamic programming and branch and bound algorithms can be used to find the optimal solution for relatively small instances of the problem. For larger instances, heuristics and approximation algorithms are often used to find near-optimal solutions.

This problem has various practical applications, such as resource allocation, portfolio optimization, and cutting stock problems.

-----------------------------

Dynamic programming is a problem-solving technique in computer science and mathematics used to efficiently solve problems by breaking them down into smaller overlapping subproblems and storing the results of these subproblems to avoid redundant computations. It is a powerful approach for solving optimization problems, particularly those with recursive or overlapping substructures. Here are the key principles and concepts of dynamic programming:

1. **Optimal Substructure:** Dynamic programming problems can be broken down into subproblems, and the optimal solution to the original problem can be constructed from optimal solutions to its subproblems. This is often expressed as a recursive formula or relation.

2. **Overlapping Subproblems:** In many dynamic programming problems, the same subproblems are encountered multiple times during the computation. Dynamic programming avoids redundant work by storing the results of already-solved subproblems in a data structure (usually a table or array) for later use.

3. **Memoization:** Memoization is the process of storing intermediate results (such as function values) in a data structure (often an array or dictionary) and returning the cached result when the same inputs reappear. This avoids redundant computation in recursive algorithms.

4. **Bottom-Up Approach:** Dynamic programming can be solved using either a top-down approach (recursion with memoization) or a bottom-up approach (iterative approach). In the bottom-up approach, you start by solving the smallest subproblems and build up to the original problem iteratively.

5. **State Transition:** Dynamic programming problems involve transitions from one state to another. These transitions can be represented by a recurrence relation or formula. The relation between subproblems and the order in which they are solved are crucial for finding an efficient solution.

6. **Tabulation:** In the bottom-up approach, you use a table (usually a multi-dimensional array) to store the results of subproblems. Each entry in the table corresponds to a specific state, and you fill it based on the optimal solution to smaller subproblems.

7. **Examples:** Dynamic programming is used in various problem domains, including:
   - **Fibonacci Sequence:** Calculating Fibonacci numbers using dynamic programming to avoid redundant calculations.
   - **0/1 Knapsack Problem:** Finding the most valuable combination of items to fit in a knapsack with limited capacity.
   - **Shortest Path Problems:** Finding the shortest path between nodes in a graph (e.g., Dijkstra's algorithm).
   - **Sequence Alignment:** Aligning sequences, such as DNA or protein sequences, to find the best match.
   - **Dynamic Time Warping:** Measuring the similarity between two time series sequences.
   - **Longest Common Subsequence (LCS):** Finding the longest subsequence common to two sequences.

8. **Time and Space Complexity:** The time and space complexity of dynamic programming solutions can vary depending on the specific problem and the chosen approach. While it often provides efficient solutions, dynamic programming can be computationally expensive for problems with large input sizes. Memoization and tabulation help manage time and space complexity.

Overall, dynamic programming is a versatile technique for solving a wide range of optimization problems, and its efficiency depends on how well the problem can be divided into smaller subproblems and how effectively overlapping subproblems are handled.
"""



# Time Complexity Analysis:
# The time complexity of the knapsack_branch_and_bound function can be summarized as O(n*log(n)) for the sorting step 
# in the best case, and potentially exponential (O(2^n)) in the worst case. The actual time taken to solve a specific instance
# of the problem will depend on the characteristics of the input data and the effectiveness of the pruning.

# Space Complexity Analysis:
# - The space complexity of this implementation is O(n) due to the recursion stack, where n is the number of items in the knapsack problem.

"""
The 0/1 Knapsack Problem is a classic optimization problem in computer science and mathematics. It is a problem of combinatorial optimization where the goal is to select a combination of items, each with a given weight and value, to maximize the total value while not exceeding a given weight limit (the capacity of the knapsack). The term "0/1" indicates that for each item, you can either choose to include it (1) or exclude it (0) in the knapsack, meaning you can't take a fractional part of an item. In other words, it's a binary choice for each item.

Here's a formal description of the problem:

- You have a set of n items, each with a specific weight (w_i) and value (v_i).
- You have a knapsack with a maximum weight capacity (W).
- You can choose to include or exclude each item in the knapsack (0 or 1).
- The goal is to find the combination of items to include in the knapsack, such that the total weight of the selected items does not exceed W, and the total value is maximized.

Mathematically, the problem can be defined as follows:

Maximize Σ(v_i * x_i) for i = 1 to n
Subject to Σ(w_i * x_i) for i = 1 to n ≤ W
x_i = 0 or 1 for all i

Where:
- v_i is the value of item i.
- w_i is the weight of item i.
- x_i is a binary variable (0 or 1) indicating whether item i is included (1) or excluded (0) from the knapsack.
- W is the maximum weight capacity of the knapsack.

The 0/1 Knapsack Problem is known to be an NP-hard problem, meaning there is no known algorithm to solve it optimally in polynomial time for all inputs. However, dynamic programming and branch and bound algorithms can be used to find the optimal solution for relatively small instances of the problem. For larger instances, heuristics and approximation algorithms are often used to find near-optimal solutions.

This problem has various practical applications, such as resource allocation, portfolio optimization, and cutting stock problems.

-----------------------------

Branch and Bound is a widely used algorithmic technique for solving optimization problems, particularly in combinatorial and discrete domains. It systematically explores the solution space by dividing it into smaller subproblems and bounding the search to find the best possible solution efficiently. It is often used when the problem cannot be solved using greedy or dynamic programming approaches. Here's an explanation of the Branch and Bound technique:

1. **Optimization Problems:** Branch and Bound is primarily used for optimization problems, where the goal is to find the best solution among a set of possible solutions. These problems often involve maximizing or minimizing an objective function under certain constraints.

2. **Systematic Search:** The key idea behind Branch and Bound is to explore the solution space in a systematic manner, similar to a tree search. It involves branching and bounding the search space to efficiently find the optimal solution.

3. **Main Components:**
   - **Branching:** At each node of the search tree, the algorithm makes branching decisions to divide the problem into smaller subproblems. These decisions often involve selecting one of several possible choices for a variable or element of the solution.
   - **Bounding:** The algorithm establishes bounds or estimates for each node in the search tree, allowing it to determine whether the subproblem at that node has the potential to lead to a better solution than the current best solution found. This helps in pruning unfruitful branches.
   - **Selection:** Branch and Bound maintains a priority queue or a list of nodes to be explored, and it selects nodes based on their bounds and estimates to ensure that more promising nodes are explored first.
   - **Completion:** The algorithm continues to branch and bound until it has explored the entire search tree or until the bounds of remaining unexplored nodes are worse than the best solution found so far.

4. **Termination and Pruning:** The algorithm terminates when it has explored the entire search space or when it determines that there cannot be a better solution than the current best solution. This pruning is achieved by comparing the bounds of nodes with the best solution found so far and skipping those nodes that are guaranteed to be suboptimal.

5. **Applications:** Branch and Bound can be applied to various optimization problems, including:
   - Traveling Salesman Problem (TSP)
   - 0/1 Knapsack Problem
   - Job Scheduling
   - Graph Coloring
   - Cutting Stock Problem
   - Combinatorial Auctions
   - Integer Linear Programming

6. **Complexity Analysis:** The efficiency of Branch and Bound depends on the problem's characteristics and the quality of the bounding function. In the worst case, it explores all nodes in the search tree, resulting in exponential time complexity. However, effective bounding strategies and heuristics can greatly improve its efficiency.

7. **Trade-offs:** Branch and Bound may not always guarantee the optimal solution, but it is designed to find good solutions efficiently. The quality of the solution found can depend on the quality of bounding estimates and the branching strategy used.

8. **Variations:** There are variations of Branch and Bound, such as Branch and Bound with Subproblems, which is used to solve larger problems by dividing them into smaller subproblems that can be solved optimally.

In summary, Branch and Bound is a powerful technique for solving optimization problems. It systematically explores the solution space by dividing it into smaller subproblems and bounding the search to find the best possible solution efficiently, making it a valuable tool in combinatorial and discrete optimization.
"""
