# DP
def knapsack_dp(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
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
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

max_value, selected_items = knapsack_dp(values, weights, capacity)

print(f"Maximum value using dynamic programming: {max_value}")
print("Items selected:")
for item in selected_items:
    print(f"Value: {values[item]}, Weight: {weights[item]}")


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