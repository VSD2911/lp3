class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.value_per_weight = value / weight

def fractional_knapsack(items, capacity):
    # Sort the items by value per unit weight in descending order
    items.sort(key=lambda x: x.value_per_weight, reverse=True)
    
    total_value = 0
    knapsack = []

    for item in items:
        if capacity >= item.weight:
            # Take the whole item
            knapsack.append(item)
            total_value += item.value
            capacity -= item.weight
        else:
            # Take a fraction of the item to fill the knapsack
            fraction = capacity / item.weight
            item_fraction = Item(item.weight * fraction, item.value * fraction)
            knapsack.append(item_fraction)
            total_value += item.value * fraction
            break

    return total_value, knapsack

# Example usage
if __name__ == "__main__":
    items = [
        Item(10, 60), 
        Item(20, 100), 
        Item(30, 120)
    ]
    capacity = 50

    total_value, knapsack = fractional_knapsack(items, capacity)

    print("Items in the knapsack:")
    for item in knapsack:
        print(f"Weight: {item.weight}, Value: {item.value}")
    
    print(f"Total value in the knapsack: {total_value}")