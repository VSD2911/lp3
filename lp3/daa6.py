# deterministic and randomized
import random
def deterministic(arr):
    if len(arr)<=1:
        return arr
    pivot=arr[len(arr)//2]
    left=[x for x in arr if x<pivot]
    middle=[x for x in arr if x==pivot]
    right=[x for x in arr if x>pivot]
    
    return deterministic(left)+middle+deterministic(right)
def random_sort(arr):
    if len(arr)<=1:
        return arr
    pivot=random.choice(arr)
    left=[x for x in arr if x< pivot]
    middle=[x for x in arr if x== pivot]
    right=[x for x in arr if x> pivot]
    
    return random_sort(left)+middle+random_sort(right)

arr=[10,16,8,12,15,6,3,9,5]
arr1=deterministic(arr)
print("deterministic:",arr1)
arr2=random_sort(arr)
print(arr2)
def quicksort(arr,low,high):
    if(low<high):
        pivot=partition(arr,low,high)
        quicksort(arr,low,pivot-1)
        quicksort(arr,pivot+1,high)
    
def partition(arr,low,high):
#     n=len(arr)//2
    pivot=arr[low]
#     pivot=arr[len(arr)//2]
    i=low
    j=high
    
    while i<j:
        while (arr[i]<pivot):
            i+=1
        while(arr[j]>pivot):
            j-=1
        if(i<j):
            arr[i],arr[j]=arr[j],arr[i]
        else:
            return j
            
n=9
arr=[10,16,8,12,15,6,3,9,5]
quicksort(arr,0,n-1)
print(arr)
# Time Complexity Analysis:
# - Deterministic Quick Sort:
#   - Average-case time complexity: O(n*log(n)) - Pivot strategy divides the list into nearly equal parts.
#   - Worst-case time complexity: O(n^2) - Occurs when the pivot is always the minimum or maximum element.
#   - Best-case time complexity: O(n) - Occurs when the pivot divides the list into roughly equal halves.
# - Randomized Quick Sort:
#   - Average-case time complexity: O(n*log(n)) - Random pivot selection reduces the likelihood of worst-case scenarios.
#   - Worst-case time complexity: O(n^2) - Unlikely to occur in practice.
#   - Best-case time complexity: O(n) - Pivot divides the list into roughly equal halves.

# Space Complexity Analysis:
# - Both variants have a space complexity of O(log(n)) in the average case, corresponding to the function call stack.
# - In the worst case, space complexity can be O(n) for deterministic Quick Sort due to unbalanced partitions, but it remains O(log(n)) on average.

"""
Quick Sort is one of the most efficient and widely used sorting algorithms. It falls into the category of divide-and-conquer algorithms and is known for its fast average-case performance. The algorithm was developed by Tony Hoare in 1960.

Here's a step-by-step explanation of how Quick Sort works:

1. **Selection of a Pivot:** Quick Sort starts by selecting a pivot element from the array. The choice of the pivot can significantly affect the algorithm's performance. Common pivot selection strategies include selecting the first element, the last element, a random element, or the middle element.

2. **Partitioning the Array:** The next step is to rearrange the elements in the array such that elements smaller than the pivot are on the left, and elements greater than the pivot are on the right. The pivot itself is in its final sorted position. This process is called partitioning.

3. **Recursion:** Once the partitioning is done, the algorithm recursively applies the same steps to the subarrays on the left and right of the pivot (elements smaller and larger than the pivot). This recursive process continues until the subarrays have only one or zero elements, which are, by definition, sorted.

4. **Combining Results:** As the recursion unwinds, the sorted subarrays are combined to form the final sorted array.

Here's a simple example of how Quick Sort works:

Let's say we have an unsorted array: `[4, 7, 2, 1, 9, 8, 6]`.

1. We choose the pivot, which is often selected as the last element. In this case, the pivot is `6`.

2. We partition the array into two subarrays:
   - Elements less than the pivot (left subarray): `[4, 2, 1]`
   - Elements greater than the pivot (right subarray): `[7, 9, 8]`

3. We recursively apply Quick Sort to the subarrays.

   Left subarray (elements less than 6):
   - Choose the pivot (e.g., last element, `1`).
   - Partition into elements less and greater than the pivot (nothing to do here as there's only one element).
   
   Right subarray (elements greater than 6):
   - Choose the pivot (e.g., last element, `8`).
   - Partition into elements less and greater than the pivot (elements less: `[7]`, elements greater: `[9]`).

4. Combine the sorted subarrays to get the final sorted array: `[1, 2, 4, 6, 7, 8, 9]`.

Quick Sort has several advantages:

- It's an in-place sorting algorithm, which means it doesn't require additional memory for sorting.
- It's efficient for large datasets and has an average-case time complexity of O(n*log(n)).
- It sorts in-place, making it suitable for situations with limited memory.
- It can be easily parallelized for even faster sorting.

However, Quick Sort's worst-case time complexity is O(n^2), which occurs when the pivot choice consistently results in highly unbalanced partitions. To mitigate this, various pivot selection strategies are employed, and randomized Quick Sort is one way to address this issue.
"""
