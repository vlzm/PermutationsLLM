import sys
import json

def solve(vector):
    """
    Sorts the vector using only L, R, and X moves.
    
    Strategy:
    - We'll place elements in their correct positions from left to right
    - To place element k at position k:
      1. Find where element k currently is
      2. Use rotations to bring it to position 0 or 1
      3. Use X if needed to get it to position 0
      4. Rotate right k times to move it to position k
      5. Then we need to "lock" the first k elements and continue
    
    The key insight is that we can use a modified approach:
    - For each position from 0 to n-1, we bring the correct element there
    - We use L/R to position elements and X to swap when needed
    """
    n = len(vector)
    if n <= 1:
        return ([], list(vector))
    
    # Work with a copy
    arr = list(vector)
    moves = []
    
    def do_L():
        """Left cyclic shift"""
        first = arr[0]
        for i in range(n - 1):
            arr[i] = arr[i + 1]
        arr[n - 1] = first
        moves.append('L')
    
    def do_R():
        """Right cyclic shift"""
        last = arr[n - 1]
        for i in range(n - 1, 0, -1):
            arr[i] = arr[i - 1]
        arr[0] = last
        moves.append('R')
    
    def do_X():
        """Swap first two elements"""
        arr[0], arr[1] = arr[1], arr[0]
        moves.append('X')
    
    def find_element(val):
        """Find position of element with value val"""
        for i in range(n):
            if arr[i] == val:
                return i
        return -1
    
    def is_sorted():
        """Check if array is sorted"""
        for i in range(n):
            if arr[i] != i:
                return False
        return True
    
    # Main sorting algorithm
    # We'll use a strategy similar to selection sort
    # For each target value from 0 to n-1, bring it to its correct position
    
    # The trick is: we can only swap positions 0 and 1, and rotate
    # To sort, we repeatedly:
    # 1. Rotate to bring the minimum unsorted element to position 0 or 1
    # 2. Use X if needed
    # 3. Rotate to place it correctly
    
    # Alternative approach: bubble-sort style
    # We can do a single pass bringing smaller elements towards the front
    
    # Let's use a different approach:
    # We can effectively do insertion sort by:
    # - Using rotations to move through the array
    # - Using X to swap adjacent elements at positions 0,1
    
    # Bubble sort adaptation:
    # To compare and swap elements at positions i and i+1:
    # 1. Rotate left i times to bring positions i and i+1 to positions 0 and 1
    # 2. If arr[0] > arr[1], do X
    # 3. Rotate right i times to restore positions
    
    for _ in range(n * n):  # At most n^2 passes for bubble sort
        if is_sorted():
            break
        
        swapped = False
        for i in range(n - 1):
            # Rotate left i times to bring positions i and i+1 to 0 and 1
            for _ in range(i):
                do_L()
            
            # Now arr[0] was originally at position i
            # and arr[1] was originally at position i+1
            
            # Compare and swap if needed
            if arr[0] > arr[1]:
                do_X()
                swapped = True
            
            # Rotate right i times to restore
            for _ in range(i):
                do_R()
        
        if not swapped:
            break
    
    return (moves, arr)


def main():
    if len(sys.argv) > 1:
        try:
            vector = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            vector = [2, 1, 0, 3]
    else:
        vector = [2, 1, 0, 3]
    
    moves, sorted_array = solve(vector)
    
    result = {
        "moves": moves,
        "sorted_array": sorted_array
    }
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()
