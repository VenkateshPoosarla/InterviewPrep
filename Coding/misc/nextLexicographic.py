# Given a list of numbers, find the next lexicographic permutation of the list.
# steps 1: find the first element from the right that is smaller than the element to its right
# [1,3,2,4,5] -> 2
# step 2: find the smallest element from the right that is greater than the element found in step 1
# step 3: swap the elements found in step 1 and step 2
# step 4: sort the elements to the right of the element found in step 1
# step 5: return the list
from typing import List


def nextLexicographic(nums: List[int]) -> List[int]:
    """
    Return the next lexicographic permutation of `nums` (in-place mutation).

    Example:
    - [1, 3, 2, 4, 5] -> [1, 3, 2, 5, 4]

    Wrap-around behavior:
    - If `nums` is already the last permutation (e.g., [3, 2, 1]),
      return the first permutation by sorting ascending (e.g., [1, 2, 3]).
    """
    # Step 1: find the first "pivot" from the right where nums[i] < nums[i+1].
    # Everything to the right of i is non-increasing.
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    # If no pivot exists, we are at the last permutation -> wrap to the first.
    if i < 0:
        nums.sort()
        return nums

    # Step 2: from the right, find the smallest element > nums[i] (the successor).
    j = len(nums) - 1
    while nums[j] <= nums[i]:
        j -= 1

    # Step 3: swap pivot with successor.
    nums[i], nums[j] = nums[j], nums[i]

    # Step 4: make the suffix (i+1..end) as small as possible (ascending).
    nums[i + 1 :] = sorted(nums[i + 1 :])
    return nums

def nextLexicographic2(s: str) -> str:
    """
    Return the next lexicographic permutation of string `s`.

    Example:
    - "abecds" -> "abecsd"

    Wrap-around behavior:
    - "dcba" (last permutation) -> "abcd" (first permutation)

    Edge cases:
    - "" or single-character strings return as-is.
    """
    chars = list(s)  # convert to list so we can swap/mutate
    if len(chars) < 2:
        return s

    # Step 1: find pivot from the right where chars[i] < chars[i+1]
    i = len(chars) - 2
    while i >= 0 and chars[i] >= chars[i + 1]:
        i -= 1

    # No pivot => already the last permutation => wrap to sorted order
    if i < 0:
        return "".join(sorted(chars))

    # Step 2: find successor from the right where chars[j] > chars[i]
    j = len(chars) - 1
    while chars[j] <= chars[i]:
        j -= 1

    # Step 3: swap pivot and successor
    chars[i], chars[j] = chars[j], chars[i]

    # Step 4: sort suffix to get the smallest possible next permutation
    chars[i + 1 :] = sorted(chars[i + 1 :])
    return "".join(chars)

if __name__ == "__main__":
    print(nextLexicographic([1,3,2,4,5]))
    
    print(nextLexicographic2("abecds"))
    print(nextLexicographic2("dcba")) # EXPECTED abcd 
    print(nextLexicographic2(""))



