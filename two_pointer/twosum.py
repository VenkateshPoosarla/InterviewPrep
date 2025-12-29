def two_sum(nums, target):
    left,right=0,len(nums)-1
    while left<right:
        if nums[left]+nums[right]==target:
            return [left,right]
        elif nums[left]+nums[right]<target:
            left=left+1
        else:
            right=right-1
    return []

print(two_sum([2,7,11,15],9))
print(two_sum([2,3,4],6))
print(two_sum([-1,0],-1))