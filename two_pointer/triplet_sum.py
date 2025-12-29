def triplet_sum(nums, target):
    nums.sort()
    for i in range(len(nums)-2):
        left,right=i+1,len(nums)-1
        while left<right:
            if nums[i]+nums[left]+nums[right]==target:
                return [nums[i],nums[left],nums[right]]
            elif nums[i]+nums[left]+nums[right]<target:
                left=left+1
            else:
                right=right-1
    return []

print(triplet_sum([-1,0,1,2,-1,-4],0))
