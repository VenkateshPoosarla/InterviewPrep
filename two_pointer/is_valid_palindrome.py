# Input: s = 'a dog! a panic in a pagoda.'

def is_valid_palindrome(s):
    left,right=0,len(s)-1
    while left<right:
        if s[left]!=s[right]:
            return False
        left+=1
        right-=1
    return True

print(is_valid_palindrome('a dog! a panic in a pagoda.'))