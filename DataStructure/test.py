# Definition for singly-linked list.
from Queue1 import SQueue
import time
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


#Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def __init__(self):
        self.output=[]
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        i = 1
        s = 0
        p1 = l1
        p2 = l2
        while p1 != None or p2 != None:
            if p1 == None:
                s += p2.val * i
                p2 = p2.next
            elif p2 == None:
                s += p1.val * i
                p1 = p1.next
            else:
                s += (p1.val + p2.val) * i
                p1 = p1.next
                p2 = p2.next
            i *= 10
        string = str(s)
        lst = [int(i) for i in string]
        l3 = ListNode(lst[-1])
        p = l3
        for j in range(len(lst)-2,-1,-1):
            p.next = ListNode(lst[j])
            p = p.next
        return l3

    def lengthOfLongestSubstring(self,s):
        if s == "":
            return 0
        i = 0
        j = 1
        maxi = 0
        maxlen = 1
        while j< len(s):
            r = s[i:j].find(s[j])
            if r == -1:
                j += 1
                if j - i > maxlen:
                    maxlen = j -i
                    maxi = i
            else:
                i += r + 1
                j += 1
        return maxi, s[maxi:maxi + maxlen]

    def findMedianSortedArrays(self,nums1, nums2):
        m, n = len(nums1), len(nums2)
        if m>n:
            nums1,nums2,m,n = nums2,nums1,n,m
        imin,imax=0,m
        while imin<=imax:
            i = (imin+imax)//2
            j = (m+n+1)//2-i
            if i>0 and nums1[i-1]>nums2[j]:
                imax=i-1
            elif i<m and nums2[j-1]>nums1[i]:
                imin=i+1
            else:
                if i==0:max_left=nums2[j-1]
                elif j ==0:max_left=nums1[i-1]
                else:max_left=max(nums1[i-1],nums2[j-1])

                if (m+n)%2==1:
                    return max_left

                if i==m:min_right=nums2[j]
                elif j==n:min_right=nums1[i]
                else:min_right=min(nums1[i],nums2[j])
                return (max_left+min_right)/2

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.output=[]
        return self.iT(root)
    def iT(self,root):
        if isinstance(root,TreeNode):
            if root.left!=None:
                self.iT(root.left)
            self.output.append(root.val)
            if root.right!=None:
                self.iT(root.right)
        return self.output


    def decodeString(self,s):
        st=list()
        for i in s:
            if i =="]":
                a=""
                while st[-1] !="[":
                    a=st.pop()+a
                st.pop()
                num=""
                while len(st)!=0 and st[-1].isdigit():
                    num=st.pop()+num
                st.append(int(num)*a)
            else:
                st.append(i)
        result=""
        while len(st)!=0:
            result=st.pop()+result
        return result

    def flooFill(self,image,sr,sc,newColor):
        self.nr=len(image)
        self.nc=len(image[0])
        self.marked=[[False for i in range(self.nc)] for i in range(self.nr)]
        self.marked[sr][sc]=True
        self.dfs(image,sr,sc,image[sr][sc],newColor)
        return image

    def dfs(self,image,sr,sc,oriColor,newColor):
        if sr>0 and self.marked[sr-1][sc]==False and image[sr-1][sc]==oriColor:
            self.marked[sr-1][sc]=True
            self.dfs(image,sr-1,sc,oriColor,newColor)

        if sr<self.nr-1 and self.marked[sr+1][sc]==False and image[sr+1][sc]==oriColor:
            self.marked[sr + 1][sc] = True
            self.dfs(image,sr+1,sc,oriColor,newColor)

        if sc>0 and self.marked[sr][sc-1]==False and image[sr][sc-1]==oriColor:
            self.marked[sr][sc-1] = True
            self.dfs(image,sr,sc-1,oriColor,newColor)

        if sc<self.nc-1 and self.marked[sr][sc+1] == False and image[sr][sc+1]==oriColor:
            self.marked[sr][sc+1] = True
            self.dfs(image, sr, sc+1,oriColor,newColor)

        image[sr][sc]=newColor

    def updateMatrix(self,matrix):
        nr=len(matrix)
        nc=len(matrix[0])
        dist=[[0 for i in range(nc)] for i in range(nr)]

        for i in range(nr):
            for j in range(nc):
                marked = [[False for i in range(nc)] for i in range(nr)]
                Q=[(i,j)]
                dis=0
                finded=False
                while Q:
                    new = []
                    for n in Q:
                        if matrix[n[0]][n[1]]==0:
                            dist[i][j]=dis
                            finded=True
                            break
                    if not finded:
                        for n in Q:
                            sr,sc=n
                            if sr>0 and marked[sr-1][sc]==False:
                                new.append((sr-1,sc))
                                marked[sr - 1][sc] =True
                            if sr<nr-1 and marked[sr+1][sc]==False:
                                new.append((sr+1,sc))
                                marked[sr + 1][sc] =True
                            if sc>0 and marked[sr][sc-1]==False:
                                new.append((sr,sc-1))
                                marked[sr][sc-1] =True
                            if sc<nc-1 and marked[sr][sc+1]==False:
                                new.append((sr,sc+1))
                                marked[sr][sc+1] =True
                    dis+=1
                    Q=new
        return dist

    def updateMatrix1(self,matrix):
        nr=len(matrix)
        nc=len(matrix[0])
        unmarked=[(i,j) for i in range(nr) for j in range(nc)]
        dis=0
        while len(unmarked) !=0:
            removed = []
            for n in unmarked:
                sr,sc=n
                if matrix[sr][sc]==dis:
                    removed.append(n)
                    jump =True
                else:
                    jump=False
                    if sr>0:
                        jump=jump or matrix[sr-1][sc]==dis
                    if sr<nr-1:
                        jump = jump or matrix[sr + 1][sc] == dis
                    if sc>0:
                        jump = jump or matrix[sr][sc-1] == dis
                    if sc<nc-1:
                        jump = jump or matrix[sr][sc+1] == dis
                if not jump:
                    matrix[sr][sc]+=1
            for i in removed:
                unmarked.remove(i)
            dis +=1
        return matrix

    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        i=len(digits)-1
        digits[i]+=1
        while digits[i]==10:
            digits[i]=0
            if i==0:
                digits.insert(0,1)
            else:
                digits[i-1]+=1
                i-=1
        return digits

    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        res=[]
        rows=len(matrix)
        cols=len(matrix[0])
        for s in range(rows+cols-1):
            if s % 2 == 1:
                for i in range(max(0,s - cols + 1), min(rows,s+1)):
                    res.append(matrix[i][s-i])
            else:
                for i in range(min(rows-1,s),max(0,s - cols + 1)-1,-1):
                    res.append(matrix[i][s-i])

        return res

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        i=len(a)-1
        j=len(b)-1
        if i<j:
            i,j,a,b=j,i,b,a
        s=""
        c=0
        while i>=0:
            a1=a[i]
            if j>=0:
                b1=b[j]
            else:
                b1="0"
            d=int(a1)+int(b1)+c
            c=d//2
            s=str(d%2)+s
            i-=1
            j-=1
        if c==1:
            return "1"+s
        else:
            return s

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle == "":
            return 0
        pnext = self.gen_pnext(needle)
        i, j = 0, 0
        n, m = len(haystack), len(needle)
        while j < n and i < m:
            if i == -1 or haystack[j] == needle[i]:
                i += 1
                j += 1
            else:
                i = pnext[i]
        if i == m:
            return j - i
        return -1

    def gen_pnext(self, p):
        i, k, m = 0, -1, len(p)
        pnext = [-1] * m
        while i < m - 1:
            if k == -1 or p[i] == p[k]:
                i += 1
                k += 1
                if p[i] == p[k]:
                    pnext[i] = pnext[k]
                else:
                    pnext[i] = k
            else:
                k = pnext[k]
        return pnext

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        n=len(strs)
        if "" in strs or n==0:
            return ""
        lens=[len(string) for string in strs]
        minlen=min(lens)
        for i in range(minlen):
            for j in range(n-1):
                if strs[j][i]!=strs[j+1][i]:
                    return strs[0][0:i]
        return strs[0][0:i+1]

    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i=0
        l=0
        ml=0
        while i <len(nums):
            if nums[i]==1:
                l+=1
            else:
                if l>ml:
                    ml=l
                l=0
            i+=1
        if l>ml:
            return l
        return ml

    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        length=len(nums)
        i,j,minl,Sum=0,0,length,0
        while i <length:
            if Sum>=s:
                if j-i<minl:
                    minl=j-i
                Sum-=nums[i]
                i+=1
            elif j<length:
                Sum+=nums[j]
                j+=1
            else:
                break
        if j-i<minl:
            return j-i
        return minl

    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        target=sum(nums)+S
        if target%2==1:
            return 0
        self.out=0
        self._dfs(nums,0,target//2)

    def _dfs(self,nums,a,target):
        if len(nums)==0:
            if a==target:
                self.out+=1
        else:
            if a>target:
                return
            self._dfs(nums[1:],a+nums[0],target)
            self._dfs(nums[1:],a,target)

    def subsetSum(self,nums,S):
        dp=[0]*(S+1)
        dp[0]=1
        for num in nums:
            for i in range(S,num-1,-1):
                dp[i]+=dp[i-num]
        return dp[S]

    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
            return 1 / self._pow(x, -n)
        else:
            return self._pow(x, n)

    def _pow(self, x, n):
        if n == 0:
            return 1.0
        a = self._pow(x, n // 2)
        if n % 2 == 1:
            return a *a * x
        else:
            return a *a

    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if inorder == []:
            return None
        preorder.reverse()
        self.preorder = preorder
        return self._buildTree(inorder, self.preorder.pop())

    def _buildTree(self, inorder, pre):
        # if inorder==[]:
        #     return None
        root = TreeNode(pre)
        i = inorder.index(pre)
        if inorder[:i] != []:
            root.left = self._buildTree(inorder[:i], self.preorder.pop())
        if inorder[i + 1:] != []:
            root.right = self._buildTree(inorder[i + 1:], self.preorder.pop())
        return root

    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        dic1,dic2={},{}
        for i in s1:
            dic1[i]=dic1.get(i,0)+1
        l=len(s1)
        substring=s2[0:l]
        for i in substring:
            dic2[i]=dic2.get(i,0)+1
        if dic1==dic2:
            return True
        for j in range(len(s2)-l):
            dic2[s2[j]]-=1
            if dic2[s2[j]]==0:
                del dic2[s2[j]]
            dic2[s2[j+l]]=dic2.get(s2[j+l],0)+1
            if dic1==dic2:
                return True
        return False

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        self.out = []
        self._dfs1(s, 4, "")
        return self.out

    def _dfs1(self, s, k, ip):
        if k == 1:
            if self.isValid(s):
                self.out.append(ip + s)

        else:
            for i in range(1, min(4, len(s) + 1)):
                if self.isValid(s[:i]):
                    self._dfs1(s[i:], k - 1, ip + s[:i] + ".")

    def isValid(self, s):
        if s == "0":
            return True
        if s != "" and s[0] != "0" and int(s) < 256:
            return True
        else:
            return False

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums_hash = {}
        result = list()
        for num in nums:
            nums_hash[num] = nums_hash.get(num, 0) + 1
        if 0 in nums_hash and nums_hash[0] >= 3:
            result.append([0, 0, 0])

        nums = sorted(list(nums_hash.keys()))

        for i, num in enumerate(nums):
            for j in nums[i + 1:]:
                if num * 2 + j == 0 and nums_hash[num] >= 2:
                    result.append([num, num, j])
                if j * 2 + num == 0 and nums_hash[j] >= 2:
                    result.append([j, j, num])

                dif = 0 - num - j
                if dif > j and dif in nums_hash:
                    result.append([num, j, dif])
        return result

    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dic={}
        for i in nums:
            dic.setdefault(i,False)
        maxl=1
        for i in nums:
            if not dic[i]:
                l=1
                dic[i]=True
                i1,i2=i+1,i-1
                while True:
                    if i1 in dic:
                        dic[i1]=True
                        i1+=1
                        l+=1
                    else:
                        break
                while True:
                    if i2 in dic:
                        dic[i2]=True
                        i2-=1
                        l+=1
                    else:
                        break
                if maxl<l:
                    maxl=l
        return maxl

    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        k=k-1
        out=""
        nums=list(range(1,n+1))
        for i in range(1,n):
            s=1
            for j in range(1,n-i+1):
                s*=j
            a=k//s
            out=out+str(nums.pop(a))
            k=k%s
        out+=str(nums[0])
        return out

    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        st=[]
        h=head
        l=0
        while h!=None:
            st.append(h)
            l+=1
            h=h.next
        if n==l:
            head=head.next
            return head
        for i in range(n):
            a=st.pop()
        if n==1:
            a=None
        else:
            a.val=a.next.val
            a.next=a.next.next
        return head

    def recursion(self,lists):
        j=0
        while j<len(lists):
            if lists[j]==None:
                lists.pop(j)
            else:
                j+=1
        if len(lists)==0:
            return None
        if len(lists)==1:
            return lists[0]
        minN=0
        for i in range(1,len(lists)):
            if lists[i].val<lists[minN].val:
                minN=i
        p=lists[minN]
        lists[minN]=lists[minN].next
        p.next=self.recursion(lists)
        return p

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        self.result = []
        self.recursion1(matrix)
        return self.result

    def recursion1(self,matrix):
        if len(matrix)==0 or len(matrix[0])==0:
            return
        if len(matrix)<=1:
            for i in matrix[0]:
                self.result.append(i)
        elif len(matrix[0])<=1:
            for i in range(len(matrix)):
                self.result.append(matrix[i][0])
        else:
            row=len(matrix)
            col=len(matrix[0])
            for i in range(col-1):
                self.result.append(matrix[0][i])
            for i in range(row-1):
                self.result.append(matrix[i][-1])
            for i in range(col-1,0,-1):
                self.result.append(matrix[-1][i])
            for i in range(row-1,0,-1):
                self.result.append(matrix[i][0])
            self.recursion1([ma[1:-1] for ma in matrix[1:-1]])

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for i in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for i in range(1, n):
            dp[0][i] = dp[0][i-1] + grid[0][i]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[m - 1][n - 1]

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        nums = list(range(1, n + 1))
        self.result = []
        self.recursion4([], nums, k)
        return self.result

    def recursion4(self, selected, candi, k):
        if k == 1:
            for i in candi:
                self.result.append(selected + [i])
        else:
            for i in range(len(candi) - k+1):
                self.recursion4(selected + [candi[i]], candi[i + 1:], k - 1)

def efficient(lst):
    n=len(lst)
    a=set(lst)
    for i in range(n+1):
        if not i in a:
            return i

if __name__ == "__main__":
    solution=Solution()
    # i, substr=solution.lengthOfLongestSubstring("pwwkew")
    # median=solution.findMedianSortedArrays([1,2],[3,4])
    # root=TreeNode(1)
    # root.left=TreeNode(2)
    # root.right=TreeNode(3)9
    # a=solution.inorderTraversal(1)
    # print(solution.decodeString("10[leetcode]"))
    # image=solution.flooFill([[1,1,1],[1,1,0],[1,0,1]],1,1,2)
    # dist = solution.updateMatrix1([[0,0,0],[0,1,0],[1,1,1]])
    # dist=solution.updateMatrix1()
    # print(solution.plusOne([9,8,9,9]))
    # print(solution.findDiagonalOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
    # print(solution.addBinary("101","101"))
    # print(solution.strStr("hello","ll"))
    # print(solution.longestCommonPrefix(["aa","a"]))
    # print(solution.minSubArrayLen(7,[2,3,1,2,4,3]))
    # solution.findTargetSumWays([42,24,30,14,38,27,12,29,43,42,5,18,0,1,12,44,45,50,21,47],38)
    # print(solution.subsetSum([1,1,1,1,1],4))
    # print(solution.myPow(2.000,-
    # print(solution.buildTree([3,9,20,15,7],[9,3,15,20,7]))
    # print(solution.checkInclusion("ab","eidbaooo"))
    # print(solution.restoreIpAddresses("010010"))
    # print(solution.threeSum([0,0,0,0,0,1,-1]))
    # print(solution.longestConsecutive([100, 4, 200, 1, 3, 2]))
    # print(solution.getPermutation(3,3))
    # r=solution.removeNthFromEnd(ListNode(1),1)
    # r=solution.recursion([ListNode(1),ListNode(2),ListNode(3)])
    # print(solution.spiralOrder([[1,11],[2,12],[3,13],[4,14],[5,15],[6,16],[7,17],[8,18],[9,19],[10,20]]))
    # print(solution.canJump([3,2,1,0,4]))
    # print(solution.minPathSum([[1,3,1],[1,5,1],[4,2,1]]))
    print(solution.combine(4,2))