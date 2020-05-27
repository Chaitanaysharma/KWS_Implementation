#!/usr/bin/env python
# coding: utf-8

# In[8]:


def editDistDP(str1, str2, m, n): 
     
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    
    for i in range(m + 1): 
        for j in range(n + 1): 
  
             
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
             
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
             
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])# Replace
    
    
    
    i = m 
    j = n 
    lst = []
    while (i != 0 and j != 0):

        if (str1[i-1] == str2[j - 1]):
            i = i-1
            j = j-1



        elif (dp[i][j] == (dp[i - 1][j - 1] + 1)):  
            temp = ['replace',i-1,j-1] 
            lst.append(temp)
            i = i-1 
            j = j-1 



        elif (dp[i][j] == (dp[i - 1][j] + 1)): 
            temp = ['delete',i-1,j]
            lst.append(temp)
            i=i-1



        elif (dp[i][j] == (dp[i][j - 1] + 1)):
            temp = ['insert',i,j-1]
            lst.append(temp)
            j=j-1
            
    if(m > n):
        while(i>0):
            temp = ['delete',i-1,j]
            lst.append(temp)
            i=i-1
            
    if(n > m):
        while(j>0):
            temp = ['insert',i,j-1]
            lst.append(temp)
            j=j-1
        
        
  
    return dp,lst


# In[11]:


s = [2,4,7,8]
d = [6,14,33,22]

dp,lst = editDistDP(s,d,len(s),len(d))

print lst


# In[ ]:




