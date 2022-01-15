#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Bi-Weekly Challenge 2
# The Goal:
# To Collect and print user's information
# To Check and print user's age category
# To Calculate user's age 


# In[4]:


# Making sure users input the correct information 
def name_func(input):
# user name and age input
    while True:
        try:
            user_name = input()
            user_age = int(input())
        except NameError:
            print('haa this wan no be name oo')
        except ValueError:
            print('look am well again. this wan no be valid number, try put the correct number')
        break


# In[74]:


# concatinate the strings and input

print(" This is " + user_name + "," + "she is " + str(user_age))

#print("This is " + user_name + ',' + " she is " + str(user_age))


# In[49]:


# Calculate the user’s DOB using the age and print to screen

# creating the user_DOB function
year = 2021
user_DOB = year - user_age

# print the output
print(user_name + " was born in "  + str(user_DOB ))

year = 2021


# In[41]:


# Determine the user's age group

#Infants: <1
#Children: 1-11 years
#Teens: 12-17 
#Adults: 18-64
#Older Adults: 65+


# In[44]:


if (user_age <1): 
  group = "Infant"
elif (user_age >=1) & (user_age <=11): 
  group = "Child"
elif (user_age >=12) & (user_age <=17):
  group = "Teen"
elif (user_age >=18) & (user_age <=64):
  group = "Adult"
else:
  group = "Older Adult"
print(group)


# In[47]:


#the user’s age a decade ago?

Decade_ago = user_age - 10

Decade_ago


# In[13]:


for i in range(10, 60, 10):
    print(i)


# In[46]:


#For the next 50 years, print what the user’s age will be after every decade (NB: The current year is 2021 :))

new_age = [] # declaring a list for the new age

new_year = [] # declaring a list for the new year

for i in range(10, 60, 10): 
  new = user_age + i
  new_age.append(new)
  print(new_age)

  new_y = year + i 
  new_year.append(new_y)
  print(new_year)


# In[15]:


new_age


# In[16]:


new_year


# In[18]:


#In 2064 you’ll be 13y.o


# In[19]:


len(new_age)


# In[21]:


for i in range(len(new_age)):
  age = new_age[i]
  decade = new_year[i]

  print(" in " + str(decade) + " you'll be " + str(age) + "y.o")


# In[22]:


new_age[0]


# In[5]:


name_func


# In[ ]:




