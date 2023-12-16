#!/usr/bin/env python
# coding: utf-8

# In[1]:


#AUTO INSURANCE CLAIM PREDICTION 


# In[2]:


# Importing the libraries 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore harmless warnings 

import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Import psql to run queries 

import pandasql as psql


# In[3]:


# Load the 'Auto Insurance Claim' data

Auto_Claim = pd.read_csv(r"C:\Users\sriha\Data Science\CAPSTONE PROJECT\synthetic data generated\auto claim insurance data set was synthetic generated\Auto_Insurance_claim_synthetic_genereated_data.csv", header=0)

# Copy the file to back-up

Auto_Claim_bk = Auto_Claim.copy()

# Display first 5 rows in the dataset

Auto_Claim_bk.head()


# In[4]:


# Display the size of the dataset

Auto_Claim_bk.shape


# In[5]:


# Display the columns in concrete dataset

Auto_Claim_bk.columns


# In[6]:


# DISPLAY INFO

Auto_Claim_bk.info()


# In[7]:


#STEP-1:CHECK THE DUPLICATE AND LOW VARIATION DATA


# In[8]:


#CHECK FOR DUPLICTAE VALUES
#Displaying Duplicate values with in DATASET, if avialble

Auto_Claim_bk_dup = Auto_Claim_bk[Auto_Claim_bk.duplicated(keep='last')]
Auto_Claim_bk_dup


# In[9]:


#IF EXIST DELETE THE DUPLICATE VALUES
#Remove the identified duplicate records 

Auto_Claim_bk = Auto_Claim_bk.drop_duplicates()
Auto_Claim_bk.shape


# In[10]:


#STEP-2:IDENTIFYING AND ADDRESSING THE MISSING VALUES AND VARIABLES
    


# In[11]:


# Identify the missing data in all variables

Auto_Claim_bk.isnull().sum()


# In[12]:


# functions for better visualization of the posterior plots

def resizeplot():
    plt.figure(figsize=(12,6))
    
# function for correlations plots

def resizecorr():
    plt.figure(figsize=(15,7))


# In[13]:


# Visualize missing data in graph

resizeplot()
sns.heatmap(Auto_Claim_bk.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[14]:


#MUTATION TECHNIQUES(FOR FILLING THE MISSING VALUES)


# In[15]:


# Identify the numerical and categorical variables

num_vars = Auto_Claim_bk.columns[Auto_Claim_bk.dtypes != 'object']
cat_vars = Auto_Claim_bk.columns[Auto_Claim_bk.dtypes == 'object']
print(num_vars)
print(cat_vars)


# In[16]:


# Use KNNImputer to address missing values
#KNNImputer-used only for replacing numerical values

#from sklearn.impute import KNNImputer

#imputer_int = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean',
#                        copy=True, add_indicator=False)

#Auto_Claim_bk['Age'] = imputer_str.fit_transform(Auto_Claim_bk[['Age']])


# In[17]:


# Use SimpleImputer to address missing values

from sklearn.impute import SimpleImputer

imputer_str = SimpleImputer(missing_values=np.nan, strategy='most_frequent', fill_value=None, verbose=0,
                            copy=True, add_indicator=False)

Auto_Claim_bk['Claim_Number'] = imputer_str.fit_transform(Auto_Claim_bk[['Claim_Number']])
Auto_Claim_bk['Accident_Date'] = imputer_str.fit_transform(Auto_Claim_bk[['Accident_Date']])
Auto_Claim_bk['Road_Type'] = imputer_str.fit_transform(Auto_Claim_bk[['Road_Type']])
Auto_Claim_bk['Accident_Severity'] = imputer_str.fit_transform(Auto_Claim_bk[['Accident_Severity']])


# In[18]:


Auto_Claim_bk.isnull().sum()


# In[19]:


# Visualize missing data in graph

resizeplot()
sns.heatmap(Auto_Claim_bk.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[20]:


Auto_Claim_bk.sample(10)


# In[21]:


#STEP-3:HANDLING THE OUTLIERS


# In[22]:


#OUTLIERS ARE PERFORMED ONLY FOR CONTINOUS DATA OR CONTINOUS VARIABLES


# In[23]:


# Display "Descriptive Statistical Analysis"

Auto_Claim_bk.describe().T


# In[24]:


# Identify the numerical and categorical variables

num_vars = Auto_Claim_bk.columns[Auto_Claim_bk.dtypes != 'object']
cat_vars = Auto_Claim_bk.columns[Auto_Claim_bk.dtypes == 'object']
print(num_vars)
print(cat_vars)


# In[25]:


#Age


# In[26]:


# Plot Histogram

plt.hist(Auto_Claim_bk.Age, bins=20, rwidth=0.8)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[27]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk.Age, bins=20, rwidth=0.8, density=True)
plt.xlabel('Age')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk.Age.min(),Auto_Claim_bk.Age.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk.Age.mean(),Auto_Claim_bk.Age.std()))


# In[28]:


# Eliminate the outlier in 'Age' and write data to new file


# In[29]:


# Calculate upper limit
Age_UL = round(Auto_Claim_bk.Age.mean() + 3 * Auto_Claim_bk.Age.std(),3)
Age_UL


# In[30]:


# Calculate Lower limit
Age_LL = round(Auto_Claim_bk.Age.mean() - 3 * Auto_Claim_bk.Age.std(),3)
Age_LL


# In[31]:


Auto_Claim_bk02 = Auto_Claim_bk[(Auto_Claim_bk.Age > Age_LL) & (Auto_Claim_bk.Age <Age_UL)]
Auto_Claim_bk02.shape


# In[32]:


# Display the outlier in the dataset

Auto_Claim_bk[(Auto_Claim_bk.Age > Age_UL) & (Auto_Claim_bk.Age > Age_LL)]


# In[33]:


#Policy_Premium


# In[34]:


# Plot Histogram

plt.hist(Auto_Claim_bk.Policy_Premium, bins=20, rwidth=0.8)
plt.xlabel('Policy_Premium')
plt.ylabel('Count')
plt.show()


# In[35]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk.Policy_Premium, bins=20, rwidth=0.8, density=True)
plt.xlabel('Policy_Premium')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk.Policy_Premium.min(),Auto_Claim_bk.Policy_Premium.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk.Policy_Premium.mean(),Auto_Claim_bk.Policy_Premium.std()))


# In[36]:


# Eliminate the outlier in 'Policy_Premium' and write data to new file


# In[37]:


# Calculate upper limit
Policy_Premium_UL = round(Auto_Claim_bk.Policy_Premium.mean() + 3 * Auto_Claim_bk.Policy_Premium.std(),3)
Policy_Premium_UL


# In[38]:


# Calculate Lower limit
Policy_Premium_LL = round(Auto_Claim_bk.Policy_Premium.mean() - 3 * Auto_Claim_bk.Policy_Premium.std(),3)
Policy_Premium_LL


# In[39]:


# Display the outlier in the dataset

Auto_Claim_bk[(Auto_Claim_bk.Policy_Premium > Policy_Premium_UL) & (Auto_Claim_bk.Policy_Premium > Age_LL)]


# In[40]:


Auto_Claim_bk.shape


# In[41]:


# Eliminate the outlier and write data to new file

Auto_Claim_bk03 = Auto_Claim_bk[(Auto_Claim_bk.Policy_Premium > Policy_Premium_LL) & (Auto_Claim_bk.Policy_Premium <Policy_Premium_UL)]
Auto_Claim_bk03.shape


# In[42]:


#Driving_Exp


# In[43]:


# Plot Histogram

plt.hist(Auto_Claim_bk03.Driving_Exp, bins=20, rwidth=0.8)
plt.xlabel('Driving_Exp')
plt.ylabel('Count')
plt.show()


# In[44]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk03.Driving_Exp, bins=20, rwidth=0.8, density=True)
plt.xlabel('Driving_Exp')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk03.Driving_Exp.min(),Auto_Claim_bk03.Driving_Exp.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk03.Driving_Exp.mean(),Auto_Claim_bk03.Driving_Exp.std()))


# In[45]:


# Eliminate the outlier in 'Driving_Exp' and write data to new file


# In[46]:


# Calculate upper limit
Driving_Exp_UL = round(Auto_Claim_bk03.Driving_Exp.mean() + 3 * Auto_Claim_bk03.Driving_Exp.std(),3)
Driving_Exp_UL


# In[47]:


# Calculate Lower limit
Driving_Exp_LL = round(Auto_Claim_bk03.Driving_Exp.mean() - 3 * Auto_Claim_bk03.Driving_Exp.std(),3)
Driving_Exp_LL


# In[48]:


# Display the outlier in the dataset

Auto_Claim_bk03[(Auto_Claim_bk03.Driving_Exp > Driving_Exp_UL) & (Auto_Claim_bk03.Driving_Exp > Driving_Exp_LL)]


# In[49]:


# Eliminate the outlier and write data to new file

Auto_Claim_bk04 = Auto_Claim_bk03[(Auto_Claim_bk03.Driving_Exp > Driving_Exp_LL) & (Auto_Claim_bk03.Driving_Exp < Driving_Exp_UL)]
Auto_Claim_bk04.shape


# In[50]:


#Annual_Miles


# In[51]:


# Plot Histogram

plt.hist(Auto_Claim_bk04.Age, bins=20, rwidth=0.8)
plt.xlabel('Annual_Miles')
plt.ylabel('Count')
plt.show()


# In[52]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk04.Annual_Miles, bins=20, rwidth=0.8, density=True)
plt.xlabel('Annual_Miles')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk04.Annual_Miles.min(),Auto_Claim_bk04.Annual_Miles.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk04.Annual_Miles.mean(),Auto_Claim_bk04.Annual_Miles.std()))


# In[53]:


# Eliminate the outlier in 'Annual_Miles' and write data to new file


# In[54]:


# Calculate upper limit
Annual_Miles_UL = round(Auto_Claim_bk04.Annual_Miles.mean() + 3 * Auto_Claim_bk04.Annual_Miles.std(),3)
Annual_Miles_UL


# In[55]:


# Calculate Lower limit
Annual_Miles_LL = round(Auto_Claim_bk04.Annual_Miles.mean() - 3 * Auto_Claim_bk04.Annual_Miles.std(),3)
Annual_Miles_LL


# In[56]:


# Display the outlier in the dataset

Auto_Claim_bk04[(Auto_Claim_bk04.Annual_Miles > Annual_Miles_UL) & (Auto_Claim_bk04.Annual_Miles > Annual_Miles_LL)]


# In[57]:


# Eliminate the outlier and write data to new file

Auto_Claim_bk05 = Auto_Claim_bk04[(Auto_Claim_bk04.Annual_Miles > Annual_Miles_LL) & (Auto_Claim_bk04.Annual_Miles < Annual_Miles_UL)]
Auto_Claim_bk05.shape


# In[58]:


#Previous_Citations


# In[59]:


# Plot Histogram

plt.hist(Auto_Claim_bk05.Previous_Citations, bins=20, rwidth=0.8)
plt.xlabel('Previous_Citations')
plt.ylabel('Count')
plt.show()


# In[60]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk05.Previous_Citations, bins=20, rwidth=0.8, density=True)
plt.xlabel('Previous_Citations')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk05.Previous_Citations.min(),Auto_Claim_bk05.Previous_Citations.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk05.Previous_Citations.mean(),Auto_Claim_bk05.Previous_Citations.std()))


# In[61]:


# Eliminate the outlier in 'Previous_Citations' and write data to new file


# In[62]:


# Calculate upper limit
Previous_Citations_UL = round(Auto_Claim_bk05.Previous_Citations.mean() + 3 * Auto_Claim_bk05.Previous_Citations.std(),3)
Previous_Citations_UL


# In[63]:


# Calculate Lower limit
Previous_Citations_LL = round(Auto_Claim_bk05.Previous_Citations.mean() - 3 * Auto_Claim_bk05.Previous_Citations.std(),3)
Previous_Citations_LL


# In[64]:


# Display the outlier in the dataset

Auto_Claim_bk05[(Auto_Claim_bk05.Previous_Citations > Previous_Citations_UL) & (Auto_Claim_bk05.Previous_Citations > Previous_Citations_LL)]


# In[65]:


# Eliminate the outlier and write data to new file

Auto_Claim_bk06 = Auto_Claim_bk05[(Auto_Claim_bk05.Previous_Citations > Previous_Citations_LL) & (Auto_Claim_bk05.Previous_Citations <Previous_Citations_UL)]
Auto_Claim_bk06.shape


# In[66]:


#Prevous_Accidents


# In[67]:


# Plot Histogram

plt.hist(Auto_Claim_bk06.Age, bins=20, rwidth=0.8)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[68]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk06.Age, bins=20, rwidth=0.8, density=True)
plt.xlabel('Age')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk06.Age.min(),Auto_Claim_bk06.Age.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk06.Age.mean(),Auto_Claim_bk06.Age.std()))


# In[69]:


# Eliminate the outlier in 'Age' and write data to new file


# In[70]:


# Calculate upper limit
Age_UL = round(Auto_Claim_bk06.Age.mean() + 3 * Auto_Claim_bk06.Age.std(),3)
Age_UL


# In[71]:


# Calculate Lower limit
Age_LL = round(Auto_Claim_bk06.Age.mean() - 3 * Auto_Claim_bk06.Age.std(),3)
Age_LL


# In[72]:


# Display the outlier in the dataset

Auto_Claim_bk06[(Auto_Claim_bk06.Age > Age_UL) & (Auto_Claim_bk06.Age > Age_LL)]


# In[73]:


# Eliminate the outlier and write data to new file

Auto_Claim_bk07 = Auto_Claim_bk06[(Auto_Claim_bk06.Age > Age_LL) & (Auto_Claim_bk06.Age <Age_UL)]
Auto_Claim_bk07.shape


# In[74]:


#Claim_Paid_Out


# In[75]:


# Plot Histogram

plt.hist(Auto_Claim_bk07.Claim_Paid_Out, bins=20, rwidth=0.8)
plt.xlabel('Claim_Paid_Out')
plt.ylabel('Count')
plt.show()


# In[76]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk07.Claim_Paid_Out, bins=20, rwidth=0.8, density=True)
plt.xlabel('Claim_Paid_Out')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk07.Claim_Paid_Out.min(),Auto_Claim_bk07.Claim_Paid_Out.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk07.Claim_Paid_Out.mean(),Auto_Claim_bk07.Claim_Paid_Out.std()))


# In[77]:


# Eliminate the outlier in 'Claim_Paid_Out' and write data to new file


# In[78]:


# Calculate upper limit
Claim_Paid_Out_UL = round(Auto_Claim_bk07.Claim_Paid_Out.mean() + 3 * Auto_Claim_bk07.Claim_Paid_Out.std(),3)
Claim_Paid_Out_UL


# In[79]:


# Calculate Lower limit
Claim_Paid_Out_LL = round(Auto_Claim_bk07.Claim_Paid_Out.mean() - 3 * Auto_Claim_bk07.Claim_Paid_Out.std(),3)
Claim_Paid_Out_LL


# In[80]:


# Display the outlier in the dataset

Auto_Claim_bk07[(Auto_Claim_bk07.Claim_Paid_Out > Claim_Paid_Out_UL) & (Auto_Claim_bk07.Claim_Paid_Out > Claim_Paid_Out_LL)]


# In[81]:


# Eliminate the outlier and write data to new file

Auto_Claim_bk08 = Auto_Claim_bk07[(Auto_Claim_bk07.Claim_Paid_Out > Claim_Paid_Out_LL) & (Auto_Claim_bk07.Claim_Paid_Out < Claim_Paid_Out_UL)]
Auto_Claim_bk08.shape


# In[82]:


#Vehicle_Cost


# In[83]:


# Plot Histogram

plt.hist(Auto_Claim_bk08.Vehicle_Cost, bins=20, rwidth=0.8)
plt.xlabel('Vehicle_Cost')
plt.ylabel('Count')
plt.show()


# In[84]:


# Gaussian distribution (also known as normal distribution) is a bell-shaped curve

from scipy.stats import norm

plt.hist(Auto_Claim_bk08.Vehicle_Cost, bins=20, rwidth=0.8, density=True)
plt.xlabel('Vehicle_Cost')
plt.ylabel('Count')

rng = np.arange(Auto_Claim_bk08.Vehicle_Cost.min(),Auto_Claim_bk08.Vehicle_Cost.max(), 0.1)
plt.plot(rng, norm.pdf(rng, Auto_Claim_bk08.Vehicle_Cost.mean(),Auto_Claim_bk08.Vehicle_Cost.std()))


# In[85]:


# Eliminate the outlier in 'Vehicle_Cost' and write data to new file


# In[86]:


# Calculate upper limit
Vehicle_Cost_UL = round(Auto_Claim_bk08.Vehicle_Cost.mean() + 3 * Auto_Claim_bk08.Vehicle_Cost.std(),3)
Vehicle_Cost_UL


# In[87]:


# Calculate Lower limit
Vehicle_Cost_LL = round(Auto_Claim_bk08.Vehicle_Cost.mean() - 3 * Auto_Claim_bk08.Vehicle_Cost.std(),3)
Vehicle_Cost_LL


# In[88]:


# Display the outlier in the dataset

Auto_Claim_bk08[(Auto_Claim_bk08.Vehicle_Cost >Vehicle_Cost_UL) & (Auto_Claim_bk08.Vehicle_Cost > Vehicle_Cost_LL)]


# In[89]:


# Eliminate the outlier and write data to new file

Auto_Claim_new = Auto_Claim_bk08[(Auto_Claim_bk08.Vehicle_Cost > Vehicle_Cost_LL) & (Auto_Claim_bk08.Vehicle_Cost <Vehicle_Cost_UL)]
Auto_Claim_new.shape


# In[90]:


#STEP-4:CATEGORICAL DATA AND ENCODING TECHNIQUES


# In[91]:


Auto_Claim_new.info()


# In[92]:


# Display unique values counts for each variable

Auto_Claim_new.nunique()


# In[93]:


# Display the data by variables wise

for i in Auto_Claim_new.columns:
    print(Auto_Claim_new[i].value_counts())


# In[94]:


Auto_Claim_new.head()


# In[95]:


# Identify the numerical and categorical variables

num_vars = Auto_Claim_new.columns[Auto_Claim_new.dtypes != 'object']
cat_vars = Auto_Claim_new.columns[Auto_Claim_new.dtypes == 'object']
print(num_vars)
print(cat_vars)


# In[96]:


# Display 'Inception_Date' categorical variable 

Auto_Claim_new['Inception_Date'].value_counts()


# In[97]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Inception_Date'] = LE.fit_transform(Auto_Claim_new[['Inception_Date']])


# In[98]:


Auto_Claim_new.head()


# In[99]:


# Display 'Inception_Date' categorical variable 

Auto_Claim_new['Inception_Date'].value_counts()


# In[100]:


#Policy_Number


# In[101]:


# Display 'Policy_Number' categorical variable 

Auto_Claim_new['Policy_Number'].value_counts()


# In[102]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Policy_Number'] = LE.fit_transform(Auto_Claim_new[['Policy_Number']])


# In[103]:


# Display 'Policy_Number' categorical variable 

Auto_Claim_new['Policy_Number'].value_counts()


# In[104]:


Auto_Claim_new.head()


# In[105]:


#Policy_Type


# In[106]:


# Display 'Policy_Type' categorical variable 

Auto_Claim_new['Policy_Type'].value_counts()


# In[107]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Policy_Type'] = LE.fit_transform(Auto_Claim_new[['Policy_Type']])


# In[108]:


# Display 'Policy_Type' categorical variable 

Auto_Claim_new['Policy_Type'].value_counts()


# In[109]:


Auto_Claim_new.head()


# In[110]:


#Channel


# In[111]:


# Display 'Channel' categorical variable 

Auto_Claim_new['Channel'].value_counts()


# In[112]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Channel'] = LE.fit_transform(Auto_Claim_new[['Channel']])


# In[113]:


# Display 'Policy_Number' categorical variable 

Auto_Claim_new['Channel'].value_counts()


# In[114]:


Auto_Claim_new.head()


# In[115]:


#State    


# In[116]:


# Display 'State' categorical variable 

Auto_Claim_new['State'].value_counts()


# In[117]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['State'] = LE.fit_transform(Auto_Claim_new[['State']])


# In[118]:


# Display 'State' categorical variable 

Auto_Claim_new['State'].value_counts()


# In[119]:


Auto_Claim_new.head()


# In[120]:


#Gender


# In[121]:


# Display 'Gender' categorical variable 

Auto_Claim_new['Gender'].value_counts()


# In[122]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Gender'] = LE.fit_transform(Auto_Claim_new[['Gender']])


# In[123]:


# Display 'Gender' categorical variable 

Auto_Claim_new['Gender'].value_counts()


# In[124]:


Auto_Claim_new.head()


# In[125]:


#Marital_Status


# In[126]:


# Display 'Marital_Status' categorical variable 

Auto_Claim_new['Marital_Status'].value_counts()


# In[127]:


# Replace variable 'Marital_Status', and convert the 'Marital_Status' to integer value.

Auto_Claim_new['Marital_Status'] = Auto_Claim_new['Marital_Status'].str.replace('Single', '1')
Auto_Claim_new['Marital_Status'] = Auto_Claim_new['Marital_Status'].str.replace('Married', '4')
Auto_Claim_new['Marital_Status'] = Auto_Claim_new['Marital_Status'].str.replace('Divorced', '2')
Auto_Claim_new['Marital_Status'] = Auto_Claim_new['Marital_Status'].str.replace('Widow', '3')
Auto_Claim_new['Marital_Status'] = Auto_Claim_new['Marital_Status'].astype(int)


# In[128]:


# Display 'Marital_Status' categorical variable 

Auto_Claim_new['Marital_Status'].value_counts()


# In[129]:


Auto_Claim_new.head()


# In[130]:


#Education


# In[131]:


# Display 'Education' categorical variable 

Auto_Claim_new['Education'].value_counts()


# In[132]:


# Replace the variable 'education', convert the 'education' to integer value.

Auto_Claim_new['Education'] = Auto_Claim_new['Education'].str.replace('High School', '1')
Auto_Claim_new['Education'] = Auto_Claim_new['Education'].str.replace('Bachelors', '2')
Auto_Claim_new['Education'] = Auto_Claim_new['Education'].str.replace('Masters', '3')
Auto_Claim_new['Education'] = Auto_Claim_new['Education'].str.replace('PhD', '4')
Auto_Claim_new['Education'] = Auto_Claim_new['Education'].astype(int)


# In[133]:


# Display 'Education' categorical variable 

Auto_Claim_new['Education'].value_counts()


# In[134]:


Auto_Claim_new.head()


# In[135]:


#Profession


# In[136]:


# Display 'Profession' categorical variable 

Auto_Claim_new['Profession'].value_counts()


# In[137]:


# Replace 'Profession' variable, and convert the 'Profession' to integer value.

Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Student', '0')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('driver', '1')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('painter', '1')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Artist', '1')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Worker', '2')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Carpenter', '2')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Teacher', '3')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Lecturer', '3')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Professor', '3')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Arts', '4')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Lawyer', '5')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Architect', '6')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Engineer', '7')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('ITENGINEER', '7')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('HR', '8')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('Manager', '8')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].str.replace('CEO', '9')
Auto_Claim_new['Profession'] = Auto_Claim_new['Profession'].astype(int)


# In[138]:


# Display 'Profession' categorical variable 

Auto_Claim_new['Profession'].value_counts()


# In[139]:


Auto_Claim_new.head()


# In[140]:


#Vehicle_Usage


# In[141]:


# Display 'Vehicle_Usage' categorical variable 

Auto_Claim_new['Vehicle_Usage'].value_counts()


# In[142]:


# Replace 'Vehicle_Usage' variable, and convert the 'Vehicle_Usage' to integer value.

Auto_Claim_new['Vehicle_Usage'] = Auto_Claim_new['Vehicle_Usage'].str.replace('Commute', '0')
Auto_Claim_new['Vehicle_Usage'] = Auto_Claim_new['Vehicle_Usage'].str.replace('Business', '1')
Auto_Claim_new['Vehicle_Usage'] = Auto_Claim_new['Vehicle_Usage'].str.replace('Pleasure', '2')
Auto_Claim_new['Vehicle_Usage'] = Auto_Claim_new['Vehicle_Usage'].astype(int)


# In[143]:


# Display 'Vehicle_Usage' categorical variable 

Auto_Claim_new['Vehicle_Usage'].value_counts()


# In[144]:


Auto_Claim_new.head()


# In[145]:


#Coverage_Type


# In[146]:


# Display 'Coverage_Type' categorical variable 

Auto_Claim_new['Coverage_Type'].value_counts()


# In[147]:


# Replace 'Coverage_Type' variable, and convert the 'Coverage_Type' to integer value.

Auto_Claim_new['Coverage_Type'] = Auto_Claim_new['Coverage_Type'].str.replace('Basic', '0')
Auto_Claim_new['Coverage_Type'] = Auto_Claim_new['Coverage_Type'].str.replace('Balanced', '1')
Auto_Claim_new['Coverage_Type'] = Auto_Claim_new['Coverage_Type'].str.replace('Enhanced', '2')
Auto_Claim_new['Coverage_Type'] = Auto_Claim_new['Coverage_Type'].astype(int)


# In[148]:


# Display 'Coverage_Type' categorical variable 

Auto_Claim_new['Coverage_Type'].value_counts()


# In[149]:


Auto_Claim_new.head()


# In[150]:


#Claim_Number


# In[151]:


# Display 'Claim_Number' categorical variable 

Auto_Claim_new['Claim_Number'].value_counts()


# In[152]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Claim_Number'] = LE.fit_transform(Auto_Claim_new[['Claim_Number']])


# In[153]:


# Display 'Claim_Number' categorical variable 

Auto_Claim_new['Claim_Number'].value_counts()


# In[154]:


Auto_Claim_new.head()


# In[155]:


#Umbrella_Policy


# In[156]:


# Display 'Umbrella_Policy' categorical variable 

Auto_Claim_new['Umbrella_Policy'].value_counts()


# In[157]:


# Replace 'Umbrella_Policy' variable, and convert the 'Umbrella_Policy' to integer value.

Auto_Claim_new['Umbrella_Policy'] = Auto_Claim_new['Umbrella_Policy'].str.replace('No', '0')
Auto_Claim_new['Umbrella_Policy'] = Auto_Claim_new['Umbrella_Policy'].str.replace('Yes', '1')
Auto_Claim_new['Umbrella_Policy'] = Auto_Claim_new['Umbrella_Policy'].astype(int)


# In[158]:


# Display 'Umbrella_Policy' categorical variable 

Auto_Claim_new['Umbrella_Policy'].value_counts()


# In[159]:


Auto_Claim_new.head()


# In[160]:


#Accident_Date


# In[161]:


# Display 'Accident_Date' categorical variable 

Auto_Claim_new['Accident_Date'].value_counts()


# In[162]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Accident_Date'] = LE.fit_transform(Auto_Claim_new[['Accident_Date']])


# In[163]:


# Display 'Accident_Date' categorical variable 

Auto_Claim_new['Accident_Date'].value_counts()


# In[164]:


Auto_Claim_new.head()


# In[165]:


#Police_File


# In[166]:


# Display 'Police_File' categorical variable 

Auto_Claim_new['Police_File'].value_counts()


# In[167]:


# Replace 'Police_File' variable, and convert the 'Police_File' to integer value.

Auto_Claim_new['Police_File'] = Auto_Claim_new['Police_File'].str.replace('No', '0')
Auto_Claim_new['Police_File'] = Auto_Claim_new['Police_File'].str.replace('Yes', '1')
Auto_Claim_new['Police_File'] = Auto_Claim_new['Police_File'].astype(int)


# In[168]:


# Display 'Police_File' categorical variable 

Auto_Claim_new['Police_File'].value_counts()


# In[169]:


Auto_Claim_new.head()


# In[170]:


#Any_Eye_Witness


# In[171]:


# Display 'Any_Eye_Witness' categorical variable 

Auto_Claim_new['Any_Eye_Witness'].value_counts()


# In[172]:


# Replace 'Any_Eye_Witness' variable, and convert the 'Any_Eye_Witness' to integer value.

Auto_Claim_new['Any_Eye_Witness'] = Auto_Claim_new['Any_Eye_Witness'].str.replace('No', '0')
Auto_Claim_new['Any_Eye_Witness'] = Auto_Claim_new['Any_Eye_Witness'].str.replace('Yes', '1')
Auto_Claim_new['Any_Eye_Witness'] = Auto_Claim_new['Any_Eye_Witness'].astype(int)


# In[173]:


# Display 'Any_Eye_Witness' categorical variable 

Auto_Claim_new['Any_Eye_Witness'].value_counts()


# In[174]:


Auto_Claim_new.head()


# In[175]:


#Hired_Attorney


# In[176]:


# Display 'Hired_Attorney' categorical variable 

Auto_Claim_new['Hired_Attorney'].value_counts()


# In[177]:


# Replace 'Hired_Attorney' variable, and convert the 'Hired_Attorney' to integer value.

Auto_Claim_new['Hired_Attorney'] = Auto_Claim_new['Hired_Attorney'].str.replace('No', '0')
Auto_Claim_new['Hired_Attorney'] = Auto_Claim_new['Hired_Attorney'].str.replace('Yes', '1')
Auto_Claim_new['Hired_Attorney'] = Auto_Claim_new['Hired_Attorney'].astype(int)


# In[178]:


# Display 'Hired_Attorney' categorical variable 

Auto_Claim_new['Hired_Attorney'].value_counts()


# In[179]:


Auto_Claim_new.head()


# In[180]:


#Make


# In[181]:


# Display 'Make' categorical variable 

Auto_Claim_new['Make'].value_counts()


# In[182]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Auto_Claim_new['Make'] = LE.fit_transform(Auto_Claim_new[['Make']])


# In[183]:


# Display 'Make' categorical variable 

Auto_Claim_new['Make'].value_counts()


# In[184]:


Auto_Claim_new.head()


# In[185]:


#Road_Type


# In[186]:


# Display 'Road_Type' categorical variable 

Auto_Claim_new['Road_Type'].value_counts()


# In[187]:


# Replace 'Road_Type' variable, and convert the 'Road_Type' to integer value.

Auto_Claim_new['Road_Type'] = Auto_Claim_new['Road_Type'].str.replace('Rural', '0')
Auto_Claim_new['Road_Type'] = Auto_Claim_new['Road_Type'].str.replace('Country side', '1')
Auto_Claim_new['Road_Type'] = Auto_Claim_new['Road_Type'].str.replace('Semi Urban', '2')
Auto_Claim_new['Road_Type'] = Auto_Claim_new['Road_Type'].str.replace('Urban', '3')
Auto_Claim_new['Road_Type'] = Auto_Claim_new['Road_Type'].astype(int)


# In[188]:


# Display 'Road_Type' categorical variable 

Auto_Claim_new['Road_Type'].value_counts()


# In[189]:


Auto_Claim_new.head()


# In[190]:


#Accident_Severity


# In[191]:


# Display 'Accident_Severity' categorical variable 

Auto_Claim_new['Accident_Severity'].value_counts()


# In[192]:


# Replace 'Accident_Severity' variable, and convert the 'Accident_Severity' to integer value.

Auto_Claim_new['Accident_Severity'] = Auto_Claim_new['Accident_Severity'].str.replace('Minor', '0')
Auto_Claim_new['Accident_Severity'] = Auto_Claim_new['Accident_Severity'].str.replace('Major', '1')
Auto_Claim_new['Accident_Severity'] = Auto_Claim_new['Accident_Severity'].str.replace('Fatal', '2')
Auto_Claim_new['Accident_Severity'] = Auto_Claim_new['Accident_Severity'].astype(int)


# In[193]:


# Display 'Accident_Severity' categorical variable 

Auto_Claim_new['Accident_Severity'].value_counts()


# In[194]:


Auto_Claim_new.head()


# In[195]:


#Claimed


# In[196]:


# Display 'Claimed' categorical variable 

Auto_Claim_new['Claimed'].value_counts()


# In[197]:


# Replace 'Claimed' variable, and convert the 'Claimed' to integer value.

Auto_Claim_new['Claimed'] = Auto_Claim_new['Claimed'].str.replace('No', '0')
Auto_Claim_new['Claimed'] = Auto_Claim_new['Claimed'].str.replace('Yes', '1')
Auto_Claim_new['Claimed'] = Auto_Claim_new['Claimed'].astype(int)


# In[198]:


# Display 'Claimed' categorical variable 

Auto_Claim_new['Claimed'].value_counts()


# In[199]:


Auto_Claim_new.head()


# In[200]:


Auto_Claim_new_count =Auto_Claim_new.Claimed.value_counts()
print('Class 0:', Auto_Claim_new_count[0])
print('Class 1:', Auto_Claim_new_count[1])
print('Proportion:', round(Auto_Claim_new_count[0] /Auto_Claim_new_count[1], 2), ': 1')
print('Total Responses:', len(Auto_Claim_new))


# In[201]:


#unbalanced data set we need to perform over sampling


# In[202]:


#STEP-5:SELECTION OF DEPENDENT AND INDEPENDENT VARIABLES


# In[203]:


# Identify the Independent and Target variables

IndepVar = []
for col in Auto_Claim_new.columns:
    if col != 'Claimed':
        IndepVar.append(col)

TargetVar = 'Claimed'

x = Auto_Claim_new[IndepVar]
y = Auto_Claim_new[TargetVar]


# In[204]:


# Random oversampling can be implemented using the RandomOverSampler class

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy=0.20)

x_over, y_over = oversample.fit_resample(x, y)

print(x_over.shape)
print(y_over.shape)


# In[205]:


# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size = 0.30, random_state = 42)

# Display the shape 

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[206]:


#STEP-6:FEATURE SCALING


# In[207]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# In[208]:


Auto_Claim_new.head()


# In[209]:


Auto_Claim_new.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\autoclaimnew.xlsx")


# In[ ]:





# In[210]:


# Load the Results dataset

CSResults = pd.read_csv(r"C:\Users\sriha\Data Science\CAPSTONE PROJECT\SR_20B91A04J11\CSResults.csv", header=0)

CSResults.head()


# In[211]:


#ALGORITHAMS


# In[212]:


#1ST-ALGORITHM


# In[213]:


#LOGISTIC REGRESSION:

# To build the decision tree model with random sampling

from sklearn.linear_model import LogisticRegression 

# Create an object for LR model

ModelLR = LogisticRegression()

# Train the model with training data

ModelLR = ModelLR.fit(x_train,y_train)

# Predict the model with test data set

y_pred = ModelLR.predict(x_test)

# confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, ModelLR.predict_proba(x_test)[:,1])
plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show() 
print('-----------------------------------------------------------------------------------------------------')
 #----------------------------------------------------------------------------------------------------------
new_row = {'Model Name' :ModelLR ,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': 'MCC',
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
CSResults = CSResults.append(new_row, ignore_index=True)


# In[214]:


CSResults


# In[215]:


# Predict the values with LogisticRegression algorithm

y_predF = ModelLR.predict(x_test)


# In[216]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[217]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[218]:


# Display the results

ResultsFinal.sample(5)


# In[219]:


Logistic_regression_analysis=ResultsFinal.copy()


# In[220]:


#Export Data Frame to Excel
Logistic_regression_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\Logistic_regression_analysis.xlsx")


# In[ ]:





# In[221]:


#2ND-ALGORITHM


# In[222]:


#DECISION TREE CLASSIFIER ALGORITHAM

# To build the 'Decision Tree' model with random sampling

from sklearn.tree import DecisionTreeClassifier 

# Create an object for model

ModelDT = DecisionTreeClassifier()
#ModelDT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
#                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
#                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
#                                 class_weight=None, ccp_alpha=0.0)

# Train the model with train data 

ModelDT.fit(x_train,y_train)

# Predict the model with test data set

y_pred = ModelDT.predict(x_test)
y_pred_prob = ModelDT.predict_proba(x_test)

# Confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, ModelDT.predict_proba(x_test)[:,1])
plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show() 
print('-----------------------------------------------------------------------------------------------------')
 #----------------------------------------------------------------------------------------------------------
new_row = {'Model Name' : ModelDT,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': 'MCC',
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
CSResults = CSResults.append(new_row, ignore_index=True)


# In[223]:


CSResults


# In[224]:


# Predict the values with DecisionTreeClassifier algorithm

y_predF = ModelDT.predict(x_test)


# In[225]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[226]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[227]:


# Display the results

ResultsFinal.sample(5)


# In[228]:


Decision_tree_classfier_analysis=ResultsFinal.copy()


# In[229]:


#Export Data Frame to Excel
Decision_tree_classfier_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\Decision_tree_classfier_analysis.xlsx")


# In[230]:


# To get feature importance

from matplotlib import pyplot

importance = ModelDT.feature_importances_

# Summarize feature importance

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# Plot feature importance

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[231]:


# Results (Run upto here only)

PredResults = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P':y_pred})

# Merge two Dataframes on index of both the dataframes

TestDataResults = Auto_Claim_new.merge(PredResults, left_index=True, right_index=True)

# Display the 10 records randomly

TestDataResults.sample(5)


# In[232]:


#3RD-ALGORITHM


# In[233]:


#RANDOM FOREST:

# To build the 'Random Forest' model with random sampling

from sklearn.ensemble import RandomForestClassifier

# Create model object

ModelRF = RandomForestClassifier()
#ModelRF = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
#                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
#                                 max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, 
#                                 n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, 
#                                 ccp_alpha=0.0, max_samples=None)

# Train the model with train data 

ModelRF.fit(x_train,y_train)

# Predict the model with test data set

y_pred = ModelRF.predict(x_test)
y_pred_prob = ModelRF.predict_proba(x_test)

# Confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, ModelRF.predict_proba(x_test)[:,1])
plt.figure()
#--------------------------------------------------------------------
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show() 
print('-----------------------------------------------------------------------------------------------------')
 #----------------------------------------------------------------------------------------------------------
new_row = {'Model Name' : ModelRF,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': 'MCC',
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
CSResults = CSResults.append(new_row, ignore_index=True)


# In[234]:


CSResults


# In[235]:


# Predict the values with RandomForestClassifier algorithm

y_predF = ModelRF.predict(x_test)


# In[236]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[237]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[238]:


# Display the results

ResultsFinal.sample(5)


# In[239]:


Random_forest_classfier_analysis=ResultsFinal.copy()


# In[240]:


#Export Data Frame to Excel
Random_forest_classfier_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\Random_forest_classfier_analysis.xlsx")


# In[241]:


# To get feature importance

from matplotlib import pyplot

importance = ModelRF.feature_importances_

# Summarize feature importance

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# Plot feature importance

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[242]:


# Results (Run upto here only)

PredResults = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P':y_pred})

# Merge two Dataframes on index of both the dataframes

TestDataResults = Auto_Claim_new.merge(PredResults, left_index=True, right_index=True)

# Display the 10 records randomly

TestDataResults.sample(5)


# In[243]:


#4TH-ALGORITHM


# In[244]:


#EXTRA TREE CLASSIFIER

# To build the 'Random Forest' model with random sampling
from sklearn.ensemble import ExtraTreesClassifier

# Create model object

ModelET = ExtraTreesClassifier()

# Evalution matrix for the algorithm

MM = [ModelET]
for models in MM:
            
    # Train the model training dataset
    
    models.fit(x_train, y_train)
    
    # Prediction the model with test dataset
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    #from math import sqrt

    #mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    Model_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    #
    plt.plot(fpr, tpr, label= 'Classification Model' % Model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': 'MCC',
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
    CSResults = CSResults.append(new_row, ignore_index=True)
    #----------------------------------------------------------------------------------------------------------


# In[245]:


CSResults


# In[246]:


# Predict the values with ExtraTreeClassifier algorithm

y_predF = ModelET.predict(x_test)


# In[247]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[248]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[249]:


# Display the results

ResultsFinal.sample(5)


# In[250]:


Extra_tree_classfier_analysis=ResultsFinal.copy()


# In[251]:


#Export Data Frame to Excel
Extra_tree_classfier_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\Extra_tree_classfier_analysis.xlsx")


# In[252]:


#5TH-ALGORITHM


# In[253]:


# Load the results dataset

KNNResults = pd.read_csv(r"C:\Users\sriha\Data Science\data by sir\day 4\session 1\KNN_Results.csv", header=0)



# In[254]:


# Initialize an array that stores the Accuracy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score

accuracy = []

for a in range(1, 11, 1):
    
    k = a
    
    # Build the model
    
    ModelKNN = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    
    ModelKNN.fit(x_train, y_train)
    
    # Predict the model
    
    y_pred = ModelKNN.predict(x_test)
    y_pred_prob = ModelKNN.predict_proba(x_test)
    
    print('KNN_K_value = ', a)
    
    # Print the model name
    
    print('Model Name: ', ModelKNN)
    
    # confusion matrix in sklearn
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    
    # actual values
    
    actual = y_test
    
    # predicted values
    
    predicted = y_pred
    
    # confusion matrix
    
    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)
    
    # outcome values order in sklearn
    
    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)
    
    # classification report for precision, recall f1-score and accuracy
    
    C_Report = classification_report(actual,predicted,labels=[1,0])
    
    print('Classification report : \n', C_Report)
    
    # calculating the metrics
    
    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);
    
    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model
    
    from math import sqrt
    
    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)
    
    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)
    
    # Area under ROC curve 
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual, ModelKNN.predict_proba(x_test)[:,1])
    plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(fpr, tpr, label= 'Classification Model' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    #--------------------------------------------------------tp, fn, fp, tn
    new_row = {'Model Name' : ModelKNN,
               'KNN K Value' : a,
               'True_Positive' : tp,
               'False_Negative' : fn,
               'False_Positive' : fp,
               'True_Negative' : tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': MCC,
               'ROC_AUC_Score': roc_auc_score(actual, predicted),
               'Balanced Accuracy': balanced_accuracy}
    KNNResults = KNNResults.append(new_row, ignore_index=True)
    #--------------------------------------------------------
    print('-----------------------------------------------------------------------------------------------------')


# In[255]:


KNNResults


# In[256]:


#After comparision above table we can say that at k=5 we get best result for KNN algorithm


# In[257]:


# Initialize an array that stores the Accuracy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score

accuracy = []

#for a in range(1, 11, 1):
    
k = a = 5
    
# Build the model
    
ModelKNN = KNeighborsClassifier(n_neighbors=k)

# Train the model
    
ModelKNN.fit(x_train, y_train)
    
# Predict the model
    
y_pred = ModelKNN.predict(x_test)
y_pred_prob = ModelKNN.predict_proba(x_test)
    
print('KNN_K_value = ', a)
    
# Print the model name
    
print('Model Name: ', ModelKNN)
    
# confusion matrix in sklearn
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
    
# actual values
    
actual = y_test
    
# predicted values
    
predicted = y_pred
    
# confusion matrix
    
matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)
    
# outcome values order in sklearn
    
tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
    
# classification report for precision, recall f1-score and accuracy
    
C_Report = classification_report(actual,predicted,labels=[1,0])
    
print('Classification report : \n', C_Report)
    
# calculating the metrics
    
sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);
    
# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model
    
from math import sqrt
    
mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)
    
print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)
    
# Area under ROC curve 
    
from sklearn.metrics import roc_curve, roc_auc_score
    
print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
# ROC Curve
    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, ModelKNN.predict_proba(x_test)[:,1])
plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr, tpr, label= 'Classification Model' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#--------------------------------------------------------tp, fn, fp, tn
new_row = {'Model Name' : ModelKNN,
               'KNN K Value' : a,
               'True_Positive' : tp,
               'False_Negative' : fn,
               'False_Positive' : fp,
               'True_Negative' : tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': MCC,
               'ROC_AUC_Score': roc_auc_score(actual, predicted),
               'Balanced Accuracy': balanced_accuracy}
CSResults = CSResults.append(new_row, ignore_index=True)
#--------------------------------------------------------
print('-----------------------------------------------------------------------------------------------------')


# In[258]:


CSResults


# In[259]:


# Predict the values with KNeighborsClassifier algorithm

y_predF = ModelKNN.predict(x_test)


# In[260]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[261]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[262]:


# Display the results

ResultsFinal.sample(5)


# In[263]:


KNeighbors_Classifier_analysis=ResultsFinal.copy()


# In[264]:


#Export Data Frame to Excel
KNeighbors_Classifier_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\KNeighbors_Classifier_analysis.xlsx")


# In[265]:


#6TH-ALGORITHM


# In[266]:


Auto_Claim_new.head()


# In[267]:


# Split the data and copy 10% data (stratified split on target variable) to new dataset 

Auto_Claim_T = Auto_Claim_new.groupby('Claimed', group_keys=False).apply(lambda x: x.sample(frac=0.1))

# Display the shape

Auto_Claim_T.shape


# In[268]:


# Identify the Independent and Target variables

IndepVar = []
for col in Auto_Claim_T.columns:
    if col != 'Claimed':
        IndepVar.append(col)

TargetVar = 'Claimed'

x = Auto_Claim_T[IndepVar]
y = Auto_Claim_T[TargetVar]


# In[269]:


# Random oversampling can be implemented using the RandomOverSampler class

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy=0.20)

x_over, y_over = oversample.fit_resample(x, y)

print(x_over.shape)
print(y_over.shape)


# In[270]:


# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size = 0.30, random_state = 42)

# Display the shape 

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[271]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# In[272]:


# Load the results dataset

EMResults = pd.read_csv(r"C:\Users\sriha\Data Science\CAPSTONE PROJECT\SR_20B91A04J11\EMResults.csv", header=0)



# In[273]:


# Build the all types of SVM Calssification models and compare the results

from sklearn.svm import SVC

# Create objects of classification algorithm with default hyper-parameters

SVMLIN = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
             probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
             max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)

SVMPLY = SVC(kernel='poly', degree=2, probability=True)

SVMGSN = SVC(kernel='rbf', random_state = 42, class_weight='balanced', probability=True)

SVMSIG = SVC(kernel='sigmoid', random_state = 42, class_weight='balanced', probability=True)

# Evalution matrix for all the algorithms

MM = [SVMLIN, SVMPLY, SVMGSN, SVMSIG]

for models in MM:
    
    # Fit the model
    
    models.fit(x_train, y_train)
    
    # Prediction
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt

    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%')
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    model_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    # plt.plot
    plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #---
    new_row = {'Model Name' : models,
               'True_Positive' : tp, 
               'False_Negative' : fn, 
               'False_Positive' : fp,
               'True_Negative' : tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC':MCC,
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
    EMResults = EMResults.append(new_row, ignore_index=True)
    #---


# In[274]:


EMResults


# In[275]:


#Out of all svm models we are getting the best results for SVM-Polynomial Kernel


# In[276]:


# Training the SVM algorithm

from sklearn.svm import SVC

CustPrb_TSVMPoly = SVC(kernel='poly', degree=2, probability=True)

# Train the model

CustPrb_TSVMPoly.fit(x_train, y_train)

# Predict the model with test data set

y_pred = CustPrb_TSVMPoly.predict(x_test)
y_pred_prob = CustPrb_TSVMPoly.predict_proba(x_test)

# Print the model name
    
print('Model Name: ', "SVM - Polynominal")

# Confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(y_test, y_pred), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test,CustPrb_TSVMPoly.predict_proba(x_test)[:,1])
plt.figure()
# plt.plot
plt.plot(fpr, tpr, label= 'Classification Model' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
print('-----------------------------------------------------------------------------------------------------')
#---
new_row = {'Model Name' : "SVM - Polynominal",
            'True_Positive' : tp, 
            'False_Negative' : fn, 
            'False_Positive' : fp,
            'True_Negative' : tn,
            'Accuracy' : accuracy,
            'Precision' : precision,
            'Recall' : sensitivity,
            'F1 Score' : f1Score,
            'Specificity' : specificity,
            'MCC':MCC,
            'ROC_AUC_Score':roc_auc_score(actual, predicted),
            'Balanced Accuracy':balanced_accuracy}
CSResults = CSResults.append(new_row, ignore_index=True)
#---


# In[277]:


CSResults


# In[278]:


# Predict the values with SVM - Polynominal algorithm

y_predF = CustPrb_TSVMPoly.predict(x_test)


# In[279]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[280]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[281]:


# Display the results

ResultsFinal.sample(5)


# In[282]:


SVM_Polynomial_Kernel_analysis=ResultsFinal.copy()


# In[283]:


#Export Data Frame to Excel
SVM_Polynomial_Kernel_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\SVM_Polynomial_Kernel_analysis.xlsx")


# In[284]:


#STEP-5:SELECTION OF DEPENDENT AND INDEPENDENT VARIABLES


# In[285]:


# Identify the Independent and Target variables

IndepVar = []
for col in Auto_Claim_new.columns:
    if col != 'Claimed':
        IndepVar.append(col)

TargetVar = 'Claimed'

x = Auto_Claim_new[IndepVar]
y = Auto_Claim_new[TargetVar]


# In[286]:


# Random oversampling can be implemented using the RandomOverSampler class

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy=0.20)

x_over, y_over = oversample.fit_resample(x, y)

print(x_over.shape)
print(y_over.shape)


# In[287]:


# Split the data into train and test (random sampling)

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[288]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# In[289]:


#7TH-ALGORITHM


# In[290]:


#Naive Bayes model (GaussianNB) Algorithm


# In[291]:


#NAIVE BAES MODEL

# Training the Naive Bayes model (GaussianNB) on the Training set

from sklearn.naive_bayes import GaussianNB

modelGNB = GaussianNB(priors=None, var_smoothing=1e-09)

# Fit the model with train data

modelGNB.fit(x_train,y_train)

# Predict the model with test data set

y_pred = modelGNB.predict(x_test)
y_pred_prob = modelGNB.predict_proba(x_test)

# Confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual,modelGNB.predict_proba(x_test)[:,1])
plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
print('-----------------------------------------------------------------------------------------------------')
 #----------------------------------------------------------------------------------------------------------
new_row = {'Model Name' : modelGNB,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': 'MCC',
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
CSResults = CSResults.append(new_row, ignore_index=True)


# In[292]:


CSResults


# In[293]:


# Predict the values with Naive Bayes model (GaussianNB) Algorithm

y_predF = modelGNB.predict(x_test)


# In[294]:


Results = pd.DataFrame({'Claimed_A':y_test, 'Claimed_P_F':y_predF})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = Auto_Claim_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


# In[295]:


# Calculate the %of Error

ResultsFinal['%Error'] = round(((ResultsFinal['Claimed_A']-ResultsFinal['Claimed_P_F'])/ResultsFinal['Claimed_A'])*100,3)


# In[296]:


# Display the results

ResultsFinal.sample(5)


# In[297]:


Naive_Bayes_classifier_analysis=ResultsFinal.copy()


# In[298]:


#Export Data Frame to Excel
Naive_Bayes_classifier_analysis.to_excel("C:\\Users\\sriha\\Data Science\\CAPSTONE PROJECT\\SR_20B91A04J11\\Excel files after analysis\\aut_ov_sampling\\Naive_Bayes_classifier_analysis.xlsx")


# In[299]:


#Calssification models and comparision of their results


# In[300]:


# Load the Results dataset

CSResults_1 = pd.read_csv(r"C:\Users\sriha\Data Science\CAPSTONE PROJECT\SR_20B91A04J11\CSResults_1.csv", header=0)

CSResults_1.head()


# In[301]:


# Build the Calssification models and compare the results

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

# Create objects of classification algorithm with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=3)
#ModelSVM = SVC(probability=True)

modelBAG = BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0,
                             bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                             n_jobs=None, random_state=None, verbose=0)

ModelGB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                     criterion='friedman_mse', min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                     init=None, random_state=None,
                                     max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False,
                                     validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
ModelLGB = lgb.LGBMClassifier()
ModelGNB = GaussianNB()

# Evalution matrix for all the algorithms
#MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelSVM, modelBAG, ModelGB, ModelLGB, ModelGNB]
MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, modelBAG, ModelGB, ModelLGB, ModelGNB]
#MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelSVM, ModelGNB]
for models in MM:
    
    # Fit the model
    
    models.fit(x_train, y_train)
    
    # Prediction
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt

    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(fpr, tpr, label= 'Classification Model' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'True_Positive' : tp, 
               'False_Negative' : fn, 
               'False_Positive' : fp,
               'True_Negative' : tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC':MCC,
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
    CSResults_1 = CSResults_1.append(new_row, ignore_index=True)
    #----------------------------------------------------------------------------------------------------------


# In[302]:


CSResults_1


# In[ ]:




