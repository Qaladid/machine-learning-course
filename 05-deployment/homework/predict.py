#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import numpy as np
import requests


# In[6]:


# Define the URLs for the model files
url_prefix = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework/"
model_file_url = url_prefix + "model1.bin"
dv_file_url = url_prefix + "dv.bin"

# Function to download a file from a URL
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")


# In[7]:


# Download the model and DictVectorizer files
download_file(model_file_url, 'model1.bin')
download_file(dv_file_url, 'dv.bin')


# In[9]:


# Load the DictVectorizer and the Logistic Regression model
dv_file = 'dv.bin'
model_file = 'model1.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)


# In[10]:


# Client data
client = {
    "job": "management",
    "duration": 400,
    "poutcome": "success"
}

# Transform the client data
X = dv.transform([client])


# In[11]:


# Get the probability of getting a subscription
probability = model.predict_proba(X)[0, 1]
print(f"The probability that this client will get a subscription is: {probability:.3f}")


# In[ ]:




