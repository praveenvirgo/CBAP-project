import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/praveen/learnPython/HH-master/Python/MachineLearning/Data/loan_data.csv')
#df.info()
print(df.describe())
