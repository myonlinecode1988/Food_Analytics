import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_json('epicurious-recipes-with-rating-and-nutrition/full_format_recipes.json').dropna(thresh=2)  #Dropping some null rows
raw_data = raw_data[raw_data.rating.notnull()]  #Drop rows with null ratings

raw_data.rating.hist(width=.4);
plt.title('Distribution of Ratings');


latest_date = raw_data.date.max()
raw_data['t_delta'] = latest_date-raw_data.date
raw_data['years_old'] = raw_data.t_delta.apply(lambda x: int(x.days/365))
raw_data['age'] = raw_data.years_old/raw_data.years_old.max()

#Drop the 13 recipes with strange dates, like 20+ years old recipes.
raw_data = raw_data[raw_data.years_old <= 12]

raw_data['half_years'] = raw_data.t_delta.apply(lambda x: int(x.days/182))

# count-list-items-1.py

#wordstring = 'it was the best of times it was the worst of times '
#wordstring += 'it was the age of wisdom it was the age of foolishness'
#
#wordlist = wordstring.split()
#
#wordfreq = []
#for w in wordlist:
#    wordfreq.append(wordlist.count(w))
#
#print("String\n" + wordstring +"\n")
#print("List\n" + str(wordlist) + "\n")
#print("Frequencies\n" + str(wordfreq) + "\n")
#print("Pairs\n" + str(zip(wordlist, wordfreq)))


#hyrs = pd.Series(raw_data.half_years.unique()).sort_values()
#mean_rating = []
#four_plus = []
#n_recipes = []
#zeros = []
#for hy in hyrs:
#    df = raw_data[raw_data.half_years == hy]
#    mean_rating.append(df.rating.mean())
#    four_plus.append(df.four_plus.sum())
#    zeros.append(df.no_rating.sum())
#    n_recipes.append(len(df))
#    
#n_recipes = pd.Series(n_recipes)
#perc_zeros = zeros/n_recipes
#four_plus = four_plus/n_recipes
#n_recipes = n_recipes/n_recipes.max()
#mean_rating = np.array(mean_rating)/5