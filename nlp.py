import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

#quoting to ignore double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t', quoting = 3)


#to clear text
import re
import nltk
nltk.download('stopwords')
#to include stopwords in spyder
from nltk.corpus import stopwords
#for stemming
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,1000):
    #removing irrelevant words
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    
    #converting to lower case
    review = review.lower()
    
    review = review.split()
    #creating object for stemmer
    ps = PorterStemmer()
    
    #checks if the word is not in stopwords
    #faster in set than inn a list
    #review = [word for word in review if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #joining the achieved string back again
    review = ' '.join(review)
    corpus.append(review)
    
