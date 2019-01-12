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
    
#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 
x = cv.fit_transform(corpus).toarray()

#dependent variable
y = dataset.iloc[:, 1].values




# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
    
