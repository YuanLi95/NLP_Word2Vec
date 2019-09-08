# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
from bs4 import BeautifulSoup
import  re
import nltk

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
example1 = BeautifulSoup(train["review"][0])
print(train["review"][0])
print(example1.get_text())
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()
words = [w for w in words if not w in nltk.corpus.stopwords.words("english")]
print (words)

