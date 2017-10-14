import nltk
from nltk.corpus import stopwords as sw

def tokenize(inputstr, onlyalpha = True, stopwords = False, stemmer = True):
	inputstr = inputstr.lower()
	tokens = nltk.word_tokenize(inputstr)

	if(onlyalpha):
		tokens = [i for i in tokens if i.isalpha()]

	if(not stopwords):
		stpwrds = set(sw.words('english'))
		tokens = [i for i in tokens if i not in stpwrds]

	if(stemmer):
		stmr = nltk.stem.porter.PorterStemmer()
		tokens = [stmr.stem(i) for i in tokens]

	return tokens

