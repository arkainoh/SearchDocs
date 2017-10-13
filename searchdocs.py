import nltk
from nltk.corpus import stopwords as sw
import numpy as np
import math
import os
from collections import OrderedDict
from itable import IndexTable
from voca import Vocabulary
from doc import Document

class Tools:
	def tokenize(self, inputstr, onlyalpha = True, stopwords = False, stemmer = True):
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

tools = Tools()
vocab = Vocabulary()

def build(dirpath, onlyalpha = True, stopwords = False, stemmer = True):
	dlist = os.listdir(dirpath)
	flist = []

	for tmp in dlist:
		tmppath = os.path.join(dirpath, tmp)
		if os.path.isfile(tmppath):
			flist.append(tmp)
		elif os.path.isdir(tmppath):
			build(tmppath, onlyalpha, stopwords, stemmer)

	for f in flist:
		doc = Document(os.path.join(dirpath, f), onlyalpha, stopwords, stemmer)
		tokens = tools.tokenize(doc.content, onlyalpha, stopwords, stemmer)
		vocab.addall(tokens)
	
	global itable
	itable = IndexTable(vocab)
	itable.addall(dirpath)
	itable.compute_tfidf()

def search(query, onlyalpha = True, stopwords = False, stemmer = True):
	tokens = tools.tokenize(query, onlyalpha, stopwords, stemmer)

	v = np.zeros(itable.doc_idx.size())
	for token in tokens:
		if vocab.has(token):
			v += itable.tfidf_vector[:, vocab.index(token)]

	ret = [(itable.doc_idx.at(i), v[i]) for i in range(itable.doc_idx.size())]
	ret = sorted(ret, key = lambda x: x[1], reverse = True)

	return [i[0] for i in ret if i[1] > 0]

