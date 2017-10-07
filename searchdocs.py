import nltk
from nltk.corpus import stopwords as sw
import numpy as np
import math
import os

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

class Document:
	def __init__(self, filename, onlyalpha = True, stopwords = False, stemmer = True):
		filepath = os.path.abspath(filename)
		meta = os.path.split(filepath)

		self.filepath = filepath
		self.dir = meta[0]
		self.filename = meta[1]
		self.content = ""
		f = open(filepath, 'r', encoding = 'utf-8')
		lines = f.readlines()
		for line in lines:
			self.content += line
		f.close()
		self.tokens = tools.tokenize(self.content, onlyalpha, stopwords, stemmer)

class Vocabulary:
	def __init__(self):
		self.vector = {}

	def add(self, tokens):
		for token in tokens:
			if token not in self.vector and not token.isspace() and token != '':
				self.vector[token] = len(self.vector)

	def indexOf(self, vocab):
		return self.vector[vocab]

	def size(self):
		return len(self.vector)

	def at(self, i): # get ith word in the vector
		return list(self.vector)[i]

	# vectorize = str -> numpy.array
	def word2vec(self, word):
		v = [0 for i in range(self.size())]
		if word in self.vector:
			v[self.indexOf(word)] = 1
		else:
			print("<ERROR> Word \'" + word + "\' Not Found")
		return np.array(v)

	# vectorize = Document -> numpy.array
	def doc2vec(self, doc):
		v = [0 for i in range(self.size())]
		for token in doc.tokens:
			if token in self.vector:
				v[self.indexOf(token)] += 1
		return np.array(v)

	def save(self, filename):
		f = open(filename, 'w', encoding='utf-8')
		for word in self.vector:
			f.write(word + '\n')
		f.close()

	def load(self, filename):
		f = open(filename, 'r', encoding='utf-8')
		lines = f.readlines()
		bow = [i[:-1] for i in lines]
		self.add(bow)
		f.close()
	
	def __str__(self):
		s = "Vocabulary("
		for word in self.vector:
			s += (str(self.vector[word]) + ": " + word + ", ")
		if self.size() != 0:
			s = s[:-2]
		s += ")"
		return s

vocab = Vocabulary()
