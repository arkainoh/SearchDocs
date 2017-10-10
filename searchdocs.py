import nltk
from nltk.corpus import stopwords as sw
import numpy as np
import math
import os
from collections import OrderedDict

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
		self.vector = OrderedDict()
		self.entry = []

	def add(self, token):
		if token not in self.vector and not token.isspace() and token != '':
			self.vector[token] = len(self.vector)
			self.entry.append(token)

	def addall(self, tokens):
		for token in tokens:
			self.add(token)

	def has(self, token):
		return token in self.vector

	def index(self, token):
		return self.vector[token]

	def size(self):
		return len(self.vector)

	def at(self, i): # get ith word in the vector
		return self.entry[i]

	# vectorize = str -> numpy.array
	def word2vec(self, word):
		v = [0 for i in range(self.size())]
		if word in self.vector:
			v[self.index(word)] = 1
		else:
			raise ValueError("Word \'" + word + "\' Not Found")
		return np.array(v)

	# vectorize = Document -> numpy.array
	def doc2vec(self, doc):
		v = [0 for i in range(self.size())]
		for token in doc.tokens:
			if token in self.vector:
				v[self.index(token)] += 1
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
		self.addall(bow)
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

class IndexTable:
	def __init__(self):
		self.idx = OrderedDict()
		self.iidx = OrderedDict()
		self.tf_vector = []
		self.idf_vector = []
		self.tfidf_vector = np.zeros(0)
		self.doc_idx = Vocabulary()

		if vocab.size() > 0:
			for word in vocab.vector:
				self.iidx[word] = set()
		else:
			raise ValueError("Vocabulary is empty")

	def add(self, doc):
		self.idx[doc.filename] = nltk.Text(doc.tokens).vocab()
		self.doc_idx.add(doc.filename)
		
		for token in doc.tokens:
			self.iidx[token].add(doc.filename)
		
		v = [0 for i in range(vocab.size())]
		fd = self.idx[doc.filename]
		for term in fd:
			v[vocab.index(term)] = 1 + math.log10(fd[term])
		self.tf_vector.append(v)

	def addall(self, dirpath, onlyalpha = True, stopwords = False, stemmer = True):
		dlist = os.listdir(dirpath)
		flist = []

		for tmp in dlist:
			tmppath = os.path.join(dirpath, tmp)
			if os.path.isfile(tmppath):
				flist.append(tmp)
			elif os.path.isdir(tmppath):
				self.addall(tmppath, onlyalpha, stopwords, stemmer)

		for f in flist:
			doc = Document(os.path.join(dirpath, f), onlyalpha, stopwords, stemmer)
			self.add(doc)

	def _calculate_idf(self):
		D = self.doc_idx.size()
		self.idf_vector = [D for i in range(vocab.size())]
		for term in self.iidx:
			self.idf_vector[vocab.index(term)] = math.log10(self.idf_vector[vocab.index(term)] / (1 + len(self.iidx[term])))
	
	def calculate_tfidf(self):
		self._calculate_idf()
		tf = np.array(self.tf_vector)
		idf = np.array(self.idf_vector)
		self.tfidf_vector = tf * idf

	def tfidf(self, term, doc):
		if doc.filename not in self.idx:
			raise ValueError("Document not found")

		if vocab.has(term):
			return self.tfidf_vector[self.doc_idx.index(doc.filename)][vocab.index(term)]
		else:
		  return 0

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
		vocab.addall(doc.tokens)
	
	global itable
	itable = IndexTable()
	itable.addall(dirpath)
	itable.calculate_tfidf()

def search(query, onlyalpha = True, stopwords = False, stemmer = True):
	tokens = tools.tokenize(query, onlyalpha, stopwords, stemmer)

	v = np.zeros(itable.doc_idx.size())
	for token in tokens:
		if vocab.has(token):
			v += itable.tfidf_vector[:, vocab.index(token)]

	ret = [(itable.doc_idx.at(i), v[i]) for i in range(itable.doc_idx.size())]
	ret = sorted(ret, key = lambda x: x[1], reverse = True)

	return [i[0] for i in ret if i[1] > 0]

