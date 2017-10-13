from voca import Vocabulary
from collections import OrderedDict
import math
import numpy as np
import os
from doc import Document
import searchdocs as sd
import nltk

class IndexTable:
	def __init__(self, vocab, onlyalpha = True, stopwords = False, stemmer = True):
		self.idx = OrderedDict()
		self.iidx = OrderedDict()
		self.tf_vector = []
		self.idf_vector = []
		self.tfidf_vector = np.zeros(0)
		self.doc_idx = Vocabulary()
		self.vocab = vocab
		self.onlyalpha = onlyalpha
		self.stopwords = stopwords
		self.stemmer = stemmer

		if self.vocab.size() > 0:
			for word in self.vocab.vector:
				self.iidx[word] = set()
		else:
			raise ValueError("Vocabulary is empty")

	def add(self, doc):
		tokens = sd.tools.tokenize(doc.content, self.onlyalpha, self.stopwords, self.stemmer)
		self.idx[doc.filename] = nltk.Text(tokens).vocab()
		self.doc_idx.add(doc.filename)
		
		for token in tokens:
			self.iidx[token].add(doc.filename)
		
		v = [0 for i in range(self.vocab.size())]
		fd = self.idx[doc.filename]
		for term in fd:
			v[self.vocab.index(term)] = 1 + math.log10(fd[term])
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

	def _compute_idf(self):
		D = self.doc_idx.size()
		self.idf_vector = [D for i in range(self.vocab.size())]
		for term in self.iidx:
			self.idf_vector[self.vocab.index(term)] = math.log10(self.idf_vector[self.vocab.index(term)] / (1 + len(self.iidx[term])))
	
	def compute_tfidf(self):
		self._compute_idf()
		tf = np.array(self.tf_vector)
		idf = np.array(self.idf_vector)
		self.tfidf_vector = tf * idf

	def tfidf(self, term, doc):
		if doc.filename not in self.idx:
			raise ValueError("Document not found")

		if self.vocab.has(term):
			return self.tfidf_vector[self.doc_idx.index(doc.filename)][self.vocab.index(term)]
		else:
		  return 0

