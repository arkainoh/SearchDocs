import numpy as np
import os
from collections import OrderedDict
import itable
from voca import Vocabulary
from doc import Document
import tools

class SearchEngine:

	def __init__(self, onlyalpha = True, stopwords = False, stemmer = True):
		self.onlyalpha = onlyalpha
		self.stopwords = stopwords
		self.stemmer = stemmer
		self.vocab = Vocabulary()

	def build(self, dirpath):
		dlist = os.listdir(dirpath)
		flist = []

		self.collect_files(dirpath, flist)
		
		for f in flist:
			doc = Document(os.path.join(dirpath, f))
			tokens = tools.tokenize(doc.content, self.onlyalpha, self.stopwords, self.stemmer)
			self.vocab.addall(tokens)
		
		self.itable = itable.IndexTable(self.vocab, self.onlyalpha, self.stopwords, self.stemmer)
		self.itable.addall(dirpath)
		self.itable.compute_tfidf()
	
	def load(self, voca_file, itable_file):
		self.vocab.load(voca_file)
		self.itable = itable.load(itable_file)

	def save(self, voca_file, itable_file):
		self.vocab.save(voca_file)
		itable.save(self.itable, itable_file)

	def collect_files(self, dirpath, l):
		dlist = os.listdir(dirpath)
		for tmp in dlist:
			tmppath = os.path.join(dirpath, tmp)
			if os.path.isfile(tmppath):
				l.append(tmp)
			elif os.path.isdir(tmppath):
				self.collect_files(tmppath, l)

	def search(self, query, vsm = True):
		tokens = tools.tokenize(query, self.onlyalpha, self.stopwords, self.stemmer)

		if(vsm):
			tf = self.vocab.doc2vec(tokens)
			for token in tokens:
				tf[self.vocab.index(token)] = self.itable.compute_tf(tf[self.vocab.index(token)])
			tfidf = tf * self.itable.idf_vector
			ret = [(self.itable.doc_idx.at(i), tools.cosine_similarity(tfidf, self.itable.tfidf_vector[i])) for i in range(self.itable.doc_idx.size())]

		else:
			v = np.zeros(self.itable.doc_idx.size())
			for token in tokens:
				if self.vocab.has(token):
					v += self.itable.tfidf_vector[:, self.vocab.index(token)]

			ret = [(self.itable.doc_idx.at(i), v[i]) for i in range(self.itable.doc_idx.size())]
		
		ret = sorted(ret, key = lambda x: x[1], reverse = True)
		return [i[0] for i in ret if i[1] > 0]


