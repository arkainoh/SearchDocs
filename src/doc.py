import os

class Document:
	def __init__(self, filename):
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

