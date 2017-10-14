import sys
sys.path.append('src')
import searchdocs as sd
import numpy as np
import sys

if __name__ == '__main__':

	if len(sys.argv) <= 1:
		exit()

	se = sd.SearchEngine()
	
	# se.build("./data")
	se.load('./sample/sample.voc', './sample/sample.itable')

	l = se.search(sys.argv[1])
	for i in range(5):
		print(str(i+1) + ": " + l[i])

