import searchdocs as sd
import numpy as np
import sys

if __name__ == '__main__':

	sd.build("./data")
	if len(sys.argv) <= 1:
		exit()
	
	sd.build("./data")
	l = sd.search(sys.argv[1])
	for i in range(5):
		print(str(i+1) + ": " + l[i])

