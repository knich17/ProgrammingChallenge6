from task_1 import *
from math import *

if __name__ == "__main__":
	# swap between the two lines for larger/smaller array printing
	np.set_printoptions(suppress=True) # print small
	# np.set_printoptions(threshold='nan', precision=4, suppress=True, linewidth=200) # print large

	parsed_data = dataParser()
	train_data, test_data = shuffleNSplit(parsed_data.data)
	print test_data