import numpy as np

# Each element in this array represents the number of different allowed valued for the corresponding attribute
#attribute_discrte = [True, False, False, True, True, True, True, False, True, True, False, True, True, False, False, True]

# Array used to convert string attributes to integers
# attributes = [["b", "a"], [], [], ["u", "y", "l", "t"], ["g", "p", "gg"], \
# ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"], \
# ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"], [], ["t", "f"], ["t", "f"], \
# [], ["t", "f"], ["g", "p", "s"], [], [], ["+", "-"]]


class dataParser:
	"""
	A data parser parses data from a text file and represents information useful for analysis, such as
	type of attributes (discrete or binary). Data is stored as a 2D Numpy array of floats, missing values
	are replaced with the most likely attribute (ignoring context - the mean or median of that attribute),
	and discrete variables are converted to integer ID's.
	"""
	def __init__(self, file='records.txt', missing_value='?', delimiter=',', attributes=None, discrete_attributes=None):
		"""
		:param file: location of file to parse data from
		:param missing_value: value that represents missing data
		:param delimiter: delimiter
		:param attributes: optional array of attribute values, where each element in the array is either
		an array of possible values if discrete, or an empty array if continuous
		:param discrete_attributes: optional array of booleans, true if that column is discrete.
		Length should match number of columns in data.
		"""
		self.file = file
		self.missing_values = [missing_value, -100]			#-100 fills continuous values
		temp_data = self.getData(delimiter, missing_value)
		self.num_rows, self.num_attributes = temp_data.shape
		self.discrete_attributes = discrete_attributes
		self.classification_index = self.num_attributes-1

		#Calculate discrete/continuous attribute distribution if not provided
		if attributes is None:
			self.attributes = []
			self.attribute_type = np.empty([self.num_attributes], dtype=int)

			for col in range(0, self.num_attributes):
				#Check if discrete or continuous, discrete using check array if passed or otherwise
				#if not passable to a float
				if (discrete_attributes is None and not is_float(temp_data[0,col])) or (discrete_attributes is not None and discrete_attributes[col]):
					values = temp_data[:,col]
					cleaned = values[values != '?']			# Remove missing data
					self.attributes.append((np.unique(cleaned)).tolist())
					self.attribute_type[col] = len(self.attributes[col])
				else:
					self.attributes.append([])
					self.attribute_type[col] = -1
		else:
			self.attributes = attributes
			self.attribute_type = np.empty([self.num_attributes], dtype=int)
			for col in range(0, self.num_attributes):
				length = len(attributes[col])
				self.attribute_type[col] = length if length != 0 else -1

		#Get final data
		self.data = self.resolveMissing(self.parseData(temp_data))


	def getData(self, delimiter, missing_value):
		"""
		Gets numpy of strings array from data
		:param delimiter: delimiter
		:param missing_value: missing value
		:return: numpy array of strings representing data
		"""
		return np.genfromtxt(self.file, dtype='S10', delimiter=delimiter, missing_values=missing_value)


	def parseData(self, data):
		"""
		Converts discrete variables from original data into ints so data can be stored as floats
		:param data: 2D numpy array of data
		:return: A 2D numpy array of floats
		"""
		parsed_data = np.empty(data.shape)
		x, y = data.shape

		for row in range(0,x):
			for col in range(0,y):
				val = data[row,col]
				# Replace missing with new missing representation
				if val == '?':
					parsed_data[row,col] = -100.0
				else:
					# Convert discrete values to ints
					if self.attribute_type[col] != -1:
						parsed_data[row,col] = float(self.attributes[col].index(val))
					else:
						parsed_data[row,col] = float(val)

		return parsed_data


	def resolveMissing(self, data):
		"""
		Replaces the missing values in 2D data with the mean for continuous variables, and median of discrete
		:param data: 2D Numpy array
		:return: New data in 2D numpy array where missing values have been replaced.
		"""
		for j in range(0, self.num_attributes):		#for every column
			temp = data[:, j]						#get a column
			temp = temp[temp != -100]				#strip -100's (numpy masked array could be quicker)
			#mostlikely is the average of continuous variables or median of discrete
			mostLikely = np.average(temp) if self.attribute_type[j] == -1 else np.median(temp)
			for i in range(0, len(data)):			#for every row
				if data[i][j] == -1:				#if value is -1
					data[i][j] = mostLikely  		#set value to median

		return data




def shuffleNSplit(data):
	"""
	Shuffles the given data and returns two arrays, split at 80% and 20% of the original size respectively
	:param data: 2D numpy array to split
	:return: Two arrays, one that is 80% of the shuffled data, one that is 20%
	"""
	dataCpy = data[:]
	np.random.shuffle(dataCpy)
	bound = int(round(len(data)*0.8))
	return dataCpy[:bound], dataCpy[bound:]


def is_float(value):
	"""
	Simple helper for testing if a variable (e.g. a string) can be converted to a float
	:param value: value to test conversion
	:return: True if possible to parse
	"""
	try:
		float(value)
		return True
	except:
		return False




#______________________________________________________________________________
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                         		 MAIN

if __name__ == "__main__":
	# swap between the two lines for larger/smaller array printing
	np.set_printoptions(suppress=True) #print small
	# np.set_printoptions(threshold='nan', precision=4, suppress=True, linewidth=200) #print large

	data = dataParser()
	train_data, test_data = shuffleNSplit(data.data)

	print train_data, len(train_data)
	print test_data, len(test_data)