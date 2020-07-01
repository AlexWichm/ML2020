import numpy as np


def read_csv( path, header = 1 ):
		
		
		in1 = open(path, "r")
		heads = None
		dat = []
		
		for x in in1:
			if header:
				heads = x.replace("\n","").split(",")
				header = 0
			else:
				dat.append(x.replace("\n","").split(","))
		
		dat = np.array(dat, dtype=np.float32)
		return heads, dat

def split_data(data):
	train = []
	y = []
	for i in data:
		y.append(i[len(i)-1])
		train.append(i[0:len(i)-1])

	train = np.array(train)
	return y, train
	



# Die Funktion in Vektor Form: 										   y = beta * X + e
# Dies kann mithilfe linearer Algebra umgeschrieben werden zu:		beta = (y - e) * (X)^-1  

# Der * ist hierbei das Dot-Produkt und (X)^-1 das Invese der Matrix X
	
def multivariateLinearRegression(data):
	
	y,X = split_data(data)
	sigma = 0.1 # beispielsweise, müsste noch spezifischer angepasst werden
	e = np.random.normal(0, sigma, len(y))
	new_y = []
	
	
	for i in range(0,len(y)):
		new_y.append(y[i]-e[i])
		
	invX = np.linalg.pinv(X) # inverse of vector X
	
	Beta = np.dot(invX, new_y)
	
	return Beta
		
	
	
	
file = "livestock.csv"
head, data = read_csv(file)

print(head)
print(data)

y,train = split_data(data)


beta = multivariateLinearRegression(data)

print (np.dot(train,beta))
print (y)





