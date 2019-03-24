from sklearn.metrics import f1_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print(f1_score(y_true, y_pred, average='macro'))

print(f1_score(y_true, y_pred, average='weighted'))



def ret():
	return 1, 2

def ret2():
	a,b = ret()
	return 3, a, b

print(ret2())