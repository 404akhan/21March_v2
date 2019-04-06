fname = '.vector_cache/1-6M-my-train-embedding-200d.txt'
fname_fixed = '.vector_cache/1-6M-my-train-embedding-200d-fixed.txt'

with open(fname) as f:
	content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

total = len(content)
error_n = 0
with open(fname_fixed, 'w') as f:

	for i in range(total):
		arr = content[i].split()

		if len(arr) != 201:
			error_n += 1
			print('error')
		else:
			f.write('%s\n' % (content[i]))

print(error_n, total)
print('done')