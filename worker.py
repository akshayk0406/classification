import sys
import os

classifiers = ['regression','nearestneighbor','centroid']
filetype = ['word','char']
features = ['tf','binary','sqrt','tfidf','binaryidf','sqrtidf']
base_name = 'data/20newsgroups'
output_file = base_name + '_output.txt'
validation_file = base_name + '_ridge.val'
neighbors = [5,10,15,20]

for classifier in classifiers:
	os.system('make ' + classifier)
	for f in filetype:
		for rep in features:
			cmd = './' + classifier + ' '
			cmd = cmd + base_name + '_' + f + '.ijv '
			cmd = cmd + base_name + '.rlabel '
			if classifier == 'regression':
				cmd = cmd + base_name + '_ridge.train '
			else:
				cmd = cmd + base_name + '.train '
			cmd = cmd + base_name + '.test '
			cmd = cmd + base_name + '.class '
			cmd = cmd + base_name + '_' + f + '.clabel '
			cmd = cmd + rep + ' '
			cmd = cmd + output_file + ' '
			
			if classifier == 'regression':
				cmd = cmd + validation_file
				print cmd
				os.system(cmd)
			elif classifier == 'nearestneighbor':
				tcmd = cmd
				for neigh in neighbors:
					cmd = tcmd
					cmd = cmd + str(neigh)
					print cmd
					os.system(cmd)
			else:
				print cmd	
				os.system(cmd)
