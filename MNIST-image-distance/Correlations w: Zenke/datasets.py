from skdata.larochelle_etal_2007 import dataset as l7
from keras.utils import np_utils
import csv


def createRandomizedDataset():
    tasks = [l7.MNIST_Basic(), l7.MNIST_Rotated(), l7.MNIST_Noise1(), \
			 l7.MNIST_Noise3(), l7.MNIST_Noise5()]
    num = 5
    name = tasks[num]
    task = name.classification_task()
    raw_data, raw_labels = task 
    classes = 10
    labels = np_utils.to_categorical(raw_labels, classes)
    t = int(raw_data.shape[0] * 5/7)
             
    f = open('t5.csv', 'w')
    with f:
        mywriter = csv.writer(f)
        print(num)
        mywriter.writerows(raw_data[:t])
        print(num)
    f.close()
    f = open('labels5.csv', 'w')
    with f:
        mywriter = csv.writer(f)
        print(num)
        mywriter.writerows(labels[:t])
        print(num)
    return l7
'''    datasets = dict()
    for i in range(len(tasks)):
        name = tasks[i]
        task = name.classification_task()
        raw_data, raw_labels = task 
        classes = 10
		
        labels = np_utils.to_categorical(raw_labels, classes)

        data = raw_data.reshape(raw_data.shape[0], 28 * 28)
		
        print(int(raw_data.shape[0] * 5/7))
        data = raw_data
        training_ex = int(len(data) * 5/7)
        valid_ex = int(len(data) * 1/7) + training_ex

        datasets[i] = {"train": data[:training_ex], "test": data[valid_ex:], \
						"valid": data[training_ex : valid_ex], "validlabels": labels[training_ex : valid_ex], \
		               "trainlabels": labels[:training_ex], "tstlabels": labels[valid_ex:]}
    return l7'''
createRandomizedDataset()