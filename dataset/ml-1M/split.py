import random
train = []
test = []
with open('ratings.dat') as f:
    for line in f:
        items = line.strip().split('::')
        new_line = ' '.join(items[:-1])+'\n'
        if int(items[-2])<4:
            continue
        if random.random() > 0.2:
            train.append(new_line)
        else:
            test.append(new_line)

with open('train.txt','w') as f:
    f.writelines(train)

with open('test.txt','w') as f:
    f.writelines(test)