record = []
with open('test.txt') as f:
    for line in f:
        items = line.strip().split()
        for i in items[1:]:
            record.append(items[0]+' '+i+' 1\n')
with open('test.txt','w') as f:
    f.writelines(record)