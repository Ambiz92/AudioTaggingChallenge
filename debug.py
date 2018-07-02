import json

stats = {}
temp = {}
temp2 = {}
class_list = []
class_list2 = []

stats['total_mistakes'] = 50
stats['total_tests'] = 500

for i in range(3):
    temp['class_name'] = "class" + str(i)
    temp['mistakes'] = 2

    for j in range(2):
                
        temp2['filename'] = "filename" + str(j)
        temp2['mistake_class'] = "mistake_class" + str(j)

        class_list2.append(temp2)
        temp2 = {}
        
    class_list.append(temp)
    class_list[i]['errors'] = class_list2
    stats['classes'] = class_list

    #class_list = []
    class_list2 = []
    temp = {}


print(json.dumps(stats, indent = 2))