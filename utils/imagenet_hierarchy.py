import numpy as np
import pandas as pd
from os.path import join as pjoin

data_path = '/home/ubuntu/Project/data'
class_mapping = pd.read_csv(pjoin(data_path, 'superClassMapping.csv'))

# prepare 1000 classes and 30 super classes
super_class_names = class_mapping.iloc[:, 3].unique().tolist()
n_class = 1000

imagenet_hierarchy = pd.DataFrame(columns=('class_name', 'parent_name', 'classID', 'parentID'),)
imagenet_hierarchy['class_name'] = class_mapping.loc[:, 'className']
imagenet_hierarchy['parent_name'] = class_mapping.loc[:, 'superClassName']
imagenet_hierarchy['classID'] = np.arange(n_class)

for row in range(n_class):
    imagenet_hierarchy.loc[row, 'parentID'] = \
        super_class_names.index(imagenet_hierarchy.loc[row, 'parent_name']) + n_class

# prepare hyper class
hyper_class = {'instrumentation':['container', 'device', 'conveyance', 'equipment',
                                  'implement', 'furnishing', 'toiletry'],
               'carnivore': ['canine', 'feline'],
               'mammal': ['primate', 'ungulate', 'carnivore'],
               'living_thing': ['fungus', 'fish', 'bird', 'amphibian', 'reptile',
                                'invertebrate', 'plant', 'mammal', 'canine', 'feline'],
               'artifact': ['structure', 'instrumentation', 'covering'],
               'whole_unit': ['artifact', 'living_thing', 'fruit'],
               'matter': ['substance', 'food', 'bar'],
               'physical_object': ['geological_formation', 'whole_unit'],
               'physical_entity': ['physical_object', 'matter', 'person'],
               'entity': ['physical_entity', 'abstraction'],
                }

add_row_index = 1000
n_cross_hierarchy = 0

for hyper_class_idx, class_name in enumerate(hyper_class.keys()):
    tmp_classes = hyper_class[class_name]
    # define parentID
    if class_name in super_class_names:
        # handle cross hierarchy case
        tmp_parentID = super_class_names.index(class_name) + n_class
        n_cross_hierarchy += 1
    else:
        tmp_parentID = hyper_class_idx + n_class + 30 - n_cross_hierarchy
    # define in each sub class
    for tmp_class in tmp_classes:
        # get classID
        if tmp_class in super_class_names:
            tmp_classID = super_class_names.index(tmp_class) + n_class
        else:
            tmp_classID = imagenet_hierarchy.loc[imagenet_hierarchy['parent_name']==tmp_class, 
                                             'parentID'].unique()[0]
        imagenet_hierarchy.loc[add_row_index] = [tmp_class, class_name, tmp_classID, tmp_parentID]
        add_row_index += 1
        
imagenet_hierarchy.to_csv(pjoin(data_path, 'imagenet_hierarchy.csv'), index=False)

