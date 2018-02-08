import _init_paths
import os
import sys
import argparse
import os.path as osp
import random
from lib.tree import Tree
from modules import *

# parser = argparse.ArgumentParser(description='generate trees for CLEVR dataset')
# parser.add_argument('--output_dir', type=str, default='',
#                     help='output path for trees')
# parser.add_argument('--train_sample', type=int, default=0,
#                     help='number of samples for training')
# parser.add_argument('--test_sample', type=int, default=0,
#                     help='number of samples for testing')
# args = parser.parse_args()

######### hyperparameters ##########
# max level of the tree
max_level = 2

# module list
module_list = ['layout', 'describe', 'combine']
module_dict = {}

# children dict
children_dict = {}
children_dict['layout'] = 2
children_dict['describe'] = 1
children_dict['combine'] = 1

# objects list
module_dict['describe'] = ['cylinder', 'sphere', 'cube']

# attributes list
module_dict['combine'] = {'material': ['metal', 'rubber'],
                          'color': ['green', 'blue', 'yellow', 'red', 'cyan', 'brown', 'gray', 'purple'],
                          'size': ['large', 'small']}

# relations list
module_dict['layout'] = ['top', 'left', 'right', 'bottom', ]
# 'next-to', 'faraway'

# degree range: curently randomize this number, \
# no need for input from the tree
# deg_range = [0, 360]

# def get_flag(level, maxlevel):
#     if level + 1 >= max_level:
#         flag = 0
#     else:
#         # flag = random.randint(0, 1)
#         flag = 1
#
#     return flag


def expand_tree(tree, level, parent, memorylist, child_idx, max_level):
    if parent is None or parent.function == 'layout':
        if level + 1 >= max_level:
            valid = [1]
        else:
            valid = [0, 1]

        # sample module
        module_id = random.randint(0, len(valid) - 1)
        tree.function = module_list[valid[module_id]]

        # sample content
        word_id = random.randint(0, len(module_dict[tree.function]) - 1)
        tree.word = module_dict[tree.function][word_id]

        if tree.function == 'layout':
            tree.function_obj = Layout(tree.word)
        else:
            tree.function_obj = Describe(tree.word)

        # num children
        if level + 1 > max_level:
            tree.num_children = 0
        else:
            tree.num_children = children_dict[tree.function]
            if tree.function == 'describe' and parent is not None:  # then the parent must be a layout node
                if child_idx == 0:
                    parent.function_obj.left_child = tree.function_obj
                else:
                    parent.function_obj.right_child = tree.function_obj

        for i in range(tree.num_children):
            tree.children.append(Tree())
            tree.children[i] = expand_tree(tree.children[i], level + 1, tree, [], i, max_level)

    # must contain only one child node, which is a combine node
    elif parent.function == 'describe' or parent.function == 'combine':
        valid = [2]
        # no need to sample module for now
        module_id = 0
        tree.function = module_list[valid[module_id]]

        # sample content
        # sample which attributes
        if len(set(module_dict[tree.function].keys()) - set(memorylist)) <= 1:
            full_attribute = True
        else:
            full_attribute = False

        attribute = random.sample(set(module_dict[tree.function].keys()) - set(memorylist), 1)[0]
        memorylist += [attribute]
        word_id = random.randint(0, len(module_dict[tree.function][attribute]) - 1)
        tree.word = module_dict[tree.function][attribute][word_id]

        if isinstance(parent.function_obj, Describe):
            carrier = parent.function_obj
        else:
            carrier = parent.function_obj.get_carrier()

        tree.function_obj = Combine(attribute, tree.word)
        tree.function_obj.set_carrier(carrier)
        carrier.set_attribute(attribute, tree.function_obj)

        if not full_attribute:
            tree.num_children = children_dict[tree.function]

            for i in range(tree.num_children):
                tree.children.append(Tree())
                tree.children[i] = expand_tree(tree.children[i], level + 1, tree, memorylist, i, max_level)

    else:
        raise ValueError('Wrong function.')
    return tree


def visualize_tree(trees):
    for i in range(len(trees)):
        print('************** tree **************')
        _visualize_tree(trees[i], 0)
        print('**********************************')


def _visualize_tree(tree, level):
    if tree == None:
        return
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        _visualize_tree(tree.children[i], level + 1)

    print(' ' * level + tree.word)
    if isinstance(tree.function_obj, Describe):
        print(tree.function_obj.attributes, tree.function_obj)
        if tree.function != 'combine':
            print('position {}'.format(tree.function_obj.position))

    for i in range((tree.num_children - 1) // 2, -1, -1):
        _visualize_tree(tree.children[i], level + 1)

    return


def allign_tree(tree, level):
    """
        A pre-order traversal
    :param tree:
    :return:
    """
    if tree is None:
        return

    if tree.function == 'describe' and level == 0:
        tree.function_obj.set_random_pos()
    elif tree.function == 'layout':
        tree.function_obj.set_children_pos()
        for i in range(tree.num_children):
            allign_tree(tree.children[i], level+1)
    else:
        pass


def extract_objects(tree):
    objects = list()

    if tree is None:
        return objects

    if tree.function == 'describe':
        objects.append(tree.function_obj)
    elif tree.function == 'layout':
        for i in range(tree.num_children):
            objects += extract_objects(tree.children[i])
    else:
        pass

    return objects


def sample_tree(max_level):
    tree = Tree()
    tree = expand_tree(tree, 0, None, [], 0, max_level)
    allign_tree(tree, 0)
    return tree

if __name__ == '__main__':
    random.seed(12113)

    # tree = Tree()
    # tree = expand_tree(tree, 0, None, [], 0)
    # allign_tree(tree)

    num_sample = 1
    trees = []
    for i in range(num_sample):
        treei = Tree()
        treei = expand_tree(treei, 0, None, [], 0, max_level=2)
        allign_tree(treei, 0)
        objects = extract_objects(treei)
        trees += [treei]
        print(objects)

    visualize_tree(trees)
