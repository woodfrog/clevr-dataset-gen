import _init_paths
import os
import sys
import argparse
import os.path as osp
import random
import numpy as np
from lib.tree import Tree
from modules import Layout, Combine, Describe

# parser = argparse.ArgumentParser(description='generate trees for CLEVR dataset')
# parser.add_argument('--output_dir', type=str, default='',
#                     help='output path for trees')
# parser.add_argument('--train_sample', type=int, default=0,
#                     help='number of samples for training')
# parser.add_argument('--test_sample', type=int, default=0,
#                     help='number of samples for testing')
# args = parser.parse_args()


######### hyperparameters ##########

# module list
module_list = ['layout', 'describe', 'combine']

# children dict
children_dict = dict()
children_dict['layout'] = 2
children_dict['describe'] = 1
children_dict['combine'] = 1

# we will have two split dict for modules for designing a zero-shot setting

module_dict_split1 = dict()
module_dict_split2 = dict()
module_dict_all = dict()

# Zero shot split
# objects list
module_dict_split1['describe'] = ['cube']
module_dict_split2['describe'] = ['cylinder', 'sphere']
module_dict_all['describe'] = ['cylinder', 'cube', 'sphere']

# attributes list
attribute_list = ['material', 'color', 'size']

module_dict_split1['combine'] = {'material': ['metal'],
                                 'color': ['green', 'blue', 'yellow', 'red'],
                                 'size': ['large', 'small']}

module_dict_split2['combine'] = {'material': ['rubber'],
                                 'color': ['cyan', 'brown', 'gray', 'purple'],
                                 'size': ['small', 'large']}

module_dict_all['combine'] = {'material': ['rubber', 'metal'],
                              'color': ['cyan', 'brown', 'gray', 'purple', 'green', 'blue', 'yellow', 'red'],
                              'size': ['small', 'large']}
# relations list
module_dict_split1['layout'] = ['left', 'left-front', 'right-front']
module_dict_split2['layout'] = ['right', 'right-behind', 'left-behind']
module_dict_all['layout'] = ['right', 'left', 'right-behind', 'left-front', 'left-behind',
                             'right-front']

module_dicts_zeroshot = [module_dict_split1, module_dict_split2]
module_dict_normal = module_dict_all

pattern_map = {'describe': 0, 'material': 1, 'color': 2, 'size': 3, 'layout': 4}

zs_training_patterns = [(0, 1, 0, 1, 0), (1, 0, 1, 0, 1)]
zs_training_probs = [1.0/3, 2.0/3]
zs_test_patterns = [(1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (1, 1, 0, 0, 0), (0, 1, 1, 1, 0),
                    (1, 0, 0, 0, 1), (0, 1, 1, 1, 1), (1, 0, 0, 0, 0)]
zs_test_probs = [1.0/6, 1.0/12, 1.0/12, 1.0/6, 1.0/12, 1.0/6, 1.0/12, 1.0/6] 

def expand_tree(tree, level, parent, memorylist, child_idx, max_layout_level, add_layout_prob, train, zero_shot=False, metadata_pattern=None):
    if parent is None or parent.function == 'layout':
        # sample module, the module can be either layout or describe here
        if level + 1 > max_layout_level:
            module_idx = 1
        else:
            rand = random.random()
            if rand >= 1 - add_layout_prob:
                module_idx = 0
            else:
                module_idx = 1
        tree.function = module_list[module_idx]
        if zero_shot and (level == 0 or tree.function == 'describe'):
            r = random.random()
            if train:
                metadata_pattern = _choose_pattern(zs_training_patterns, zs_training_probs, r)
            else:
                metadata_pattern = _choose_pattern(zs_test_patterns, zs_test_probs, r)
        # sample content
        if zero_shot:
            assert (metadata_pattern is not None)
            dict_index = metadata_pattern[pattern_map[tree.function]]
            module_dict = module_dicts_zeroshot[dict_index]
        else:
            module_dict = module_dict_normal

        word_id = random.randint(0, len(module_dict[tree.function]) - 1)
        tree.word = module_dict[tree.function][word_id]

        if tree.function == 'layout':
            tree.function_obj = Layout(tree.word)
            print('add layout')
        else:
            tree.function_obj = Describe(tree.word)
            print('add describe')

        tree.num_children = children_dict[tree.function]
        if parent is not None:  # then the parent must be a layout node
            if child_idx == 0:
                parent.function_obj.left_child = tree.function_obj
            else:
                parent.function_obj.right_child = tree.function_obj

        for i in range(tree.num_children):
            tree.children.append(Tree())
            tree.children[i] = expand_tree(tree.children[i], level + 1, tree, [], i, max_layout_level,
                                           add_layout_prob,
                                           train, zero_shot, metadata_pattern)

    # must contain only one child node, which is a combine node
    elif parent.function == 'describe' or parent.function == 'combine':
        print('add combine')
        valid = [2]
        # no need to sample module for now
        module_id = 0
        tree.function = module_list[valid[module_id]]

        # sample content
        # sample which attributes
        if len(set(attribute_list) - set(memorylist)) <= 1:
            full_attribute = True
        else:
            full_attribute = False

        attribute = random.sample(set(attribute_list) - set(memorylist), 1)[0]
        memorylist += [attribute]

        if zero_shot:
            assert(metadata_pattern is not None)
            dict_idx = metadata_pattern[pattern_map[attribute]]
            module_dict = module_dicts_zeroshot[dict_idx]
        else:
            module_dict = module_dict_normal

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
                tree.children[i] = expand_tree(tree.children[i], level + 1, tree, memorylist, i, max_layout_level,
                                               add_layout_prob,
                                               train, zero_shot, metadata_pattern)
    else:
        raise ValueError('Wrong function.')
    return tree


def _choose_pattern(patterns, probs, r):
    assert(sum(probs) == 1, 'Given prob list should sum up to 1')
    assert(len(patterns) == len(probs), 'Given patterns should have the same length as the given probs')
    accum = 0
    for i, prob in enumerate(probs):
        accum += prob
        if r < accum:
            return patterns[i]

def visualize_trees(trees):
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
        A pre-order traversal, set the position of tree nodes according to the layouts
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
            allign_tree(tree.children[i], level + 1)
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


def sample_tree(max_layout_level, add_layout_prob, zero_shot=False, train=True):
    tree = Tree()
    tree = expand_tree(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, zero_shot=zero_shot)
    allign_tree(tree, 0)
    return tree


def refine_tree_info(tree):
    tree = _set_describe_bbox(tree)
    tree = _set_layout_bbox(tree)
    return tree


def remove_function_obj(tree):
    tree = _remove_function_obj(tree)
    return tree


def _remove_function_obj(tree):
    if hasattr(tree, 'function_obj'):
        delattr(tree, 'function_obj')
    for child in tree.children:
        _remove_function_obj(child)
    return tree


def _set_describe_bbox(tree):
    function_obj = tree.function_obj
    # set the bbox for the tree node
    if hasattr(function_obj, 'bbox'):
        left_top_coord, right_bottom_coord = function_obj.bbox
        bbox = (left_top_coord[0], left_top_coord[1], right_bottom_coord[0] - left_top_coord[0],
                right_bottom_coord[1] - left_top_coord[1])
        tree.bbox = np.array(bbox)

    for child in tree.children:
        _set_describe_bbox(child)
    return tree


def _set_layout_bbox(tree):
    if tree.function != 'layout':
        return tree
    else:
        for child in tree.children:
            _set_layout_bbox(child)
        # set the bbox for layout module
        left_child_bbox = tree.children[0].bbox
        right_child_bbox = tree.children[1].bbox
        tree.bbox = np.array(_combine_bbox(left_child_bbox, right_child_bbox))

        return tree


def _correct_layout_word(tree):
    if tree.function != 'layout':
        return tree
    else:
        left_child_bbox = tree.children[0].bbox
        right_child_bbox = tree.children[1].bbox
        if left_child_bbox[0] < right_child_bbox[0]:
            if right_child_bbox[1] - 5 < left_child_bbox[1] < right_child_bbox[1] + 5:
                tree.word = 'left'
            elif left_child_bbox[1] <= right_child_bbox[1] - 5:
                tree.word = 'left-behind'
            else:
                tree.word = 'left-front'
        else:
            if right_child_bbox[1] - 5 < left_child_bbox[1] < right_child_bbox[1] + 5:
                tree.word = 'right'
            elif left_child_bbox[1] <= right_child_bbox[1] - 5:
                tree.word = 'right-behind'
            else:
                tree.word = 'right-front'

        for child in tree.children:
            _correct_layout_word(child)

        return tree


def _combine_bbox(bbox1, bbox2):
    left = min(bbox1[0], bbox2[0])
    top = min(bbox1[1], bbox2[1])
    right = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    bottom = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    return [left, top, right - left, bottom - top]


if __name__ == '__main__':
    # random.seed(12113)
    #
    # # tree = Tree()
    # # tree = expand_tree(tree, 0, None, [], 0)
    # # allign_tree(tree)
    #
    # num_sample = 1
    # trees = []
    # for i in range(num_sample):
    #     treei = Tree()
    #     treei = expand_tree(treei, 0, None, [], 0, max_level=2)
    #     allign_tree(treei, 0)
    #     objects = extract_objects(treei)
    #     trees += [treei]
    #     print(objects)
    #
    # visualize_tree(trees)

    for i in range(1):
        tree = sample_tree(max_layout_level=2, add_layout_prob=0.6, zero_shot=True, train=True)
        visualize_trees([tree])
