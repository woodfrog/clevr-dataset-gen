import random


# todo: 1. the position setting part of Layout
# todo: 2. set size randomly
# todo: 3. check how to determine the bounding box (not sure about the 2-D mapping of the 3-D position)

def position_add(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return x1 + x2, y1 + y2


class Layout:
    position_dict = {
        'left': [(-2, 0), (2, 0)],
        'right': [(2, 0), (-2, 0)],
        'top': [(0, 2), (0, -2)],
        'bottom': [(0, -2), (0, 2)],
        'next-to': [(0, 0), (2, 0)],
        'faraway': [(-3, -1), (1, 3)]
    }

    def __init__(self, layout_type):
        self.layout_type = layout_type
        self.position = (0, 0)
        self.left_child = None
        self.right_child = None

    def change_position(self, pos):
        self.position = position_add(self.position, pos)

    def set_children_pos(self):
        layout_setting = self.position_dict[self.layout_type]
        self.left_child.change_position(position_add(self.position, layout_setting[0]))
        self.right_child.change_position(position_add(self.position, layout_setting[1]))


class Describe:
    random_pos_range = 3

    def __init__(self, object_type):
        self.object_type = object_type
        self.attributes = dict()
        self.position = (0, 0)  # position: tuple of (x,y)
        self.size = 0

    def __repr__(self):
        return '<Describe, obj_type:{}>'.format(self.object_type)

    def set_attribute(self, name, obj):
        self.attributes[name] = obj

    def get_attribute(self, name):
        return self.attributes[name]

    def change_position(self, pos):
        self.position = position_add(self.position, pos)

    def set_random_pos(self):
        x = random.uniform(-self.random_pos_range, self.random_pos_range)
        y = random.uniform(-self.random_pos_range, self.random_pos_range)
        self.position = (x, y)


class Combine:
    def __init__(self, attr_type, val):
        self.carrier = None
        self.attr_type = attr_type
        self.attr_val = val

    def __repr__(self):
        return '<Combine, attr {}:{}>'.format(self.attr_type, self.attr_val)

    def set_carrier(self, obj):
        self.carrier = obj

    def get_carrier(self):
        if self.carrier is None:
            raise ValueError("The Combine module doesn't have carrier!")
        return self.carrier