from collections import namedtuple

class Position(namedtuple("Position", ["x", "y"])): pass
class Action(namedtuple("Action", ["piece_id", "position"])): pass