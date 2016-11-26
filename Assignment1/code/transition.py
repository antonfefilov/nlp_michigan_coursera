class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.stack:
            return -1
        if conf.stack[-1] == 0:
            return -1
        for arc in conf.arcs:
          if conf.stack[-1] == arc[0]:
            break
            return -1

        # print "Doing left_arc"
        # print "Before:", conf

        idx_wi = conf.buffer[0]
        idx_wj = conf.stack.pop()

        conf.arcs.append((idx_wi, relation, idx_wj))

        # print "After:", conf
        # print relation
        # print

        # return conf

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # print "Doing right_arc"
        # print "Before:", conf

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

        # print "After:", conf
        # print relation
        # print

        # return conf

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.arcs or not conf.stack:
            return -1
        for arc in conf.arcs:
          if not (conf.stack[-1] == arc[0]):
            break
            return -1

        # print "Doing reduce"
        # print "Before:", conf

        conf.stack.pop()

        # print "After:", conf
        # print

        # return conf

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer:
            return -1

        # print "Doing shift"
        # print "Before:", conf

        conf.stack.append(conf.buffer.pop(0))

        # print "After:", conf
        # print

        # return conf
