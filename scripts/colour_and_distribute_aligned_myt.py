import pyenki
import sys
import math
from typing import Union


# TODO
#  Move the robots in such a way they stand at equal distances from each other without using communication.
#  Color the robot half and half (assuming that the total number of robots is well known): they should understand
#  on which side they are compared to the medium.
#  Assume that the robots are already arranged in an "indian row" (all x-axes aligned) and within the
#  proximity sensor range


class ColourDistributedThymio2(pyenki.Thymio2):


if __name__ == '__main__':
    world = setup('--aseba' in sys.argv)
    run(world, '--gui' in sys.argv)