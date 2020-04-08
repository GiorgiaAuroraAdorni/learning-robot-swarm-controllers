import pyenki
import sys
from math import pi, sqrt
import numpy as np
from pid import PID


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class DistributedThymio2(pyenki.Thymio2):

    INITIAL = 0
    DISTRIBUTING = 1

    def __init__(self, name, index, goal_position, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.index = index
        self.goal_position = goal_position

        # Aseba uses integers and encodes speed between -500 and +500
        self.distributed_controller = PID(-0.01, 0, 0, max_out=500, min_out=-500)
        self.state = self.INITIAL
        self.distribute = True

    def euclidean_distance(current_position, goal_postion):
        """
        :param current_position
        :param goal_postion
        :return: Euclidean distance between current and the goal position
        """
        return sqrt(pow((goal_postion.position[0] - current_position.position[0]), 2) +
                    pow((goal_postion.position[1] - current_position.position[1]), 2))

    def linear_vel(current_position, goal_postion, constant=4):
        """
        :param current_position
        :param goal_postion
        :param constant
        :return: clipped linear velocity
        """
        velocity = constant * euclidean_distance(current_position, goal_postion)
        return min(max(-500, velocity), 500)

    def move_to_goal(current_position, goal_postion):
        """
        Moves the thymio to the goal.
        :param current_position
        :param goal_postion
        :return: speed
        """
        return linear_vel(current_position, goal_postion)






def setup(aseba: bool = False) -> pyenki.World:
    """
    Set up the world and create the thymios
    :param aseba
    :return world
    """
    # Create an unbounded world
    world = pyenki.World()

    # Create multiple Thymios and position them such as all x-axes are aligned
    myt_quantity = 8
    myts = [DistributedThymio2(name='myt%d' % (i + 1), index=i, goal_position=None, use_aseba_units=aseba)
            for i in range(myt_quantity)]

    # The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range
    # ~ 14 cm is the proximity sensors maximal range
    distances = np.random.randint(5, 8, 8)

    # Distance between the origin and the front of the robot along x-axis
    constant = 7.95

    for i, myt in enumerate(myts):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = myts[i - 1].position[0]

        current_pos = prev_pos + float(distances[i]) + constant
        myt.position = (current_pos, 0)
        myt.angle = 0

    # Decide the goal pose for each robot
    goal_positions = np.linspace(myts[0].position[0], myts[myt_quantity - 1].position[0], num=8)

    for i, myt in enumerate(myts):
        myt.goal_position = (goal_positions[i], 0)
        world.add_object(myt)

    return world


def run(world: pyenki.World, gui: bool = False, T: float = 10, dt: float = 0.1) -> None:
    """
    :param world
    :param gui
    :param T
    :param dt: update timestep in seconds, should be below 1 (typically .02-.1)
    """

    if gui:
        # We can either run a simulation [in real-time] inside a Qt application
        world.run_in_viewer(cam_position=(60, 0), cam_altitude=110.0, cam_yaw=0.0, cam_pitch=-pi / 2,
                            walls_height=10.0, orthographic=True)
    else:
        # or we can write our own loop that run the simulation as fast as possible.
        steps = int(T // dt)
        for _ in range(steps):
            world.step(dt)


if __name__ == '__main__':
    world = setup('--aseba' in sys.argv)

    try:
        run(world, '--gui' in sys.argv)
    except:
        raise
