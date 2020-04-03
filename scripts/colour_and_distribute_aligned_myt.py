import pyenki
import sys
from math import pi
from typing import Union
import numpy as np


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class ColourDistributedThymio2(pyenki.Thymio2):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name

    def controlStep(self, dt: float) -> None:
        """

        :param dt:
        """
        # 		//! The infrared sensor 0 (front-left-left)
        # 		//! The infrared sensor 1 (front-left)
        # 		//! The infrared sensor 2 (front-front)
        # 		//! The infrared sensor 3 (front-right)
        # 		//! The infrared sensor 4 (front-right-right)
        # 		//! The infrared sensor 5 (back-left)
        # 		//! The infrared sensor 6 (back-right)

        #  Move the robots in such a way they stand at equal distances from each other without using communication.
        #  Color the robot half and half they should understand on which side they are compared to the medium.

        # Check if there is a robot ahead using the infrared sensor 2 (front-front)
        # If the robot is in the range (~14 cm) the maximum front value shoud be ~1700
        front = self.prox_values[2]

        # Check if there is a robot ahead using the infrared sensor 5 and 6 (back-left) (back-right)
        back = np.mean(np.array([self.prox_values[5], self.prox_values[6]]))

        print(self.name, front, back)

        # if value > 3000:
        #     # A lot of light is reflected => there is an obstacle, so we stop and switch the LED to red
        #     speed: Union[float, int] = 0
        #
        #     if self.use_aseba_units:
        #         # Aseba uses integers and encodes LED values between 0 and 32 (fully lit)
        #         self.set_led_top(red=32)
        #     else:
        #         # The native enki interface uses instead float and encodes colors between 0 and 1
        #         self.set_led_top(red=1.0)
        # else:
        #
        #     if self.use_aseba_units:
        #         # Aseba uses integers and encodes speed between -500 and +500
        #         speed = 300
        #         self.set_led_top(green=32)
        #     else:
        #         # The native enki interface uses instead float and encodes speed in centimeter per second.
        #         speed = 10.0
        #         self.set_led_top(green=1.0)
        #
        # self.motor_left_target = speed
        # self.motor_right_target = speed


def setup(aseba: bool = False) -> pyenki.World:

    # Create an unbounded world
    world = pyenki.World()

    # Create multiple Thymios and position them such as all x-axes are aligned
    myt_quantity = 10
    myts = [ColourDistributedThymio2(name='myt%d' % (i + 1), use_aseba_units=aseba) for i in range(myt_quantity)]

    # The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range
    # ~ 14 cm is the proximity sensors maximal range
    distances = np.random.randint(1, 14, 10)

    # distance between the origin and the front of the robot along x-axis
    constant = 7.95

    for i, myt in enumerate(myts):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = myts[i - 1].position[0]

        # myt.position = (prev_pos + 14 + constant, 0)
        myt.position = (prev_pos + float(distances[i]) + constant, 0)
        myt.angle = 0
        world.add_object(myt)

    return world


def run(world: pyenki.World, gui: bool = False, T: float = 10, dt: float = 0.1) -> None:

    if gui:
        # We can either run a simulation [in real-time] inside a Qt application
        world.run_in_viewer(cam_position=(90, 0), cam_altitude=150.0, cam_yaw=0.0, cam_pitch=-pi / 2)
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
