import pyenki
import sys
from math import pi
import numpy as np
from pid import PID


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class DistributedThymio2(pyenki.Thymio2):
    INITIAL = 0
    COLOURING = 1

    def __init__(self, myt_quantity, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.myt_quantity = myt_quantity
        self.name = name

        # Encodes speed between -16.6 and +16.6 (aseba units: [-500,500])
        self.distributed_controller = PID(-0.01, 0, 0, max_out=16.6, min_out=-16.6)
        self.state = self.INITIAL
        self.distribute = True

        # Position index
        self.index = None

    def neighbors_distance(self):
        """
        Check if there is a robot ahead using the infrared sensor 2 (front-front).
        Check if there is a robot ahead using the infrared sensor 5 (back-left) and 6 (back-right).
        :return back, front: response values of the rear and front sensors
        """
        front = self.prox_values[2]
        back = np.mean(np.array([self.prox_values[5], self.prox_values[6]]))

        return back, front

    def compute_difference(self):
        """
        :return: the difference between the response value of front and the rear sensor
        """
        back, front = self.neighbors_distance()

        # Apply a small correction to the distance measured by the rear sensors: the front sensor used is at a
        # different x coordinate from the point to which the rear sensor of the robot that follows points. this is
        # because of the curved shape of the face of the Thymio
        delta_x = 7.41150769
        x = 7.95

        # Maximum possible response values
        delta_x_m = 4505 * delta_x / 14
        x_m = 4505 * x / 14

        correction = x_m - delta_x_m

        return front - correction - back

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots not to the end of the line using the distributed controller, setting the target {left,
        right} wheel speed each at the same value in order to moves the robot straight ahead. This distributed
        controller is a simple proportional controller PID(-0.01, 0, 0, max_out=16.6, min_out=-16.6) that takes in
        input the difference between the response value of front and the rear sensor. To the distance measured by the
        rear sensors is applied a small correction since the front sensor used is at a different x coordinate from
        the point to which the rear sensor of the robot that follows points. This is because of the curved shape of
        the face of the Thymio.
        The response values are actually intensities: the front correspond to the frontal center sensor and the back
        to the mean of the response values of the rear sensors.
        The final difference is computed ad follow: out = front - correction - back
        The speed are clipped to [min_out=-16.6, max_out=16.6].
        Then, using communication, colour the robot half and half. They should understand on which side they are
        compared to the mid of the line
        :param dt: control step duration
        """
        back, front = self.neighbors_distance()

        if self.state == self.INITIAL:
            if back == 0:
                self.index = 0
                self.prox_comm_tx = self.index
                self.prox_comm_enable = True
                self.set_led_top(blue=1.0)
            elif front == 0:
                self.set_led_top(green=1.0)

            self.state = self.COLOURING

        elif self.state == self.COLOURING:
            if len(self.prox_comm_events) > 0:
                if back == 0:
                    if self.index != 0:
                        self.index = 0
                        self.prox_comm_tx = self.index
                        self.prox_comm_enable = True
                else:
                    if self.prox_comm_events[0].intensities[5] != 0 or self.prox_comm_events[0].intensities[6] != 0:
                        received_idx = self.prox_comm_events[0].rx
                        self.index = received_idx + 1

                        self.prox_comm_tx = self.index
                        self.prox_comm_enable = True

            # Color the robot half and half they should understand on which side they are compared to the mid of the
            # line
            if self.index is not None:
                if self.index > (self.myt_quantity / 2) - 1:
                    self.set_led_top(green=1.0)
                else:
                    self.set_led_top(blue=1.0)

            # Check which are the first and last robots in the line and don't move them
            if not (front == 0 or back == 0):
                speed = self.distributed_controller.step(self.compute_difference(), dt)

                self.motor_left_target = speed
                self.motor_right_target = speed


def setup(aseba: bool = False) -> pyenki.World:
    """
    Set up the world as an unbounded world. Create multiple Thymios and position them such as all x-axes are aligned.
    The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range.

    :param aseba
    :return world
    """
    # Create an unbounded world
    world = pyenki.World()

    myt_quantity = 8
    myts = [DistributedThymio2(myt_quantity=myt_quantity, name='myt%d' % (i + 1), use_aseba_units=aseba) for i in range(
        myt_quantity)]

    # ~ 14 cm is the proximity sensors maximal range
    distances = np.random.randint(5, 8, 8)

    # Distance between the origin and the front of the robot along x-axis
    constant = 7.95

    for i, myt in enumerate(myts):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = myts[i - 1].position[0]

        myt.position = (prev_pos + float(distances[i]) + constant, 0)
        myt.angle = 0
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
