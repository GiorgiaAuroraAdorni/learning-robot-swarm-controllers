# pid.py
# © 2020 Alessandro Giusti
# © 2020 Giorgia Adorni
# Adapted from https://github.com/alessandro-giusti/teaching-notebooks/blob/master/robotics/04%20closedloopcontrol.ipynb


class PID:
    """
    A PID controller that implements the following formula.

    .. math::
        K_p \epsilon + K_d \\frac {d \epsilon}{d t} + K_i \int{\epsilon(t) dt}

    :param Kp: apply a correction based on a proportional term
    :param Ki: apply a correction based on a integral term
    :param Kd: apply a correction based on a derivative term
    :param min_out=-float("inf"): clip the speed to the minimum allowed velocity
    :param max_out=float("inf")): clip the speed to the maximum allowed velocity

    """
    def __init__(self, Kp, Ki, Kd, min_out=-float("inf"), max_out=float("inf")):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.min_out = min_out
        self.max_out = max_out

        self.last_e = None
        self.sum_e = 0

    def step(self, e, dt):
        """
        :param e: error
        :param dt: should be the time elapsed from the last time step was called
        :return output: speed
        """

        if self.last_e is not None:
            derivative = (e - self.last_e) / dt
        else:
            derivative = 0

        self.last_e = e
        self.sum_e += e * dt

        output = self.Kp * e + self.Kd * derivative + self.Ki * self.sum_e
        output = min(max(self.min_out, output), self.max_out)

        return output
