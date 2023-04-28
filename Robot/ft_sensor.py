import rospy
from geometry_msgs.msg import WrenchStamped
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import math


TIME = 'Time'
FX = "Fx"
FY = "Fy"
FZ = "Fz"
EQUIVALENT_FORCE = "F"
MX = "Mx"
MY = "My"
MZ = "Mz"
EQUIVALENT_TORQUE = "M"

def get_vector_magnitude(vector):
    return math.sqrt(pow(vector.x, 2) + pow(vector.y, 2) + pow(vector.z, 2))


class FTSensor:
    def __init__(self, should_plot=True) -> None:
        self.data = None
        self._wrench_topic = '/bus0/ft_sensor0/ft_sensor_readings/wrench'
        self._subscriber = rospy.Subscriber(self._wrench_topic, WrenchStamped, self._callback)

        # used for ploting.. to have timestamp data
        self.plot_data = None
        self._data_buffer = pd.DataFrame(columns=[TIME, FX, FY, FZ, EQUIVALENT_FORCE, MX, MY, MZ, EQUIVALENT_TORQUE])
        self._start_time = rospy.get_time()
        self._should_plot = should_plot
        if self._should_plot:
            self._run_graph()
        rospy.on_shutdown(self._clean_shutdown)

    def _callback(self, sensor_data):
        force = sensor_data.wrench.force
        torque = sensor_data.wrench.torque
        self.data = [force.x, force.y, force.z, torque.x, torque.y, torque.z]

        if self._should_plot:
            latest_data = [rospy.get_time() - self._start_time,
                                sensor_data.wrench.force.x,
                                sensor_data.wrench.force.y,
                                sensor_data.wrench.force.z,
                                get_vector_magnitude(sensor_data.wrench.force),
                                sensor_data.wrench.torque.x,
                                sensor_data.wrench.torque.y,
                                sensor_data.wrench.torque.z,
                                get_vector_magnitude(sensor_data.wrench.torque)
                                ]

            idx_labels = [TIME, FX, FY, FZ, EQUIVALENT_FORCE, MX, MY, MZ, EQUIVALENT_TORQUE]
            last_reading = pd.Series(latest_data, idx_labels)
            self._data_buffer = pd.concat([self._data_buffer, last_reading.to_frame().T], ignore_index=True)
            
    def _clean_shutdown(self):
        pass

    def get_data(self):
        return self.data

    def _run_graph(self):
        fig, (f_ax, m_ax) = plt.subplots(2, 1, sharex = True)
        f_lines = f_ax.plot([0], [0], 'r-',[0], [0], 'g-', [0], [0], 'b-', [0], [0], 'k-') 
        f_ax.legend(f_lines,(FX, FY, FZ, EQUIVALENT_FORCE))
        m_lines = m_ax.plot([0], [0], 'r-', [0], [0], 'g-', [0], [0], 'b-', [0], [0], 'k-') 
        m_ax.legend(m_lines,(MX, MY, MZ, EQUIVALENT_TORQUE))

        rospy.loginfo("Setting up animated graph")
        # Add labels
        fig.canvas.set_window_title(self._wrench_topic)

        subtitle = rospy.get_param('~plot_title', self._wrench_topic)
        fig.suptitle(subtitle, fontsize="x-large")
        
        f_ax.set_title("Force Feedback")
        f_ax.grid()
        m_ax.set_title("Torque Feedback")
        m_ax.grid()
        f_ax.set(ylabel="Force [N]", xlabel="Time [s]")
        m_ax.set(ylabel="Torque [Nm]", xlabel="Time [s]")
        f_ax.set_xlim(0, 25)
        f_ax.set_ylim(-150, 150)
        m_ax.set_xlim(0, 25)
        m_ax.set_ylim(-15, 15)

        def update_plot(data):
            if data is None:
                return []
            last_timestamp = self._data_buffer[TIME].iloc[-1] if self._data_buffer[TIME].size > 0 else 0 
            # Update time axis 
            xmin, xmax = f_ax.get_xlim()
            if xmax > last_timestamp - 5 and (xmax < last_timestamp or xmax > last_timestamp + 20):
                f_ax.set_xlim(xmin, last_timestamp + 10)
                m_ax.set_xlim(xmin, last_timestamp + 10)
                fig.canvas.draw()

            # Update for lines for forces
            # try: 
            f_lines[0].set_data(self._data_buffer[TIME], self._data_buffer[FX])
            f_lines[1].set_data(self._data_buffer[TIME], self._data_buffer[FY])
            f_lines[2].set_data(self._data_buffer[TIME], self._data_buffer[FZ])
            f_lines[3].set_data(self._data_buffer[TIME], self._data_buffer[EQUIVALENT_FORCE])
            # Update for lines for torques 
            m_lines[0].set_data(self._data_buffer[TIME], self._data_buffer[MX])
            m_lines[1].set_data(self._data_buffer[TIME], self._data_buffer[MY])
            m_lines[2].set_data(self._data_buffer[TIME], self._data_buffer[MZ])
            m_lines[3].set_data(self._data_buffer[TIME], self._data_buffer[EQUIVALENT_TORQUE])
            # except:
                # return []
            return f_lines + m_lines

        def get_plot_data():
            # rospy.loginfo("data update")
            # rospy.loginfo("Time buffer: " + str( time_buffer))
            yield self._data_buffer
        
        plot_animation = animation.FuncAnimation(fig, 
                                update_plot, 
                                get_plot_data, 
                                blit=True, 
                                interval=1000)
        plt.show()


def main():
    rospy.init_node('FTSensor_node')
    ftsensor = FTSensor()
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        print(ftsensor.data)
        rate.sleep()

if __name__ == "__main__":
    # main()
    rospy.init_node('FTSensor_node')
    ftsensor = FTSensor()
    
