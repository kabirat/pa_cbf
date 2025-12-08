
#!/usr/bin/env python
import rospy
import os
import subprocess
import time

# Default ROS master port
port = "11311"
try:
		subprocess.Popen(["roscore", "-p", port])
		print("Roscore launched!")
		# give roscore a moment to start
		time.sleep(1.0)
except Exception as e:
		print("Failed to launch roscore:", e)

# Launch the simulation with the given launchfile name (private param ~launchfile)
rospy.init_node("gym", anonymous=True)
import rospkg
rospack = rospkg.RosPack()
default_launchfile = "/home/mien-group/catkin_ws/src/pa_cbf/launch/robot_assests.launch"
launchfile = rospy.get_param("~launchfile", default_launchfile)
print("Using launchfile: {}".format(launchfile))

try:
	subprocess.Popen(["roslaunch", "-p", port, launchfile])
	print("Roslaunch started for: {}".format(launchfile))
except Exception as e:
	print("Failed to start roslaunch for {}: {}".format(launchfile, e))
