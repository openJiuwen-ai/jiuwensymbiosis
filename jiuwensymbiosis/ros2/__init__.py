# coding: utf-8
"""Body-agnostic ROS 2 transport helpers.

Only what every ROS 2 body needs: ``sensor_msgs/Image`` → numpy decoding. The
topic names, message types and control loops of a particular robot belong to its
adapter (see ``adapters/cruzr/ros2/`` for the reference pair).
"""
