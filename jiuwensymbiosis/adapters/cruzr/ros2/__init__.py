# coding: utf-8
"""Cruzr's ROS 2 subprocess workers (waist RGBD frame grab + differential base drive).

Both run under ``/usr/bin/python3`` (conda 3.11 cannot import rclpy), so they are
executed by file path, never imported into the agent process. They speak Cruzr's
own topics and message types — a different ROS 2 body writes its own pair. The
body-agnostic half (``sensor_msgs/Image`` → numpy) stays in ``jiuwensymbiosis.ros2``.
"""
