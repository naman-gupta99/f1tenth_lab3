#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from math import isfinite
import time
import atexit

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self, kp, kd, ki):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        initialpose_topic = '/initialpose'

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.initalpose_pub = self.create_publisher(AckermannDriveStamped, initialpose_topic, 10)

        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.lookahead_dist = 1.0

        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0
        self.current_reading_time = time.time()

        self.angle_min = 0
        self.angle_increment = 0

    def get_range(self, range_data: [float], angle: float):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """
        index = int(np.floor((angle - self.angle_min) / self.angle_increment))
        return range_data[index]
    def get_error(self, range_data: [float], dist: float):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        angle_b = np.pi/2
        angle_a = np.pi/4
        while (not isfinite(self.get_range(range_data, angle_a))) or (angle_a == angle_b):
            print('skipping a')
            angle_a += 0.02

        while (not isfinite(self.get_range(range_data, angle_b))) or (angle_a == angle_b):
            print('skipping b')
            angle_b += 0.02

        a = self.get_range(range_data, angle_a)
        b = self.get_range(range_data, angle_b)
        theta = angle_b - angle_a
        
        alpha = np.arctan2((a * np.cos(theta) - b), (a * np.sin(theta)))
        curr_dist = b * np.cos(alpha)
        lookahead_dist = curr_dist + self.lookahead_dist * np.sin(alpha)
        return dist - lookahead_dist

    def get_velocity(self, angle: float):
        if angle < np.pi/18:
            return 1.5
        if angle < np.pi/9:
            return 1.0
        return 0.5

    def pid_control(self, error):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        angle = 0.0
        
        previous_reading_time = self.current_reading_time
        self.current_reading_time = time.time()
        dt = (self.current_reading_time - previous_reading_time)

        self.integral = self.prev_error * dt
        derivative = (error - self.prev_error) / dt

        angle = -(self.kp * error + self.ki * self.integral + self.kd * derivative)

        self.prev_error = error

        if abs(error) < 0.1:
            angle = 0.0

        velocity = self.get_velocity(abs(angle))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle

        self.drive_pub.publish(drive_msg)
        

    def scan_callback(self, msg: LaserScan):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """

        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

        error = self.get_error(msg.ranges, 1.0)
        self.pid_control(error)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow(float(input("kp: ")), float(input("kd: ")), float(input("ki: ")))
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()