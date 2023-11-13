#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self, kp, kd, ki):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 0)

        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.lookahead_dist = 1

        self.integral = 0
        self.prev_error = 0
        self.error = 0
        self.current_reading_time = self.get_clock().now()

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
        index = int((angle - self.angle_min) / self.angle_increment)
        return (range_data[index], self.angle_min + index * self.angle_increment)

    def get_error(self, range_data: [float], dist: float):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        a, angle_a = self.get_range(range_data, np.pi/4)
        b, angle_b = self.get_range(range_data, np.pi/2)
        theta = angle_a - angle_b
        
        alpha = np.arctan2((a * np.cos(theta) - b), (a * np.sin(theta)))
        curr_dist = b * np.cos(alpha)
        lookahead_dist = curr_dist + self.lookahead_dist * np.sin(alpha)
        return -lookahead_dist - dist

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
        self.current_reading_time = self.get_clock().now()
        dt = (self.current_reading_time - previous_reading_time).nanoseconds / 1e9

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        angle = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error

        velocity = self.get_velocity(angle)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle

        self.get_logger().info('angle: %f' % angle)

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

        error = self.get_error(msg.ranges, 0.78)
        self.get_logger().info('error: %f' % error)
        self.pid_control(error) # TODO: actuate the car with PID
        # exit()


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