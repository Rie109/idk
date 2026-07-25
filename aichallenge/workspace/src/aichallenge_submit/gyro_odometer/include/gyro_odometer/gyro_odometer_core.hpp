// Copyright 2015-2019 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GYRO_ODOMETER__GYRO_ODOMETER_CORE_HPP_
#define GYRO_ODOMETER__GYRO_ODOMETER_CORE_HPP_

#include "tier4_autoware_utils/ros/transform_listener.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <tf2/transform_datatypes.h>
#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <deque>
#include <memory>
#include <string>

// ---------- Lightweight 1-D Kalman filter ----------
struct SimpleKalman1D {
  double x = 0.0;
  double P = 1.0;
  double Q = 0.04;
  double R = 0.5;
  bool initialized = false;

  double update(double z, double dt = 1.0) {
    if (!initialized) {
      x = z;
      initialized = true;
      return x;
    }
    // Predict — scale process noise by elapsed time so irregular
    // publish cadence doesn't silently distort the filter's trust balance
    P += Q * dt;
    // Update
    double K = P / (P + R);
    x += K * (z - x);
    P *= (1.0 - K);
    return x;
  }
};


class GyroOdometer : public rclcpp::Node {
private:
  using COV_IDX = tier4_autoware_utils::xyz_covariance_index::XYZ_COV_IDX;

public:
  explicit GyroOdometer(const rclcpp::NodeOptions &options);
  ~GyroOdometer();

private:
  void callbackVehicleTwist(
      const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr
          vehicle_twist_msg_ptr);
  void callbackImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr);
  void publishData(
      const geometry_msgs::msg::TwistWithCovarianceStamped &twist_with_cov_raw);

  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::
      SharedPtr vehicle_twist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_raw_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
      twist_with_covariance_raw_pub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
      twist_with_covariance_pub_;

  std::shared_ptr<tier4_autoware_utils::TransformListener> transform_listener_;

  std::string output_frame_;
  double message_timeout_sec_;

  bool vehicle_twist_arrived_;
  bool imu_arrived_;
  std::deque<geometry_msgs::msg::TwistWithCovarianceStamped>
      vehicle_twist_queue_;
  std::deque<sensor_msgs::msg::Imu> gyro_queue_;

  // Kalman filters for velocity and yaw rate
  SimpleKalman1D kalman_vx_;
  SimpleKalman1D kalman_wz_;

  rclcpp::Time last_filter_update_time_;
  bool filter_time_initialized_ = false;
};

#endif // GYRO_ODOMETER__GYRO_ODOMETER_CORE_HPP_
