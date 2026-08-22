#ifndef MYPCL_H
#define MYPCL_H

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "constants.h"

namespace HybridAStar {
namespace MyPcl {

// Copied from upstream T-Hybrid mypcl.h::Car_Angle. Projects a terrain
// normal into vehicle pitch/roll given yaw. No PCL needed at search time
// because terrain voxels are precomputed.
static inline Eigen::Vector3d Car_Angle(Eigen::Vector3d Point_nomal, float t_in) {
  Eigen::Vector3d eulerAngle;
  Eigen::Vector3d x_1, y_1, x_2, y_2, z_2, fenzi;
  Eigen::Matrix3d rotation_matrix3d;
  int normal_dir, normal_size;

  x_1 << cos(t_in), sin(t_in), 0;
  y_1 << -sin(t_in), cos(t_in), 0;
  normal_size = x_1.dot(Point_nomal);
  if (normal_size > 0) {
    normal_dir = -1;
  } else {
    normal_dir = 1;
  }
  z_2 = Point_nomal;
  fenzi = y_1.cross(z_2);
  x_2 = fenzi / fenzi.norm();
  y_2 = z_2.cross(x_2);
  rotation_matrix3d << x_2, y_2, z_2;
  eulerAngle = rotation_matrix3d.eulerAngles(2, 1, 0);
  double a = eulerAngle(1);
  double b = eulerAngle(2);
  if (std::abs(a) > 1.57) {
    eulerAngle(1) = a / std::abs(a) * (3.1415 - std::abs(a));
  }
  if (std::abs(b) > 1.57) {
    eulerAngle(2) = b / std::abs(b) * (3.1415 - std::abs(b));
  }
  return normal_dir * eulerAngle;
}

}  // namespace MyPcl
}  // namespace HybridAStar

#endif
