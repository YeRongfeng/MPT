#ifndef VISUALIZE_H
#define VISUALIZE_H

#include "node2d.h"
#include "node3d.h"

namespace HybridAStar {

class Visualize {
 public:
  Visualize() {}
  void clear() {}
  void clear2D() {}
  void publishNode3DPose(Node3D&) {}
  void publishNode3DPoses(Node3D&) {}
  void publishNode3DCosts(Node3D*, int, int, int) {}
  void publishNode2DPose(Node2D&) {}
  void publishNode2DPoses(Node2D&) {}
  void publishNode2DCosts(Node2D*, int, int) {}
};

}  // namespace HybridAStar

#endif
