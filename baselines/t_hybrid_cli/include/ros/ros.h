#ifndef T_HYBRID_ROS_STUB_H
#define T_HYBRID_ROS_STUB_H

#include <string>

// Minimal ROS stubs so the diagnostic CLI can still compile. The paper
// baseline is the official ROS node, not this stub.

namespace ros {
struct Duration {
  explicit Duration(double = 0) {}
  void sleep() const {}
};
namespace param {
inline bool get(const std::string&, std::string&) { return false; }
}  // namespace param
}  // namespace ros

#endif
