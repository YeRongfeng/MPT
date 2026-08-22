#ifndef T_HYBRID_OCCUPANCY_GRID_STUB_H
#define T_HYBRID_OCCUPANCY_GRID_STUB_H

#include <cstdint>
#include <memory>
#include <vector>

namespace nav_msgs {

struct MapMetaData {
  unsigned int width = 0;
  unsigned int height = 0;
  double resolution = 0.2;
};

struct OccupancyGrid {
  MapMetaData info;
  std::vector<int8_t> data;
  using Ptr = std::shared_ptr<OccupancyGrid>;
};

}  // namespace nav_msgs

#endif
