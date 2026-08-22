#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "algorithm.h"
#include "collisiondetection.h"
#include "constants.h"
#include "helper.h"
#include "nav_msgs/OccupancyGrid.h"
#include "node2d.h"
#include "node3d.h"
#include "visualize.h"

using HybridAStar::Algorithm;
using HybridAStar::CollisionDetection;
using HybridAStar::Node2D;
using HybridAStar::Node3D;
using HybridAStar::Visualize;
namespace C = HybridAStar::Constants;
namespace Helper = HybridAStar::Helper;

static bool load_occupancy(const std::string& path, nav_msgs::OccupancyGrid::Ptr grid) {
  std::ifstream in(path.c_str());
  if (!in) {
    std::cerr << "cannot read occupancy " << path << "\n";
    return false;
  }
  int width = 0;
  int height = 0;
  in >> width >> height;
  if (width <= 0 || height <= 0) {
    std::cerr << "bad occupancy size\n";
    return false;
  }
  grid->info.width = static_cast<unsigned int>(width);
  grid->info.height = static_cast<unsigned int>(height);
  grid->info.resolution = C::cellSize;
  grid->data.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int value = 0;
      in >> value;
      grid->data[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
          value ? 100 : 0;
    }
  }
  return static_cast<bool>(in);
}

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cerr << "usage: t_hybrid_plan occupancy.txt terrain.txt sx sy st gx gy gt\n";
    return 2;
  }
  const std::string occ_path = argv[1];
  const std::string terrain_path = argv[2];
  const float sx = std::stof(argv[3]);
  const float sy = std::stof(argv[4]);
  const float st = std::stof(argv[5]);
  const float gx = std::stof(argv[6]);
  const float gy = std::stof(argv[7]);
  const float gt = std::stof(argv[8]);

  setenv("T_HYBRID_TERRAIN", terrain_path.c_str(), 1);

  auto grid = std::make_shared<nav_msgs::OccupancyGrid>();
  if (!load_occupancy(occ_path, grid)) {
    return 2;
  }

  CollisionDetection configurationSpace;
  configurationSpace.updateGrid(grid);
  Visualize visualization;

  const int width = static_cast<int>(grid->info.width);
  const int height = static_cast<int>(grid->info.height);
  const int depth = C::headings;
  Node3D* nodes3D = new Node3D[width * height * depth]();
  Node2D* nodes2D = new Node2D[width * height]();
  float* dubinsLookup = nullptr;

  Node3D nStart(
      sx / C::cellSize,
      sy / C::cellSize,
      Helper::normalizeHeadingRad(st),
      0,
      0,
      nullptr);
  const Node3D nGoal(
      gx / C::cellSize,
      gy / C::cellSize,
      Helper::normalizeHeadingRad(gt),
      0,
      0,
      nullptr);
  const int start_xi = static_cast<int>(nStart.getX());
  const int start_yi = static_cast<int>(nStart.getY());
  int8_t start_occ = 0;
  if (start_xi >= 0 && start_yi >= 0 && start_xi < width && start_yi < height) {
    start_occ = grid->data[static_cast<size_t>(start_yi) * static_cast<size_t>(width) +
                           static_cast<size_t>(start_xi)];
  }
  std::cerr << "grid " << width << "x" << height << " start_cell=" << nStart.getX() << ","
            << nStart.getY() << " occ=" << static_cast<int>(start_occ)
            << " trav=" << configurationSpace.isTraversable(&nStart) << "\n";

  const auto t0 = std::chrono::steady_clock::now();
  Node3D* solution = Algorithm::hybridAStar(
      nStart,
      nGoal,
      nodes3D,
      nodes2D,
      width,
      height,
      configurationSpace,
      dubinsLookup,
      visualization);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (solution == nullptr || !solution->isNearGoal(nGoal)) {
    std::cout << "found 0\n";
    std::cerr << "planning_time_ms=" << ms << " failure=no_path\n";
    delete[] nodes3D;
    delete[] nodes2D;
    return 1;
  }

  std::vector<Node3D> path;
  const Node3D* node = solution;
  int guard = 0;
  while (node != nullptr && guard++ < 200000) {
    path.push_back(*node);
    node = node->getPred();
  }
  std::reverse(path.begin(), path.end());

  std::cout << "found " << path.size() << "\n";
  for (const Node3D& p : path) {
    std::cout << p.getX() * C::cellSize << " " << p.getY() * C::cellSize << " " << p.getT()
              << "\n";
  }
  std::cerr << "planning_time_ms=" << ms << "\n";

  delete[] nodes3D;
  delete[] nodes2D;
  return 0;
}
