#ifndef CONSTANTS
#define CONSTANTS
/*!
   Adapted T-Hybrid constants for Path MeanFlow maps.

   Original T-Hybrid (Liu et al., IROS 2023) used gridSize=0.3 m, mapLength=27 m,
   vehicle 0.5 x 0.5 m, and r=6 (treated as cell units by the motion primitives
   and OMPL heuristic, i.e. 1.8 m world radius at 0.3 m cells).

   This overlay keeps the same search / traversability code and only changes
   the numbers that must match dataset1:
     - 20 m map, 0.2 m cells
     - min turning radius 1/2.1 m in *cell* units (same convention as upstream)
     - 0.4 x 0.4 m footprint (our 0.2 m vehicle radius)
*/
#include <cmath>

namespace HybridAStar {
namespace Constants {

static const float gridSize = 0.2f;
static const float mapLength = 20.f;
static const float traverPenality = 4.0f;
static const float TurnPenality = 1.05f;
static const int volnum = (int)round(mapLength / gridSize);
static const float thetaYmin = -0.3f;
static const float thetaYmax = 0.25f;
static const float thetaXmax = 0.18f;
static const float pitchWeight = 0.4f;
static const float rollWeight = 0.4f;
static const float roughWeight = 0.3f;

static const bool coutDEBUG = false;
static const bool manual = true;
static const bool visualization = false && manual;
static const bool visualization2D = false && manual;
static const bool reverse = true;
static const bool dubinsShot = true;
static const bool dubins = false;
static const bool dubinsLookup = false && dubins;
static const bool twoD = true;

static const int iterations = 200000;
static const double bloating = 0;
static const double width = 0.4 + 2 * bloating;
static const double length = 0.4 + 2 * bloating;
// Cell units. World radius = r * cellSize = 0.47619 m = 1 / 2.1.
static const float r = 2.380952f;
static const int headings = 72;
static const float deltaHeadingDeg = 360 / (float)headings;
static const float deltaHeadingRad = 2 * M_PI / (float)headings;
static const float deltaHeadingNegRad = 2 * M_PI - deltaHeadingRad;
static const double squarLength = 1.4142;
static const float cellSize = gridSize;
static const float tieBreaker = 0.01f;

static const float factor2D = sqrt(5) / sqrt(2) + 1;
static const float penaltyTurning = TurnPenality;
static const float penaltyReversing = 2.0f;
static const float penaltyCOD = 2.0f;
static const float penalityTraver = traverPenality;
static const float dubinsShotDistance = 1.0f;
static const float dubinsStepSize = 1;

static const int dubinsWidth = 15;
static const int dubinsArea = dubinsWidth * dubinsWidth;

static const int bbSize = std::ceil((sqrt(width * width + length * length) + 4) / cellSize);
static const int positionResolution = 10;
static const int positions = positionResolution * positionResolution;

struct relPos {
  int x;
  int y;
};
struct config {
  int length;
  relPos pos[64];
};

static const float minRoadWidth = 2;

struct color {
  float red;
  float green;
  float blue;
};
static constexpr color teal = {102.f / 255.f, 217.f / 255.f, 239.f / 255.f};
static constexpr color green = {166.f / 255.f, 226.f / 255.f, 46.f / 255.f};
static constexpr color orange = {253.f / 255.f, 151.f / 255.f, 31.f / 255.f};
static constexpr color pink = {249.f / 255.f, 38.f / 255.f, 114.f / 255.f};
static constexpr color purple = {174.f / 255.f, 129.f / 255.f, 255.f / 255.f};

}
}

#endif
