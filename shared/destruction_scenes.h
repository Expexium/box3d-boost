// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "box3d/destruction.h"
#include "box3d/id.h"
#include "box3d/math_functions.h"

#include <stdbool.h>
#include <stdint.h>

// Procedural destructible structures shared by the unit tests, the benchmark app, and the samples.
//
// A real game cooks its structures from authored fracture geometry. These helpers stand in for that:
// they cut boxes into convex fragments with planes, which gives neighbors exactly coplanar faces just
// like a fracture tool does, and hand the fragments to b3CreateStructure.

typedef struct b3Capacity b3Capacity;
typedef struct b3HullData b3HullData;

#ifdef __cplusplus
extern "C"
{
#endif

// Collects fragments and cooks them into a structure.
typedef struct DestructionBuilder DestructionBuilder;

DestructionBuilder* CreateDestructionBuilder( uint32_t seed );
void DestroyDestructionBuilder( DestructionBuilder* builder );

// Add a material and get its index. Material 0 is the default material unless one is added.
int AddDestructionMaterial( DestructionBuilder* builder, const b3StructureMaterial* material );

// How a box is cut into fragments
typedef struct FractureDef
{
	// Target size of a support fragment
	float cellSize;

	// Target size of the debris a support fragment crumbles into. Zero for no debris level.
	float childSize;

	// How far a cut may sit from the middle of a fragment, in [0, 1)
	float jitter;

	// How far a cut may lean away from the box axes. Zero gives plain boxes, which cooks much faster.
	float tilt;

	int materialIndex;

	// Fix the fragments to the world
	bool isAnchored;

	// The cluster the fragments belong to on each cluster layer. See b3FragmentDef.
	int clusters[B3_MAX_CLUSTER_LAYERS];
} FractureDef;

FractureDef DefaultFractureDef( void );

// Add a box cut into fragments. The transform places the box in structure space.
void AddFracturedBox( DestructionBuilder* builder, b3Transform transform, b3Vec3 halfExtents, const FractureDef* def );

// Add a single convex fragment. The hull is cloned. Returns the fragment index.
int AddFragment( DestructionBuilder* builder, const b3HullData* hull, int parentIndex, int materialIndex, bool isAnchored );

int GetFragmentCount( const DestructionBuilder* builder );

// Is this a support fragment, as opposed to the debris one crumbles into?
bool IsSupportFragment( const DestructionBuilder* builder, int fragmentIndex );

// Center of a fragment in structure space
b3Vec3 GetFragmentCenter( const DestructionBuilder* builder, int fragmentIndex );

// Put a fragment into a cluster on one cluster layer
void SetFragmentCluster( DestructionBuilder* builder, int fragmentIndex, int layer, int cluster );

// Put the support fragments added since firstFragment into clusters on one cluster layer, by the
// cell of a grid their center lies in. A cell size of zero along an axis makes one cell of that axis.
// Cuts a wall into panels or a tower into segments.
void ClusterFragmentsByGrid( DestructionBuilder* builder, int firstFragment, int layer, b3Vec3 cellSize );

// Set the number of cluster layers and the strength of the connections on each strength layer.
// The array holds two more entries than there are cluster layers. See b3StructureDef::layerStrength.
void SetDestructionLayers( DestructionBuilder* builder, int clusterLayerCount, const float* layerStrength );

// Cook everything added so far. The caller owns the result and frees it with b3DestroyStructure.
b3StructureData* BuildStructure( DestructionBuilder* builder );

// Split a convex hull with a plane. The caller owns the two parts. Returns false if the plane does
// not cut the hull into two solid parts, in which case nothing is created.
bool SplitHull( const b3HullData* hull, b3Plane plane, b3HullData** below, b3HullData** above );

// A multi-story building: a grid of columns carrying floor slabs, with optional walls.
// The structure origin is at ground level in the middle of the footprint and y is up.
typedef struct BuildingDef
{
	int floorCount;
	int bayCountX;
	int bayCountZ;
	float baySize;
	float floorHeight;
	float columnSize;
	float slabThickness;
	float wallThickness;

	// Outer walls with door and window openings
	bool enableWalls;

	FractureDef fracture;
	b3StructureMaterial material;

	// Strength of the connections between stories, between the members of a story (a column, a
	// wall panel, a bay of the slab), between the fragments of a member, and inside a fragment
	float layerStrength[B3_MAX_STRENGTH_LAYERS];
	uint32_t seed;
} BuildingDef;

BuildingDef DefaultBuildingDef( void );
b3StructureData* CreateBuildingStructure( const BuildingDef* def );

// A free standing wall in the x-y plane, standing on y = 0.
b3StructureData* CreateWallStructure( float width, float height, float thickness, const FractureDef* fracture, uint32_t seed );

// A beam along the x-axis starting at the origin. Fragment 0 is the first cell at the origin end.
b3StructureData* CreateBeamStructure( float length, float height, float depth, int cellCount, bool anchorFirst,
									  const b3StructureMaterial* material );

// A tower standing on y = 0.
b3StructureData* CreateTowerStructure( float width, float height, const FractureDef* fracture, uint32_t seed );

// A boulder: one support fragment that crumbles into debris.
b3StructureData* CreateBoulderStructure( float radius, float childSize, const b3StructureMaterial* material, uint32_t seed );

// Blast damage plus a push, the usual pairing for an explosion.
void ApplyBlast( b3WorldId worldId, b3Pos position, float radius, float damage, float impulsePerArea );

// The building collapse benchmark: several buildings on a ground plate. After a short rest charges
// take out the columns of the ground floor and of every third floor above it, line by line and
// floor by floor, and the buildings come down in turn.
void GetBuildingCollapseCapacity( b3Capacity* capacity );
void CreateBuildingCollapse( b3WorldId worldId );
void StepBuildingCollapse( b3WorldId worldId, int stepIndex );
void DestroyBuildingCollapse( void );

// The destructible of a building of the benchmark scene, or a null id.
b3DestructibleId GetBuildingCollapseBuilding( int index );

// Scale of the benchmark scene. Set before CreateBuildingCollapse.
void SetBuildingCollapseScale( int buildingCount, int floorCount );

#ifdef __cplusplus
}
#endif
