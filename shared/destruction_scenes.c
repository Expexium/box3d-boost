// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

#include "destruction_scenes.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/constants.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#ifdef NDEBUG
#define BENCHMARK_DEBUG 0
#else
#define BENCHMARK_DEBUG 1
#endif

enum
{
	e_maxMaterials = 16,
	e_maxFractureDepth = 16,
};

typedef struct BuilderFragment
{
	b3HullData* hull;
	b3Transform transform;
	int parentIndex;
	int materialIndex;
	bool isAnchored;
} BuilderFragment;

struct DestructionBuilder
{
	BuilderFragment* fragments;
	int count;
	int capacity;
	b3StructureMaterial materials[e_maxMaterials];
	int materialCount;
	uint32_t seed;
};

// Random number in [0, 1). Local to the builder so scenes do not disturb each other.
static float BuilderRandom( DestructionBuilder* builder )
{
	uint32_t x = builder->seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	builder->seed = x;
	return (float)( x >> 8 ) * ( 1.0f / 16777216.0f );
}

DestructionBuilder* CreateDestructionBuilder( uint32_t seed )
{
	DestructionBuilder* builder = calloc( 1, sizeof( DestructionBuilder ) );
	builder->seed = seed != 0 ? seed : 1;
	return builder;
}

void DestroyDestructionBuilder( DestructionBuilder* builder )
{
	if ( builder == NULL )
	{
		return;
	}

	for ( int i = 0; i < builder->count; ++i )
	{
		b3DestroyHull( builder->fragments[i].hull );
	}

	free( builder->fragments );
	free( builder );
}

int AddDestructionMaterial( DestructionBuilder* builder, const b3StructureMaterial* material )
{
	assert( builder->materialCount < e_maxMaterials );
	builder->materials[builder->materialCount] = *material;
	builder->materialCount += 1;
	return builder->materialCount - 1;
}

FractureDef DefaultFractureDef( void )
{
	FractureDef def = { 0 };
	def.cellSize = 1.0f;
	def.childSize = 0.0f;
	def.jitter = 0.3f;
	def.tilt = 0.0f;
	def.materialIndex = 0;
	def.isAnchored = false;
	return def;
}

// Takes ownership of the hull
static int AddOwnedFragment( DestructionBuilder* builder, b3HullData* hull, b3Transform transform, int parentIndex, int materialIndex,
							 bool isAnchored )
{
	if ( builder->count == builder->capacity )
	{
		builder->capacity = builder->capacity == 0 ? 256 : 2 * builder->capacity;
		builder->fragments = realloc( builder->fragments, builder->capacity * sizeof( BuilderFragment ) );
	}

	BuilderFragment* fragment = builder->fragments + builder->count;
	fragment->hull = hull;
	fragment->transform = transform;
	fragment->parentIndex = parentIndex;
	fragment->materialIndex = materialIndex;
	fragment->isAnchored = isAnchored;
	builder->count += 1;
	return builder->count - 1;
}

int AddFragment( DestructionBuilder* builder, const b3HullData* hull, int parentIndex, int materialIndex, bool isAnchored )
{
	return AddOwnedFragment( builder, b3CloneHull( hull ), b3Transform_identity, parentIndex, materialIndex, isAnchored );
}

int GetFragmentCount( const DestructionBuilder* builder )
{
	return builder->count;
}

bool SplitHull( const b3HullData* hull, b3Plane plane, b3HullData** below, b3HullData** above )
{
	*below = NULL;
	*above = NULL;

	const b3Vec3* points = b3GetHullPoints( hull );
	const b3HullHalfEdge* edges = b3GetHullEdges( hull );
	int vertexCount = hull->vertexCount;

	// Points this close to the plane go to both sides, so the two parts share the cut exactly.
	float extent = b3Length( b3Sub( hull->aabb.upperBound, hull->aabb.lowerBound ) );
	float tolerance = 1.0e-5f * extent;

	float distances[B3_MAX_HULL_VERTICES];
	b3Vec3 lower[2 * B3_MAX_HULL_VERTICES];
	b3Vec3 upper[2 * B3_MAX_HULL_VERTICES];
	int lowerCount = 0;
	int upperCount = 0;
	bool anyBelow = false;
	bool anyAbove = false;

	for ( int i = 0; i < vertexCount; ++i )
	{
		float distance = b3Dot( plane.normal, points[i] ) - plane.offset;
		distances[i] = distance;
		anyBelow = anyBelow || distance < -tolerance;
		anyAbove = anyAbove || distance > tolerance;

		if ( distance <= tolerance )
		{
			lower[lowerCount++] = points[i];
		}

		if ( distance >= -tolerance )
		{
			upper[upperCount++] = points[i];
		}
	}

	if ( anyBelow == false || anyAbove == false )
	{
		return false;
	}

	// Every edge that crosses the plane adds a vertex to both parts. Visit each edge once.
	for ( int i = 0; i < hull->edgeCount; ++i )
	{
		int twin = edges[i].twin;
		if ( twin < i )
		{
			continue;
		}

		int index1 = edges[i].origin;
		int index2 = edges[twin].origin;
		float d1 = distances[index1];
		float d2 = distances[index2];
		bool crosses = ( d1 < -tolerance && d2 > tolerance ) || ( d1 > tolerance && d2 < -tolerance );
		if ( crosses == false )
		{
			continue;
		}

		float t = d1 / ( d1 - d2 );
		b3Vec3 point = b3Lerp( points[index1], points[index2], t );
		if ( lowerCount < 2 * B3_MAX_HULL_VERTICES && upperCount < 2 * B3_MAX_HULL_VERTICES )
		{
			lower[lowerCount++] = point;
			upper[upperCount++] = point;
		}
	}

	b3HullData* lowerHull = b3CreateHull( lower, lowerCount, B3_MAX_HULL_VERTICES );
	b3HullData* upperHull = b3CreateHull( upper, upperCount, B3_MAX_HULL_VERTICES );
	if ( lowerHull == NULL || upperHull == NULL )
	{
		if ( lowerHull != NULL )
		{
			b3DestroyHull( lowerHull );
		}

		if ( upperHull != NULL )
		{
			b3DestroyHull( upperHull );
		}

		return false;
	}

	*below = lowerHull;
	*above = upperHull;
	return true;
}

typedef struct HullList
{
	b3HullData** hulls;
	int count;
	int capacity;
} HullList;

static void PushHull( HullList* list, b3HullData* hull )
{
	if ( list->count == list->capacity )
	{
		list->capacity = list->capacity == 0 ? 16 : 2 * list->capacity;
		list->hulls = realloc( list->hulls, list->capacity * sizeof( b3HullData* ) );
	}

	list->hulls[list->count++] = hull;
}

static int LongestAxis( b3Vec3 extents )
{
	if ( extents.x >= extents.y && extents.x >= extents.z )
	{
		return 0;
	}

	return extents.y >= extents.z ? 1 : 2;
}

static float GetAxis( b3Vec3 v, int axis )
{
	return axis == 0 ? v.x : ( axis == 1 ? v.y : v.z );
}

static void SetAxis( b3Vec3* v, int axis, float value )
{
	if ( axis == 0 )
	{
		v->x = value;
	}
	else if ( axis == 1 )
	{
		v->y = value;
	}
	else
	{
		v->z = value;
	}
}

// Cut a hull with leaning planes until every part is about the cell size. Takes ownership of the hull.
static void FractureHull( DestructionBuilder* builder, b3HullData* hull, float cellSize, float jitter, float tilt, int depth,
						  HullList* output )
{
	b3Vec3 lower = hull->aabb.lowerBound;
	b3Vec3 size = b3Sub( hull->aabb.upperBound, lower );
	int axis = LongestAxis( size );
	float length = GetAxis( size, axis );

	if ( length <= 1.5f * cellSize || depth >= e_maxFractureDepth )
	{
		PushHull( output, hull );
		return;
	}

	// A cut across the longest axis, off center and leaning a little
	float fraction = 0.5f + jitter * ( BuilderRandom( builder ) - 0.5f );
	b3Vec3 point = b3MulAdd( lower, 0.5f, size );
	SetAxis( &point, axis, GetAxis( lower, axis ) + fraction * length );

	b3Vec3 normal = b3Vec3_zero;
	SetAxis( &normal, axis, 1.0f );
	normal.x += tilt * ( 2.0f * BuilderRandom( builder ) - 1.0f );
	normal.y += tilt * ( 2.0f * BuilderRandom( builder ) - 1.0f );
	normal.z += tilt * ( 2.0f * BuilderRandom( builder ) - 1.0f );
	normal = b3Normalize( normal );

	b3Plane plane = { normal, b3Dot( normal, point ) };
	b3HullData* below = NULL;
	b3HullData* above = NULL;
	bool split = SplitHull( hull, plane, &below, &above );

	// Reject slivers
	if ( split && ( below->volume < 0.1f * hull->volume || above->volume < 0.1f * hull->volume ) )
	{
		b3DestroyHull( below );
		b3DestroyHull( above );
		split = false;
	}

	if ( split == false )
	{
		// Fall back to a straight cut through the middle
		b3Vec3 straight = b3Vec3_zero;
		SetAxis( &straight, axis, 1.0f );
		b3Vec3 middle = b3MulAdd( lower, 0.5f, size );
		b3Plane straightPlane = { straight, b3Dot( straight, middle ) };
		split = SplitHull( hull, straightPlane, &below, &above );
	}

	if ( split == false )
	{
		PushHull( output, hull );
		return;
	}

	b3DestroyHull( hull );
	FractureHull( builder, below, cellSize, jitter, tilt, depth + 1, output );
	FractureHull( builder, above, cellSize, jitter, tilt, depth + 1, output );
}

// Cut a box with planes along its axes. The parts are boxes, so no hull has to be computed.
static void FractureBox( DestructionBuilder* builder, b3Vec3 center, b3Vec3 halfExtents, float cellSize, float jitter, int depth,
						 HullList* output )
{
	b3Vec3 size = b3MulSV( 2.0f, halfExtents );
	int axis = LongestAxis( size );
	float length = GetAxis( size, axis );

	if ( length <= 1.5f * cellSize || depth >= e_maxFractureDepth )
	{
		b3BoxHull box = b3MakeOffsetBoxHull( halfExtents.x, halfExtents.y, halfExtents.z, center );
		PushHull( output, b3CloneHull( &box.base ) );
		return;
	}

	float fraction = 0.5f + jitter * ( BuilderRandom( builder ) - 0.5f );
	float lower = GetAxis( center, axis ) - GetAxis( halfExtents, axis );
	float cut = lower + fraction * length;

	b3Vec3 centerA = center, centerB = center;
	b3Vec3 halfA = halfExtents, halfB = halfExtents;
	SetAxis( &centerA, axis, 0.5f * ( lower + cut ) );
	SetAxis( &halfA, axis, 0.5f * ( cut - lower ) );
	SetAxis( &centerB, axis, 0.5f * ( cut + lower + length ) );
	SetAxis( &halfB, axis, 0.5f * ( lower + length - cut ) );

	FractureBox( builder, centerA, halfA, cellSize, jitter, depth + 1, output );
	FractureBox( builder, centerB, halfB, cellSize, jitter, depth + 1, output );
}

void AddFracturedBox( DestructionBuilder* builder, b3Transform transform, b3Vec3 halfExtents, const FractureDef* def )
{
	HullList cells = { 0 };

	if ( def->tilt > 0.0f )
	{
		b3BoxHull box = b3MakeBoxHull( halfExtents.x, halfExtents.y, halfExtents.z );
		FractureHull( builder, b3CloneHull( &box.base ), def->cellSize, def->jitter, def->tilt, 0, &cells );
	}
	else
	{
		FractureBox( builder, b3Vec3_zero, halfExtents, def->cellSize, def->jitter, 0, &cells );
	}

	for ( int i = 0; i < cells.count; ++i )
	{
		b3HullData* cell = cells.hulls[i];

		// The debris level: cut a copy of the cell again, smaller
		HullList children = { 0 };
		if ( def->childSize > 0.0f )
		{
			if ( def->tilt > 0.0f )
			{
				FractureHull( builder, b3CloneHull( cell ), def->childSize, def->jitter, def->tilt, 0, &children );
			}
			else
			{
				b3Vec3 center = b3Lerp( cell->aabb.lowerBound, cell->aabb.upperBound, 0.5f );
				b3Vec3 half = b3MulSV( 0.5f, b3Sub( cell->aabb.upperBound, cell->aabb.lowerBound ) );
				FractureBox( builder, center, half, def->childSize, def->jitter, 0, &children );
			}
		}

		int parentIndex = AddOwnedFragment( builder, cell, transform, B3_NULL_INDEX, def->materialIndex, def->isAnchored );

		if ( children.count > 1 )
		{
			for ( int j = 0; j < children.count; ++j )
			{
				AddOwnedFragment( builder, children.hulls[j], transform, parentIndex, def->materialIndex, false );
			}
		}
		else
		{
			for ( int j = 0; j < children.count; ++j )
			{
				b3DestroyHull( children.hulls[j] );
			}
		}

		free( children.hulls );
	}

	free( cells.hulls );
}

b3StructureData* BuildStructure( DestructionBuilder* builder )
{
	if ( builder->count == 0 )
	{
		return NULL;
	}

	b3FragmentDef* fragmentDefs = malloc( builder->count * sizeof( b3FragmentDef ) );
	for ( int i = 0; i < builder->count; ++i )
	{
		const BuilderFragment* fragment = builder->fragments + i;
		b3FragmentDef* fragmentDef = fragmentDefs + i;
		*fragmentDef = b3DefaultFragmentDef();
		fragmentDef->hulls = (const b3HullData* const*)&fragment->hull;
		fragmentDef->hullCount = 1;
		fragmentDef->transform = fragment->transform;
		fragmentDef->parentIndex = fragment->parentIndex;
		fragmentDef->materialIndex = fragment->materialIndex;
		fragmentDef->isAnchored = fragment->isAnchored;
		fragmentDef->userData = (uint64_t)i;
	}

	b3StructureDef def = b3DefaultStructureDef();
	def.fragments = fragmentDefs;
	def.fragmentCount = builder->count;
	if ( builder->materialCount > 0 )
	{
		def.materials = builder->materials;
		def.materialCount = builder->materialCount;
	}

	b3StructureData* structure = b3CreateStructure( &def );
	free( fragmentDefs );
	return structure;
}

BuildingDef DefaultBuildingDef( void )
{
	BuildingDef def = { 0 };
	def.floorCount = 5;
	def.bayCountX = 3;
	def.bayCountZ = 3;
	def.baySize = 5.0f;
	def.floorHeight = 3.2f;
	def.columnSize = 0.5f;
	def.slabThickness = 0.3f;
	def.wallThickness = 0.25f;
	def.enableWalls = true;
	def.fracture = DefaultFractureDef();
	def.fracture.cellSize = 1.0f;
	def.fracture.childSize = 0.5f;
	def.material = b3DefaultStructureMaterial();
	def.seed = 1234;
	return def;
}

static void AddBox( DestructionBuilder* builder, b3Vec3 center, b3Vec3 halfExtents, const FractureDef* fracture )
{
	b3Transform transform = { center, b3Quat_identity };
	AddFracturedBox( builder, transform, halfExtents, fracture );
}

// A wall panel between two columns. Every other panel has a window, and the ground floor has a door.
static void AddWallPanel( DestructionBuilder* builder, const BuildingDef* def, b3Vec3 center, float length, float height,
						  bool alongX, int style )
{
	float thickness = 0.5f * def->wallThickness;
	float halfLength = 0.5f * length;
	float halfHeight = 0.5f * height;

	if ( style == 0 )
	{
		// Solid
		b3Vec3 half = alongX ? (b3Vec3){ halfLength, halfHeight, thickness } : (b3Vec3){ thickness, halfHeight, halfLength };
		AddBox( builder, center, half, &def->fracture );
		return;
	}

	if ( style == 2 )
	{
		// Door: an open bay
		return;
	}

	// Window: a sill below, a lintel above, and a pier at each end
	float sill = 0.3f * height;
	float lintel = 0.2f * height;
	float pier = 0.2f * length;
	float bottom = center.y - halfHeight;

	b3Vec3 sillCenter = center;
	sillCenter.y = bottom + 0.5f * sill;
	b3Vec3 sillHalf = alongX ? (b3Vec3){ halfLength, 0.5f * sill, thickness } : (b3Vec3){ thickness, 0.5f * sill, halfLength };
	AddBox( builder, sillCenter, sillHalf, &def->fracture );

	b3Vec3 lintelCenter = center;
	lintelCenter.y = bottom + height - 0.5f * lintel;
	b3Vec3 lintelHalf = alongX ? (b3Vec3){ halfLength, 0.5f * lintel, thickness } : (b3Vec3){ thickness, 0.5f * lintel, halfLength };
	AddBox( builder, lintelCenter, lintelHalf, &def->fracture );

	float pierHeight = height - sill - lintel;
	for ( int side = -1; side <= 1; side += 2 )
	{
		b3Vec3 pierCenter = center;
		pierCenter.y = bottom + sill + 0.5f * pierHeight;
		float offset = (float)side * ( halfLength - 0.5f * pier );
		if ( alongX )
		{
			pierCenter.x += offset;
		}
		else
		{
			pierCenter.z += offset;
		}

		b3Vec3 pierHalf = alongX ? (b3Vec3){ 0.5f * pier, 0.5f * pierHeight, thickness } : (b3Vec3){ thickness, 0.5f * pierHeight, 0.5f * pier };
		AddBox( builder, pierCenter, pierHalf, &def->fracture );
	}
}

b3StructureData* CreateBuildingStructure( const BuildingDef* def )
{
	DestructionBuilder* builder = CreateDestructionBuilder( def->seed );
	AddDestructionMaterial( builder, &def->material );

	float width = def->bayCountX * def->baySize;
	float depth = def->bayCountZ * def->baySize;
	float columnHalf = 0.5f * def->columnSize;
	float columnHeight = def->floorHeight - def->slabThickness;

	for ( int floor = 0; floor < def->floorCount; ++floor )
	{
		float base = floor * def->floorHeight;

		// Columns on the grid lines
		for ( int i = 0; i <= def->bayCountX; ++i )
		{
			for ( int j = 0; j <= def->bayCountZ; ++j )
			{
				b3Vec3 center = { -0.5f * width + i * def->baySize, base + 0.5f * columnHeight, -0.5f * depth + j * def->baySize };
				b3Vec3 half = { columnHalf, 0.5f * columnHeight, columnHalf };
				AddBox( builder, center, half, &def->fracture );
			}
		}

		// Outer walls fill the gaps between the outer columns
		if ( def->enableWalls )
		{
			float panelLength = def->baySize - def->columnSize;
			float wallY = base + 0.5f * columnHeight;

			for ( int i = 0; i < def->bayCountX; ++i )
			{
				float x = -0.5f * width + ( i + 0.5f ) * def->baySize;
				int style = ( i + floor ) % 2;
				int frontStyle = ( floor == 0 && i == def->bayCountX / 2 ) ? 2 : style;
				AddWallPanel( builder, def, (b3Vec3){ x, wallY, -0.5f * depth }, panelLength, columnHeight, true, frontStyle );
				AddWallPanel( builder, def, (b3Vec3){ x, wallY, 0.5f * depth }, panelLength, columnHeight, true, style );
			}

			for ( int j = 0; j < def->bayCountZ; ++j )
			{
				float z = -0.5f * depth + ( j + 0.5f ) * def->baySize;
				int style = ( j + floor + 1 ) % 2;
				AddWallPanel( builder, def, (b3Vec3){ -0.5f * width, wallY, z }, panelLength, columnHeight, false, style );
				AddWallPanel( builder, def, (b3Vec3){ 0.5f * width, wallY, z }, panelLength, columnHeight, false, style );
			}
		}

		// The floor slab rests on the columns and the walls
		b3Vec3 slabCenter = { 0.0f, base + columnHeight + 0.5f * def->slabThickness, 0.0f };
		b3Vec3 slabHalf = { 0.5f * width + columnHalf, 0.5f * def->slabThickness, 0.5f * depth + columnHalf };
		AddBox( builder, slabCenter, slabHalf, &def->fracture );
	}

	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	return structure;
}

b3StructureData* CreateWallStructure( float width, float height, float thickness, const FractureDef* fracture, uint32_t seed )
{
	DestructionBuilder* builder = CreateDestructionBuilder( seed );
	b3Transform transform = { { 0.0f, 0.5f * height, 0.0f }, b3Quat_identity };
	b3Vec3 half = { 0.5f * width, 0.5f * height, 0.5f * thickness };
	AddFracturedBox( builder, transform, half, fracture );

	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	return structure;
}

b3StructureData* CreateBeamStructure( float length, float height, float depth, int cellCount, bool anchorFirst,
									  const b3StructureMaterial* material )
{
	DestructionBuilder* builder = CreateDestructionBuilder( 1 );
	if ( material != NULL )
	{
		AddDestructionMaterial( builder, material );
	}

	float cellLength = length / cellCount;
	for ( int i = 0; i < cellCount; ++i )
	{
		b3Vec3 center = { ( i + 0.5f ) * cellLength, 0.0f, 0.0f };
		b3BoxHull box = b3MakeOffsetBoxHull( 0.5f * cellLength, 0.5f * height, 0.5f * depth, center );
		AddFragment( builder, &box.base, B3_NULL_INDEX, 0, anchorFirst && i == 0 );
	}

	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	return structure;
}

b3StructureData* CreateTowerStructure( float width, float height, const FractureDef* fracture, uint32_t seed )
{
	DestructionBuilder* builder = CreateDestructionBuilder( seed );
	b3Transform transform = { { 0.0f, 0.5f * height, 0.0f }, b3Quat_identity };
	b3Vec3 half = { 0.5f * width, 0.5f * height, 0.5f * width };
	AddFracturedBox( builder, transform, half, fracture );

	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	return structure;
}

b3StructureData* CreateBoulderStructure( float radius, float childSize, const b3StructureMaterial* material, uint32_t seed )
{
	DestructionBuilder* builder = CreateDestructionBuilder( seed );
	if ( material != NULL )
	{
		AddDestructionMaterial( builder, material );
	}

	b3HullData* rock = b3CreateRock( radius );
	int parentIndex = AddFragment( builder, rock, B3_NULL_INDEX, 0, false );

	// The debris level: the same rock cut into chunks
	HullList children = { 0 };
	FractureHull( builder, rock, childSize, 0.3f, 0.35f, 0, &children );
	for ( int i = 0; i < children.count; ++i )
	{
		AddOwnedFragment( builder, children.hulls[i], b3Transform_identity, parentIndex, 0, false );
	}
	free( children.hulls );

	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	return structure;
}

void ApplyBlast( b3WorldId worldId, b3Pos position, float radius, float damage, float impulsePerArea )
{
	b3RadialDamageDef damageDef = b3DefaultRadialDamageDef();
	damageDef.position = position;
	damageDef.innerRadius = 0.5f * radius;
	damageDef.outerRadius = radius;
	damageDef.damage = damage;
	b3World_ApplyRadialDamage( worldId, &damageDef );

	b3ExplosionDef explosionDef = b3DefaultExplosionDef();
	explosionDef.position = position;
	explosionDef.radius = 0.5f * radius;
	explosionDef.falloff = 0.5f * radius;
	explosionDef.impulsePerArea = impulsePerArea;
	b3World_Explode( worldId, &explosionDef );
}

// Building collapse benchmark

enum
{
	e_maxBuildings = 16,
	e_blastStep = 30,

	// Steps between the column lines of one building, and between two buildings
	e_lineDelay = 8,
	e_buildingDelay = 45,
};

static b3StructureData* s_buildingStructure;
static b3DestructibleId s_buildingIds[e_maxBuildings];
static b3Vec3 s_buildingOrigins[e_maxBuildings];
static int s_buildingCount = 4;
static int s_buildingFloorCount = 6;
static BuildingDef s_buildingDef;

void SetBuildingCollapseScale( int buildingCount, int floorCount )
{
	s_buildingCount = b3ClampInt( buildingCount, 1, e_maxBuildings );
	s_buildingFloorCount = b3ClampInt( floorCount, 1, 40 );
}

void GetBuildingCollapseCapacity( b3Capacity* capacity )
{
	capacity->staticBodyCount = 1 + s_buildingCount;
	capacity->dynamicBodyCount = 4096 * s_buildingCount;
	capacity->staticShapeCount = 4096 * s_buildingCount;
	capacity->dynamicShapeCount = 8192 * s_buildingCount;
	capacity->contactCount = 16384 * s_buildingCount;
}

void CreateBuildingCollapse( b3WorldId worldId )
{
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
		b3BodyId groundId = b3CreateBody( worldId, &bodyDef );

		b3BoxHull box = b3MakeBoxHull( 250.0f, 1.0f, 250.0f );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateHullShape( groundId, &shapeDef, &box.base );
	}

	s_buildingDef = DefaultBuildingDef();
	s_buildingDef.floorCount = BENCHMARK_DEBUG ? b3MinInt( 2, s_buildingFloorCount ) : s_buildingFloorCount;
	s_buildingStructure = CreateBuildingStructure( &s_buildingDef );

	// Buildings in a row with a street between them
	float spacing = s_buildingDef.bayCountX * s_buildingDef.baySize + 12.0f;
	int count = BENCHMARK_DEBUG ? 1 : s_buildingCount;

	for ( int i = 0; i < e_maxBuildings; ++i )
	{
		s_buildingIds[i] = b3_nullDestructibleId;
	}

	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 origin = { ( (float)i - 0.5f * (float)( count - 1 ) ) * spacing, 0.0f, 0.0f };
		s_buildingOrigins[i] = origin;

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = s_buildingStructure;
		def.position = b3ToPos( origin );
		def.seed = 100 + (uint32_t)i;
		def.name = "building";
		s_buildingIds[i] = b3CreateDestructible( worldId, &def );
	}
}

void StepBuildingCollapse( b3WorldId worldId, int stepIndex )
{
	// A demolition: a charge on every ground floor column, set off one column line after the other
	// from one side, so the building leans that way as it comes down. The buildings go in turn.
	float halfWidth = 0.5f * s_buildingDef.bayCountX * s_buildingDef.baySize;
	float halfDepth = 0.5f * s_buildingDef.bayCountZ * s_buildingDef.baySize;
	int lineCount = s_buildingDef.bayCountX + 1;

	for ( int i = 0; i < e_maxBuildings; ++i )
	{
		if ( b3Destructible_IsValid( s_buildingIds[i] ) == false )
		{
			continue;
		}

		int elapsed = stepIndex - e_blastStep - i * e_buildingDelay;
		if ( elapsed < 0 || elapsed % e_lineDelay != 0 || elapsed / e_lineDelay >= lineCount )
		{
			continue;
		}

		int line = elapsed / e_lineDelay;
		for ( int j = 0; j <= s_buildingDef.bayCountZ; ++j )
		{
			b3Vec3 offset = { -halfWidth + line * s_buildingDef.baySize, 1.2f, -halfDepth + j * s_buildingDef.baySize };
			b3Pos position = b3ToPos( b3Add( s_buildingOrigins[i], offset ) );
			ApplyBlast( worldId, position, 3.0f, 4.0e6f, 6000.0f );
		}
	}
}

b3DestructibleId GetBuildingCollapseBuilding( int index )
{
	return index >= 0 && index < e_maxBuildings ? s_buildingIds[index] : b3_nullDestructibleId;
}

void DestroyBuildingCollapse( void )
{
	// The benchmark app calls this before it destroys the world and the samples call it after. A
	// structure must outlive every destructible made from it, so destroy any that are still alive.
	for ( int i = 0; i < e_maxBuildings; ++i )
	{
		if ( b3Destructible_IsValid( s_buildingIds[i] ) )
		{
			b3DestroyDestructible( s_buildingIds[i] );
		}

		s_buildingIds[i] = b3_nullDestructibleId;
	}

	b3DestroyStructure( s_buildingStructure );
	s_buildingStructure = NULL;
}
