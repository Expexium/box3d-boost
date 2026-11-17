// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"
#include "determinism.h"
#include "simd.h"
#include "stability.h"
#include "test_macros.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef BOX3D_PROFILE
	#include <tracy/TracyC.h>
#else
	#define TracyCFrameMark
#endif

// Double precision accumulates body positions in double, so the settle/sleep step and the
// state hash differ from the float build. Both modes are internally deterministic.
#if defined( BOX3D_DOUBLE_PRECISION )
#define RAGDOLL_SLEEP_STEP 282
#define RAGDOLL_HASH 0x6E4B13AD
#define WAVE_PILE_SLEEP_STEP 316
#define WAVE_PILE_HASH 0x7CB7D931
#define QUERY_SPAWN_SLEEP_STEP 242
#define QUERY_SPAWN_HASH 0x489585E6
#define QUERY_SPAWN_HIT_COUNT 59
#define QUERY_SPAWN_QUERY_HASH 0x5B4429DC
#define MESH_DROP_SLEEP_STEP 197
#define MESH_DROP_HASH 0xA2E8BB21
#else
#define RAGDOLL_SLEEP_STEP 277
#define RAGDOLL_HASH 0x5E042B6E
#define WAVE_PILE_SLEEP_STEP 300
#define WAVE_PILE_HASH 0x3155DA54
#define QUERY_SPAWN_SLEEP_STEP 242
#define QUERY_SPAWN_HASH 0xF3AD9BA6
#define QUERY_SPAWN_HIT_COUNT 59
#define QUERY_SPAWN_QUERY_HASH 0xE3271F3D
#define MESH_DROP_SLEEP_STEP 197
#define MESH_DROP_HASH 0x96B62947
#endif

// The goldens above pin exact values for the default four point manifold. A build that
// overrides B3_MAX_MANIFOLD_POINTS produces a different contact set, so those checks drop
// out and only the run to run agreement below applies.
#if B3_MAX_MANIFOLD_POINTS == 4
#define ENSURE_GOLDEN( condition ) ENSURE( condition )
#else
#define ENSURE_GOLDEN( condition ) ( (void)0 )
#endif

typedef struct DeterminismResult
{
	int sleepStep;
	uint32_t hash;
	int queryHitCount;
	uint32_t queryHash;
	bool seeded;
} DeterminismResult;

// Every worker count has to reproduce the first run exactly. Scenarios without queries leave
// the query fields zero in both the reference and the result.
static int EnsureRepeatable( DeterminismResult* reference, DeterminismResult result )
{
	if ( reference->seeded == false )
	{
		result.seeded = true;
		*reference = result;
		return 0;
	}

	ENSURE( result.sleepStep == reference->sleepStep );
	ENSURE( result.hash == reference->hash );
	ENSURE( result.queryHitCount == reference->queryHitCount );
	ENSURE( result.queryHash == reference->queryHash );

	return 0;
}

static int SingleMultithreadingTest( int workerCount, DeterminismResult* reference )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;

	b3WorldId worldId = b3CreateWorld( &worldDef );

	FallingRagdollData data = CreateFallingRagdolls( worldId );

	float timeStep = 1.0f / 60.0f;

	int stepLimit = 500;
	for ( int i = 0; i < stepLimit; ++i )
	{
		int subStepCount = 4;
		b3World_Step( worldId, timeStep, subStepCount );
		TracyCFrameMark;

		bool done = UpdateFallingRagdolls( worldId, &data );
		if ( done )
		{
			break;
		}
	}

	b3DestroyWorld( worldId );

	if ( data.sleepStep != RAGDOLL_SLEEP_STEP || data.hash != RAGDOLL_HASH )
	{
		printf( "  workers=%d sleepStep=%d hash=0x%08X\n", workerCount, data.sleepStep, data.hash );
	}

	ENSURE_GOLDEN( data.sleepStep == RAGDOLL_SLEEP_STEP );
	ENSURE_GOLDEN( data.hash == RAGDOLL_HASH );
	ENSURE( EnsureRepeatable( reference, (DeterminismResult){ .sleepStep = data.sleepStep, .hash = data.hash } ) == 0 );

	DestroyFallingRagdolls( &data );

	return 0;
}

// Test multithreaded determinism.
static int MultithreadingTest( void )
{
	DeterminismResult reference = { 0 };
	for ( int workerCount = 1; workerCount < 6; ++workerCount )
	{
		int result = SingleMultithreadingTest( workerCount, &reference );
		ENSURE( result == 0 );
	}

	return 0;
}

// Test cross platform determinism.
static int CrossPlatformTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	FallingRagdollData data = CreateFallingRagdolls( worldId );

	float timeStep = 1.0f / 60.0f;

	bool done = false;
	while ( done == false )
	{
		int subStepCount = 4;
		b3World_Step( worldId, timeStep, subStepCount );
		TracyCFrameMark;

		done = UpdateFallingRagdolls( worldId, &data );
	}

	if ( data.sleepStep != RAGDOLL_SLEEP_STEP || data.hash != RAGDOLL_HASH )
	{
		printf( "  cross-platform sleepStep=%d hash=0x%08X\n", data.sleepStep, data.hash );
	}

	ENSURE_GOLDEN( data.sleepStep == RAGDOLL_SLEEP_STEP );
	ENSURE_GOLDEN( data.hash == RAGDOLL_HASH );

	DestroyFallingRagdolls( &data );

	b3DestroyWorld( worldId );

	return 0;
}

static int SingleWavePileTest( int workerCount, DeterminismResult* reference )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;

	b3WorldId worldId = b3CreateWorld( &worldDef );

	WavePileData data = CreateWavePile( worldId );

	float timeStep = 1.0f / 60.0f;

	// Rolling resistance must put the pile to sleep within 500 steps
	bool done = false;
	for ( int i = 0; i < 500 && done == false; ++i )
	{
		int subStepCount = 4;
		b3World_Step( worldId, timeStep, subStepCount );
		TracyCFrameMark;

		done = UpdateWavePile( worldId, &data );
	}

	b3DestroyWorld( worldId );

	if ( data.sleepStep != WAVE_PILE_SLEEP_STEP || data.hash != WAVE_PILE_HASH )
	{
		printf( "  wave pile workers=%d sleepStep=%d hash=0x%08X\n", workerCount, data.sleepStep, data.hash );
	}

	ENSURE( done == true );
	ENSURE_GOLDEN( data.sleepStep == WAVE_PILE_SLEEP_STEP );
	ENSURE_GOLDEN( data.hash == WAVE_PILE_HASH );
	ENSURE( EnsureRepeatable( reference, (DeterminismResult){ .sleepStep = data.sleepStep, .hash = data.hash } ) == 0 );

	DestroyWavePile( &data );

	return 0;
}

// Test multithreaded determinism of a mixed convex pile on a wave height field.
static int WavePileTest( void )
{
	DeterminismResult reference = { 0 };
	for ( int workerCount = 1; workerCount <= 4; ++workerCount )
	{
		int result = SingleWavePileTest( workerCount, &reference );
		ENSURE( result == 0 );
	}

	return 0;
}

static int SingleQuerySpawnTest( int workerCount, DeterminismResult* reference )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;

	b3WorldId worldId = b3CreateWorld( &worldDef );

	QuerySpawnData data = CreateQuerySpawn( worldId );

	float timeStep = 1.0f / 60.0f;

	bool done = false;
	for ( int i = 0; i < 1000 && done == false; ++i )
	{
		int subStepCount = 4;
		b3World_Step( worldId, timeStep, subStepCount );
		TracyCFrameMark;

		done = UpdateQuerySpawn( worldId, &data );
	}

	b3DestroyWorld( worldId );

	if ( data.sleepStep != QUERY_SPAWN_SLEEP_STEP || data.hash != QUERY_SPAWN_HASH || data.queryHitCount != QUERY_SPAWN_HIT_COUNT ||
		 data.queryHash != QUERY_SPAWN_QUERY_HASH )
	{
		printf( "  query spawn workers=%d sleepStep=%d hash=0x%08X hits=%d queryHash=0x%08X\n", workerCount, data.sleepStep,
				data.hash, data.queryHitCount, data.queryHash );
	}

	ENSURE( done == true );
	ENSURE( data.spawnCount == QUERY_SPAWN_COUNT );
	ENSURE_GOLDEN( data.sleepStep == QUERY_SPAWN_SLEEP_STEP );
	ENSURE_GOLDEN( data.hash == QUERY_SPAWN_HASH );
	ENSURE_GOLDEN( data.queryHitCount == QUERY_SPAWN_HIT_COUNT );
	ENSURE_GOLDEN( data.queryHash == QUERY_SPAWN_QUERY_HASH );
	ENSURE( EnsureRepeatable( reference, (DeterminismResult){ .sleepStep = data.sleepStep,
															  .hash = data.hash,
															  .queryHitCount = data.queryHitCount,
															  .queryHash = data.queryHash } ) == 0 );

	DestroyQuerySpawn( &data );

	return 0;
}

// Test determinism of world queries by feeding their results back into the simulation.
static int QuerySpawnTest( void )
{
	DeterminismResult reference = { 0 };
	for ( int workerCount = 1; workerCount <= 4; ++workerCount )
	{
		int result = SingleQuerySpawnTest( workerCount, &reference );
		ENSURE( result == 0 );
	}

	return 0;
}

static int SingleMeshDropTest( int workerCount, DeterminismResult* reference )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;

	b3WorldId worldId = b3CreateWorld( &worldDef );

	MeshDropData data = CreateMeshDrop( worldId, b3Pos_zero );

	float timeStep = 1.0f / 60.0f;

	bool done = false;
	for ( int i = 0; i < 400 && done == false; ++i )
	{
		int subStepCount = 4;
		b3World_Step( worldId, timeStep, subStepCount );
		TracyCFrameMark;

		done = UpdateMeshDrop( worldId, &data );
	}

	b3DestroyWorld( worldId );

	if ( data.sleepStep != MESH_DROP_SLEEP_STEP || data.hash != MESH_DROP_HASH )
	{
		printf( "  mesh drop workers=%d sleepStep=%d hash=0x%08X\n", workerCount, data.sleepStep, data.hash );
	}

	ENSURE( done == true );
	ENSURE_GOLDEN( data.sleepStep == MESH_DROP_SLEEP_STEP );
	ENSURE_GOLDEN( data.hash == MESH_DROP_HASH );
	ENSURE( EnsureRepeatable( reference, (DeterminismResult){ .sleepStep = data.sleepStep, .hash = data.hash } ) == 0 );

	DestroyMeshDrop( &data );

	return 0;
}

// Test continuous collision determinism. Thin fast boxes need CCD against the wave mesh.
// The scene is large, so only the single threaded and widest schedules run.
static int MeshDropTest( void )
{
	DeterminismResult reference = { 0 };
	int workerCounts[2] = { 1, 4 };
	for ( int i = 0; i < 2; ++i )
	{
		int result = SingleMeshDropTest( workerCounts[i], &reference );
		ENSURE( result == 0 );
	}

	return 0;
}

static uint32_t HashFloats( uint32_t hash, const float* values, int count )
{
	return b3Hash( hash, (const uint8_t*)values, count * (int)sizeof( float ) );
}

// Rolling resistance is mixed per contact and the manifolds have one, two, and four points, so a width 8
// constraint can mix lanes that width 4 keeps in separate constraints. The hash covers the stored contact
// impulses bitwise, including the sign of zero, because those survive into warm starting and contact data.
static int SingleRollingMixTest( int workerCount, DeterminismResult* reference )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef groundDef = b3DefaultBodyDef();
	b3BodyId groundId = b3CreateBody( worldId, &groundDef );
	b3BoxHull groundBox = b3MakeBoxHull( 40.0f, 1.0f, 40.0f );
	b3ShapeDef groundShapeDef = b3DefaultShapeDef();
	b3CreateHullShape( groundId, &groundShapeDef, &groundBox.base );

	enum
	{
		rowCount = 8,
		columnCount = 8,
		bodyCount = rowCount * columnCount
	};

	b3BodyId bodyIds[bodyCount];
	b3BoxHull box = b3MakeBoxHull( 0.4f, 0.4f, 0.4f );
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.4f };
	b3Capsule capsule = { { -0.3f, 0.0f, 0.0f }, { 0.3f, 0.0f, 0.0f }, 0.25f };

	uint32_t seed = 271828u;
	for ( int i = 0; i < bodyCount; ++i )
	{
		int row = i / columnCount;
		int column = i % columnCount;

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ 2.0f * (float)column - 7.0f, 1.45f, 2.0f * (float)row - 7.0f };
		bodyIds[i] = b3CreateBody( worldId, &bodyDef );

		seed = seed * 1664525u + 1013904223u;
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.density = 1.0f;
		shapeDef.baseMaterial.rollingResistance = ( seed >> 28 ) < 5 ? 0.2f : 0.0f;

		switch ( i % 3 )
		{
			case 0:
				b3CreateSphereShape( bodyIds[i], &shapeDef, &sphere );
				break;
			case 1:
				b3CreateCapsuleShape( bodyIds[i], &shapeDef, &capsule );
				break;
			default:
				b3CreateHullShape( bodyIds[i], &shapeDef, &box.base );
				break;
		}

		float sign = ( i & 1 ) ? -1.0f : 1.0f;
		b3Body_SetLinearVelocity( bodyIds[i], (b3Vec3){ -0.0f, -0.5f, sign * 0.5f } );
		b3Body_SetAngularVelocity( bodyIds[i], (b3Vec3){ sign * 3.0f, -0.0f, -2.0f * sign } );
	}

	uint32_t hash = B3_HASH_INIT;
	int stepCount = 120;
	for ( int step = 0; step < stepCount; ++step )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );

		for ( int i = 0; i < bodyCount; ++i )
		{
			b3WorldTransform xf = b3Body_GetTransform( bodyIds[i] );
			hash = b3Hash( hash, (const uint8_t*)&xf, sizeof( b3WorldTransform ) );

			b3ContactData contactData[4];
			int contactCount = b3Body_GetContactData( bodyIds[i], contactData, ARRAY_COUNT( contactData ) );
			for ( int c = 0; c < contactCount; ++c )
			{
				for ( int m = 0; m < contactData[c].manifoldCount; ++m )
				{
					const b3Manifold* manifold = contactData[c].manifolds + m;
					hash = HashFloats( hash, &manifold->rollingImpulse.x, 3 );
					hash = HashFloats( hash, &manifold->frictionImpulse.x, 3 );
					hash = HashFloats( hash, &manifold->twistImpulse, 1 );

					for ( int p = 0; p < manifold->pointCount; ++p )
					{
						hash = HashFloats( hash, &manifold->points[p].normalImpulse, 1 );
						hash = HashFloats( hash, &manifold->points[p].totalNormalImpulse, 1 );
					}
				}
			}
		}
	}

	b3DestroyWorld( worldId );

	ENSURE( EnsureRepeatable( reference, (DeterminismResult){ .sleepStep = stepCount, .hash = hash } ) == 0 );
	return 0;
}

// Mesh contacts are grouped across lanes by manifold count and padded to the largest count in the group. Width 8 groups
// differ from width 4 groups, so mixing large and small shapes on a fine mesh gives each body a width dependent number
// of padded manifolds. The hash covers velocities and impulses bitwise to catch a padded manifold flipping a signed zero.
static int SingleMeshMixTest( int workerCount, DeterminismResult* reference )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3MeshData* mesh = b3CreateWaveMesh( 40, 40, 0.5f, 0.3f, 0.2f, 0.3f );
	{
		b3BodyDef groundDef = b3DefaultBodyDef();
		groundDef.position = (b3Pos){ -10.0f, 0.0f, -10.0f };
		b3BodyId groundId = b3CreateBody( worldId, &groundDef );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateMeshShape( groundId, &shapeDef, mesh, b3Vec3_one );
	}

	enum
	{
		rowCount = 8,
		columnCount = 8,
		bodyCount = rowCount * columnCount
	};

	b3BodyId bodyIds[bodyCount];
	b3BoxHull smallBox = b3MakeBoxHull( 0.2f, 0.2f, 0.2f );
	b3BoxHull largeBox = b3MakeBoxHull( 0.9f, 0.3f, 0.9f );
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.3f };
	b3Capsule capsule = { { -0.5f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f }, 0.2f };

	uint32_t seed = 314159u;
	for ( int i = 0; i < bodyCount; ++i )
	{
		int row = i / columnCount;
		int column = i % columnCount;

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ 2.2f * (float)column - 7.7f, 1.2f, 2.2f * (float)row - 7.7f };
		bodyIds[i] = b3CreateBody( worldId, &bodyDef );

		seed = seed * 1664525u + 1013904223u;
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.density = 1.0f;
		shapeDef.baseMaterial.rollingResistance = ( seed >> 28 ) < 5 ? 0.2f : 0.0f;

		switch ( ( seed >> 24 ) & 3 )
		{
			case 0:
				b3CreateHullShape( bodyIds[i], &shapeDef, &largeBox.base );
				break;
			case 1:
				b3CreateHullShape( bodyIds[i], &shapeDef, &smallBox.base );
				break;
			case 2:
				b3CreateSphereShape( bodyIds[i], &shapeDef, &sphere );
				break;
			default:
				b3CreateCapsuleShape( bodyIds[i], &shapeDef, &capsule );
				break;
		}

		float sign = ( i & 1 ) ? -1.0f : 1.0f;
		b3Body_SetLinearVelocity( bodyIds[i], (b3Vec3){ -0.0f, -0.5f, -0.0f } );
		b3Body_SetAngularVelocity( bodyIds[i], (b3Vec3){ -0.0f, sign * 0.5f, -0.0f } );
	}

	uint32_t hash = B3_HASH_INIT;
	int stepCount = 180;
	for ( int step = 0; step < stepCount; ++step )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );

		for ( int i = 0; i < bodyCount; ++i )
		{
			b3WorldTransform xf = b3Body_GetTransform( bodyIds[i] );
			hash = b3Hash( hash, (const uint8_t*)&xf, sizeof( b3WorldTransform ) );

			b3Vec3 v = b3Body_GetLinearVelocity( bodyIds[i] );
			b3Vec3 w = b3Body_GetAngularVelocity( bodyIds[i] );
			hash = HashFloats( hash, &v.x, 3 );
			hash = HashFloats( hash, &w.x, 3 );

			b3ContactData contactData[8];
			int contactCount = b3Body_GetContactData( bodyIds[i], contactData, ARRAY_COUNT( contactData ) );
			for ( int c = 0; c < contactCount; ++c )
			{
				for ( int m = 0; m < contactData[c].manifoldCount; ++m )
				{
					const b3Manifold* manifold = contactData[c].manifolds + m;
					hash = HashFloats( hash, &manifold->rollingImpulse.x, 3 );
					hash = HashFloats( hash, &manifold->frictionImpulse.x, 3 );
					hash = HashFloats( hash, &manifold->twistImpulse, 1 );

					for ( int p = 0; p < manifold->pointCount; ++p )
					{
						hash = HashFloats( hash, &manifold->points[p].normalImpulse, 1 );
						hash = HashFloats( hash, &manifold->points[p].totalNormalImpulse, 1 );
					}
				}
			}
		}
	}

	b3DestroyWorld( worldId );
	b3DestroyMesh( mesh );

	ENSURE( EnsureRepeatable( reference, (DeterminismResult){ .sleepStep = stepCount, .hash = hash } ) == 0 );
	return 0;
}

typedef int SceneFcn( int workerCount, DeterminismResult* reference );

static int RunSceneAtWidth( SceneFcn* scene, int width, DeterminismResult* result )
{
	b3SetSIMDWidth( width );
	int status = scene( 1, result );
	b3SetSIMDWidth( 0 );
	return status;
}

// Width 4 and the native width must produce bitwise identical simulations.
static int SIMDWidthTest( void )
{
	b3SetSIMDWidth( 0 );
	int nativeWidth = b3GetSIMDWidth();
	if ( nativeWidth == 4 )
	{
		printf( "  subtest skipped: SIMDWidthTest, native SIMD width is 4\n" );
		return 0;
	}

	SceneFcn* scenes[] = {
		SingleMultithreadingTest, SingleWavePileTest, SingleQuerySpawnTest, SingleMeshDropTest, SingleRollingMixTest, SingleMeshMixTest,
	};

	for ( int i = 0; i < ARRAY_COUNT( scenes ); ++i )
	{
		DeterminismResult narrow = { 0 };
		DeterminismResult wide = { 0 };

		int narrowStatus = RunSceneAtWidth( scenes[i], 4, &narrow );
		int wideStatus = RunSceneAtWidth( scenes[i], nativeWidth, &wide );

		ENSURE( narrowStatus == 0 );
		ENSURE( wideStatus == 0 );
		ENSURE( narrow.seeded && wide.seeded );
		ENSURE( narrow.sleepStep == wide.sleepStep );
		ENSURE( narrow.hash == wide.hash );
		ENSURE( narrow.queryHitCount == wide.queryHitCount );
		ENSURE( narrow.queryHash == wide.queryHash );
	}

	return 0;
}

// Stacks of two slabs, each made of many box shapes, rest on the ground with boxes dropped on top.
// A slab touches the ground or the other slab through more shapes than there are graph colors, so
// most of these contacts overflow. The two slabs of a stack only touch each other, which links the
// overflow contacts of the stack into one group. Run the scene and hash every body. Also reports
// the largest overflow seen.
static uint32_t RunOverflowScene( int workerCount, int* maxOverflowCount )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );

	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
		b3BodyId groundId = b3CreateBody( worldId, &bodyDef );
		b3BoxHull box = b3MakeBoxHull( 100.0f, 1.0f, 100.0f );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateHullShape( groundId, &shapeDef, &box.base );
	}

	enum
	{
		stackCount = 6,
		cellsPerSide = 8,
		boxesPerStack = 4,
		bodyCapacity = stackCount * ( 2 + boxesPerStack ),
	};

	b3BodyId bodyIds[bodyCapacity];
	int bodyCount = 0;

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	for ( int i = 0; i < stackCount; ++i )
	{
		// Separate stacks, so their contacts form separate groups
		float x = 6.0f * ( (float)i - 0.5f * (float)( stackCount - 1 ) );

		// The top slab first, so the shapes of the two slabs are not in the order the slabs stack
		for ( int level = 1; level >= 0; --level )
		{
			b3BodyDef bodyDef = b3DefaultBodyDef();
			bodyDef.type = b3_dynamicBody;
			bodyDef.position = (b3Pos){ x, 0.3f + 0.55f * (float)level, 0.0f };
			b3BodyId slabId = b3CreateBody( worldId, &bodyDef );
			bodyIds[bodyCount++] = slabId;

			for ( int j = 0; j < cellsPerSide * cellsPerSide; ++j )
			{
				b3Vec3 center = { 0.5f * ( (float)( j % cellsPerSide ) - 3.5f ), 0.0f, 0.5f * ( (float)( j / cellsPerSide ) - 3.5f ) };
				b3BoxHull cell = b3MakeOffsetBoxHull( 0.25f, 0.25f, 0.25f, center );
				b3CreateHullShape( slabId, &shapeDef, &cell.base );
			}
		}

		for ( int j = 0; j < boxesPerStack; ++j )
		{
			b3BodyDef bodyDef = b3DefaultBodyDef();
			bodyDef.type = b3_dynamicBody;
			bodyDef.position = (b3Pos){ x + 0.4f * (float)j - 0.6f, 2.0f + 0.7f * (float)j, 0.3f * (float)( j % 2 ) };
			b3BodyId boxId = b3CreateBody( worldId, &bodyDef );
			b3BoxHull box = b3MakeBoxHull( 0.3f, 0.3f, 0.3f );
			b3CreateHullShape( boxId, &shapeDef, &box.base );
			bodyIds[bodyCount++] = boxId;
		}
	}

	*maxOverflowCount = 0;
	for ( int i = 0; i < 90; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
		b3Counters counters = b3World_GetCounters( worldId );
		*maxOverflowCount = b3MaxInt( *maxOverflowCount, counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1] );
	}

	uint32_t hash = B3_HASH_INIT;
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3WorldTransform transform = b3Body_GetTransform( bodyIds[i] );
		hash = b3Hash( hash, (const uint8_t*)&transform, sizeof( b3WorldTransform ) );
	}

	b3DestroyWorld( worldId );
	return hash;
}

// Overflow contacts that share no dynamic body are solved in parallel, in groups. That must give
// exactly what one worker gives by solving them one after the other.
static int OverflowGroupTest( void )
{
	int serialOverflow = 0;
	uint32_t serialHash = RunOverflowScene( 1, &serialOverflow );

	// Enough overflow for the groups to be solved in parallel
	ENSURE( serialOverflow >= 128 );

	for ( int workerCount = 2; workerCount <= 4; ++workerCount )
	{
		int overflow = 0;
		uint32_t hash = RunOverflowScene( workerCount, &overflow );
		ENSURE( overflow == serialOverflow );
		ENSURE( hash == serialHash );
	}

	return 0;
}

int DeterminismTest( void )
{
	RUN_SUBTEST( MultithreadingTest );
	RUN_SUBTEST( CrossPlatformTest );
	RUN_SUBTEST( WavePileTest );
	RUN_SUBTEST( QuerySpawnTest );
	RUN_SUBTEST( MeshDropTest );
	RUN_SUBTEST( SIMDWidthTest );
	RUN_SUBTEST( OverflowGroupTest );

	return 0;
}
