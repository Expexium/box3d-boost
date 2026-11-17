// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

#include "destruction_scenes.h"
#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/destruction.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float s_timeStep = 1.0f / 60.0f;
static const int s_subStepCount = 4;

static b3WorldId CreateTestWorld( int workerCount )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.workerCount = (uint32_t)workerCount;
	return b3CreateWorld( &worldDef );
}

// Ground plate with its top surface at y = 0
static b3BodyId CreateGround( b3WorldId worldId )
{
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
	b3BodyId groundId = b3CreateBody( worldId, &bodyDef );

	b3BoxHull box = b3MakeBoxHull( 100.0f, 1.0f, 100.0f );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateHullShape( groundId, &shapeDef, &box.base );
	return groundId;
}

static void StepWorld( b3WorldId worldId, int stepCount )
{
	for ( int i = 0; i < stepCount; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );
	}
}

// A stack of unit cubes along y, starting at the given height
static b3StructureData* CreateStack( int count, float bottom, bool anchorFirst, const b3StructureMaterial* material )
{
	DestructionBuilder* builder = CreateDestructionBuilder( 1 );
	if ( material != NULL )
	{
		AddDestructionMaterial( builder, material );
	}

	for ( int i = 0; i < count; ++i )
	{
		b3BoxHull box = b3MakeOffsetBoxHull( 0.5f, 0.5f, 0.5f, (b3Vec3){ 0.0f, bottom + 0.5f + (float)i, 0.0f } );
		AddFragment( builder, &box.base, B3_NULL_INDEX, 0, anchorFirst && i == 0 );
	}

	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	return structure;
}

// The cook must find the surfaces neighbors share and get their area, centroid, and normal right.
static int StructureCookTest( void )
{
	// Four boxes in a row along x, each 0.5 long with a 0.4 by 0.3 cross section
	b3StructureMaterial material = b3DefaultStructureMaterial();
	b3StructureData* beam = CreateBeamStructure( 2.0f, 0.4f, 0.3f, 4, false, &material );
	ENSURE( beam != NULL );
	ENSURE( beam->version == B3_STRUCTURE_VERSION );
	ENSURE( beam->fragmentCount == 4 );
	ENSURE( beam->supportCount == 4 );
	ENSURE( beam->bondCount == 3 );
	ENSURE( beam->hullCount == 4 );

	const b3StructureBond* bonds = b3GetStructureBonds( beam );
	for ( int i = 0; i < 3; ++i )
	{
		ENSURE( bonds[i].fragmentA == i );
		ENSURE( bonds[i].fragmentB == i + 1 );
		ENSURE_SMALL( bonds[i].area - 0.4f * 0.3f, 1.0e-4f );
		ENSURE_SMALL( bonds[i].centroid.x - 0.5f * (float)( i + 1 ), 1.0e-4f );
		ENSURE_SMALL( bonds[i].centroid.y, 1.0e-4f );
		ENSURE_SMALL( bonds[i].centroid.z, 1.0e-4f );
		ENSURE_SMALL( bonds[i].normal.x - 1.0f, 1.0e-4f );
	}

	const b3StructureFragment* fragments = b3GetStructureFragments( beam );
	float expectedMass = material.density * 0.5f * 0.4f * 0.3f;
	for ( int i = 0; i < 4; ++i )
	{
		ENSURE_SMALL( fragments[i].mass - expectedMass, 1.0e-2f );
		ENSURE_SMALL( fragments[i].center.x - ( 0.25f + 0.5f * (float)i ), 1.0e-4f );
		ENSURE( fragments[i].parent == B3_NULL_INDEX );
		ENSURE( fragments[i].bondCount == ( i == 0 || i == 3 ? 1 : 2 ) );
	}

	ENSURE_SMALL( beam->mass - 4.0f * expectedMass, 0.1f );

	// The block has no pointers, so a copy is a valid structure
	void* copy = NULL;
#if defined( _MSC_VER )
	copy = _aligned_malloc( (size_t)beam->byteCount, 32 );
#else
	copy = aligned_alloc( 32, ( (size_t)beam->byteCount + 31 ) & ~(size_t)31 );
#endif
	memcpy( copy, beam, (size_t)beam->byteCount );
	const b3StructureData* loaded = b3ValidateStructure( copy, beam->byteCount );
	ENSURE( loaded != NULL );
	ENSURE( loaded->hash == beam->hash );
	ENSURE( b3GetStructureHull( loaded, 0 )->version == B3_HULL_VERSION );
	ENSURE( b3ValidateStructure( copy, beam->byteCount - 8 ) == NULL );
#if defined( _MSC_VER )
	_aligned_free( copy );
#else
	free( copy );
#endif

	b3DestroyStructure( beam );

	// Cooking is deterministic
	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	fracture.tilt = 0.25f;
	b3StructureData* wall1 = CreateWallStructure( 4.0f, 2.0f, 0.3f, &fracture, 7 );
	b3StructureData* wall2 = CreateWallStructure( 4.0f, 2.0f, 0.3f, &fracture, 7 );
	ENSURE( wall1 != NULL && wall2 != NULL );
	ENSURE( wall1->hash == wall2->hash );
	ENSURE( wall1->byteCount == wall2->byteCount );

	// Leaning cuts still give a connected wall whose fragments add up to the volume of the box
	ENSURE( wall1->supportCount > 8 );
	ENSURE( wall1->bondCount >= wall1->supportCount - 1 );
	ENSURE( wall1->fragmentCount > wall1->supportCount );
	ENSURE_SMALL( wall1->mass - material.density * 4.0f * 2.0f * 0.3f, 2.0f );

	// Children add up to the mass of their parent
	const b3StructureFragment* wallFragments = b3GetStructureFragments( wall1 );
	const int32_t* children = b3GetStructureChildren( wall1 );
	for ( int i = 0; i < wall1->fragmentCount; ++i )
	{
		const b3StructureFragment* fragment = wallFragments + i;
		if ( fragment->childCount == 0 )
		{
			continue;
		}

		float childMass = 0.0f;
		for ( int j = 0; j < fragment->childCount; ++j )
		{
			int child = children[fragment->firstChild + j];
			ENSURE( wallFragments[child].parent == i );
			childMass += wallFragments[child].mass;
		}

		ENSURE_SMALL( childMass - fragment->mass, 1.0e-3f * fragment->mass );
		ENSURE( fragment->health > 0.0f );
	}

	b3DestroyStructure( wall1 );
	b3DestroyStructure( wall2 );
	return 0;
}

// A column standing on the ground: every bond must carry the weight of what is above it.
static int ColumnStressTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	// Make the column strong enough that nothing breaks
	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.strengthVariation = 0.0f;
	const int count = 8;
	b3StructureData* structure = CreateStack( count, 0.0f, false, &material );
	ENSURE( structure->bondCount == count - 1 );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	ENSURE( b3Destructible_IsValid( destructibleId ) );

	// The bottom cube touches the ground, so it gets an anchor and the column is one static piece
	ENSURE( b3Destructible_GetBondCount( destructibleId ) == count );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );
	b3BodyId bodyId = b3Destructible_GetFragmentBody( destructibleId, 0 );
	ENSURE( b3Body_GetType( bodyId ) == b3_staticBody );
	ENSURE( B3_ID_EQUALS( b3Body_GetDestructible( bodyId ), destructibleId ) );

	StepWorld( worldId, 30 );

	// Compressive stress in the bond below cube i+1 is the weight of the cubes above over the unit area
	float cubeWeight = material.density * 10.0f;
	for ( int i = 0; i < count - 1; ++i )
	{
		b3BondState state = b3Destructible_GetBondState( destructibleId, i );
		ENSURE( state.isBroken == false );
		ENSURE( state.fragmentA == i && state.fragmentB == i + 1 );

		float expectedStress = cubeWeight * (float)( count - 1 - i );
		float expectedRatio = expectedStress / material.compressiveFatalLimit;
		ENSURE_SMALL( state.fatalRatio - expectedRatio, 0.02f * expectedRatio + 1.0e-6f );
		ENSURE_SMALL( state.health - 1.0f, 1.0e-6f );
	}

	// The anchor carries the whole column
	b3BondState anchor = b3Destructible_GetBondState( destructibleId, count - 1 );
	ENSURE( anchor.fragmentA == 0 && anchor.fragmentB == B3_NULL_INDEX );
	ENSURE_SMALL( anchor.area - 1.0f, 1.0e-3f );
	float anchorRatio = cubeWeight * (float)count / material.compressiveFatalLimit;
	ENSURE_SMALL( anchor.fatalRatio - anchorRatio, 0.02f * anchorRatio );

	// Nothing moved and nothing broke
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == count );
	ENSURE( b3World_GetDestructionEvents( worldId ).fragmentCount == 0 );

	// A solved structure at rest costs nothing
	b3DestructionCounters counters = b3World_GetDestructionCounters( worldId );
	ENSURE( counters.destructibleCount == 1 );
	ENSURE( counters.solvedPieceCount == 0 );

	b3DestroyDestructible( destructibleId );
	ENSURE( b3Destructible_IsValid( destructibleId ) == false );
	ENSURE( b3Body_IsValid( bodyId ) == false );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// Run a cantilever of the given length. Returns the index of the first bond that broke, or -1.
static int RunCantilever( int cellCount, float* rootRatio )
{
	b3WorldId worldId = CreateTestWorld( 1 );

	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.strengthVariation = 0.0f;

	// Cells of 0.5 m with a 0.3 m square section, fixed at the first cell and sticking out along x
	float cellLength = 0.5f;
	b3StructureData* structure = CreateBeamStructure( cellLength * (float)cellCount, 0.3f, 0.3f, cellCount, true, &material );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.position = (b3Pos){ 0.0f, 10.0f, 0.0f };
	def.anchorToStatic = false;
	def.enableBondEvents = true;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	int firstBroken = -1;
	*rootRatio = 0.0f;

	for ( int i = 0; i < 240 && firstBroken == -1; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );

		b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
		if ( events.bondCount > 0 )
		{
			firstBroken = events.bondEvents[0].bondIndex;
		}
		else
		{
			*rootRatio = b3MaxFloat( *rootRatio, b3Destructible_GetBondState( destructibleId, 1 ).fatalRatio );
		}
	}

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return firstBroken;
}

// A cantilever breaks at its root once it is long enough. The bending stress at the root must match
// beam theory while it holds.
static int CantileverTest( void )
{
	b3StructureMaterial material = b3DefaultStructureMaterial();

	// Bond 0 is the rigid anchor of the first cell. Bond 1 is the root: the first cell to the second.
	// Root moment = weight of the hanging cells times the distance of their center from the root.
	float side = 0.3f;
	float cellLength = 0.5f;
	float area = side * side;
	float cellWeight = material.density * cellLength * area * 10.0f;

	{
		int cellCount = 4;
		float rootRatio = 0.0f;
		int broken = RunCantilever( cellCount, &rootRatio );
		ENSURE( broken == -1 );

		float hanging = (float)( cellCount - 1 );
		float moment = hanging * cellWeight * 0.5f * hanging * cellLength;
		float tension = 6.0f * moment / ( area * sqrtf( area ) );
		float expectedRatio = tension / material.tensileFatalLimit;
		ENSURE( expectedRatio < 0.5f );
		ENSURE_SMALL( rootRatio - expectedRatio, 0.03f * expectedRatio );
	}

	{
		// Long enough that the root tension passes the fatal limit: it must snap at the root
		int cellCount = 14;
		float hanging = (float)( cellCount - 1 );
		float moment = hanging * cellWeight * 0.5f * hanging * cellLength;
		float tension = 6.0f * moment / ( area * sqrtf( area ) );
		ENSURE( tension > material.tensileFatalLimit );

		float rootRatio = 0.0f;
		int broken = RunCantilever( cellCount, &rootRatio );
		ENSURE( broken == 1 );
	}

	return 0;
}

// Breaking the bond that holds a piece up must turn it into a falling body that keeps its fragments.
static int SeveredPieceTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );

	b3StructureData* structure = CreateBeamStructure( 3.0f, 0.4f, 0.4f, 6, true, NULL );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.position = (b3Pos){ 0.0f, 20.0f, 0.0f };
	def.anchorToStatic = false;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	StepWorld( worldId, 10 );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	// Find the bond between cells 2 and 3 and cut it
	int cut = -1;
	for ( int i = 0; i < b3Destructible_GetBondCount( destructibleId ); ++i )
	{
		b3BondState state = b3Destructible_GetBondState( destructibleId, i );
		if ( state.fragmentA == 2 && state.fragmentB == 3 )
		{
			cut = i;
		}
	}
	ENSURE( cut != -1 );

	b3Destructible_BreakBond( destructibleId, cut );
	b3World_Step( worldId, s_timeStep, s_subStepCount );

	// Cells 3, 4, 5 moved to a new dynamic body. Cells 0, 1, 2 stayed on the static body.
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 2 );
	ENSURE( b3Destructible_GetBondState( destructibleId, cut ).isBroken );

	b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
	ENSURE( events.bodyCount == 1 );
	ENSURE( events.bodyEvents[0].fragmentCount == 3 );
	ENSURE( events.fragmentCount == 3 );
	for ( int i = 0; i < events.fragmentCount; ++i )
	{
		ENSURE( events.fragmentEvents[i].type == b3_fragmentMoved );
		ENSURE( events.fragmentEvents[i].fragmentIndex >= 3 );
		ENSURE( B3_ID_EQUALS( events.fragmentEvents[i].bodyId, events.bodyEvents[0].bodyId ) );
	}

	b3BodyId staticId = b3Destructible_GetFragmentBody( destructibleId, 0 );
	b3BodyId fallingId = b3Destructible_GetFragmentBody( destructibleId, 5 );
	ENSURE( B3_ID_EQUALS( staticId, b3Destructible_GetFragmentBody( destructibleId, 2 ) ) );
	ENSURE( B3_ID_EQUALS( fallingId, b3Destructible_GetFragmentBody( destructibleId, 3 ) ) );
	ENSURE( b3Body_GetType( staticId ) == b3_staticBody );
	ENSURE( b3Body_GetType( fallingId ) == b3_dynamicBody );
	ENSURE( b3Body_GetShapeCount( staticId ) == 3 );
	ENSURE( b3Body_GetShapeCount( fallingId ) == 3 );

	// The new body has the mass and center of its three cells
	b3StructureMaterial material = b3DefaultStructureMaterial();
	float cellMass = material.density * 0.5f * 0.4f * 0.4f;
	ENSURE_SMALL( b3Body_GetMass( fallingId ) - 3.0f * cellMass, 0.01f * cellMass );
	ENSURE_SMALL( b3Body_GetLocalCenter( fallingId ).x - 2.25f, 1.0e-3f );

	// Shapes map back to their fragment
	b3ShapeId shapeIds[3];
	ENSURE( b3Body_GetShapes( fallingId, shapeIds, 3 ) == 3 );
	for ( int i = 0; i < 3; ++i )
	{
		ENSURE( B3_ID_EQUALS( b3Shape_GetDestructible( shapeIds[i] ), destructibleId ) );
		ENSURE( b3Shape_GetFragmentIndex( shapeIds[i] ) >= 3 );
	}

	float y0 = (float)b3Body_GetPosition( fallingId ).y;
	StepWorld( worldId, 30 );
	float y1 = (float)b3Body_GetPosition( fallingId ).y;

	// Half a second of free fall is 1.25 m
	ENSURE_SMALL( ( y0 - y1 ) - 1.25f, 0.15f );
	ENSURE_SMALL( (float)b3Body_GetPosition( staticId ).y - 20.0f, 1.0e-5f );

	// A falling piece carries no stress and needs no solve
	b3DestructionCounters counters = b3World_GetDestructionCounters( worldId );
	ENSURE( counters.solvedPieceCount == 0 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// A piece that splits in flight must keep its momentum.
static int SplitMomentumTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	b3World_SetGravity( worldId, b3Vec3_zero );

	b3StructureData* structure = CreateBeamStructure( 4.0f, 0.5f, 0.5f, 4, false, NULL );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.position = (b3Pos){ 0.0f, 10.0f, 0.0f };
	def.anchorToStatic = false;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	// A free beam is a single dynamic body
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );
	b3BodyId bodyId = b3Destructible_GetFragmentBody( destructibleId, 0 );
	ENSURE( b3Body_GetType( bodyId ) == b3_dynamicBody );

	// Send it tumbling about the z-axis
	b3Vec3 linearVelocity = { 2.0f, 1.0f, 0.5f };
	b3Vec3 angularVelocity = { 0.0f, 0.0f, 1.5f };
	b3Body_SetLinearVelocity( bodyId, linearVelocity );
	b3Body_SetAngularVelocity( bodyId, angularVelocity );
	float totalMass = b3Body_GetMass( bodyId );

	StepWorld( worldId, 5 );

	b3Vec3 momentumBefore = b3MulSV( totalMass, b3Body_GetLinearVelocity( bodyId ) );

	// Cut it in the middle
	int cut = -1;
	for ( int i = 0; i < b3Destructible_GetBondCount( destructibleId ); ++i )
	{
		b3BondState state = b3Destructible_GetBondState( destructibleId, i );
		if ( state.fragmentA == 1 && state.fragmentB == 2 )
		{
			cut = i;
		}
	}
	ENSURE( cut != -1 );
	b3Destructible_BreakBond( destructibleId, cut );
	b3World_Step( worldId, s_timeStep, s_subStepCount );

	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 2 );
	b3BodyId bodyA = b3Destructible_GetFragmentBody( destructibleId, 0 );
	b3BodyId bodyB = b3Destructible_GetFragmentBody( destructibleId, 3 );
	ENSURE( B3_ID_EQUALS( bodyA, bodyB ) == false );

	float massA = b3Body_GetMass( bodyA );
	float massB = b3Body_GetMass( bodyB );
	ENSURE_SMALL( massA + massB - totalMass, 1.0e-2f );

	b3Vec3 momentumAfter =
		b3Add( b3MulSV( massA, b3Body_GetLinearVelocity( bodyA ) ), b3MulSV( massB, b3Body_GetLinearVelocity( bodyB ) ) );
	ENSURE_SMALL( momentumAfter.x - momentumBefore.x, 1.0e-3f * totalMass );
	ENSURE_SMALL( momentumAfter.y - momentumBefore.y, 1.0e-3f * totalMass );
	ENSURE_SMALL( momentumAfter.z - momentumBefore.z, 1.0e-3f * totalMass );

	// Both halves keep the spin, and their centers fly apart the way the ends of the beam were moving
	ENSURE_SMALL( b3Body_GetAngularVelocity( bodyA ).z - 1.5f, 1.0e-3f );
	ENSURE_SMALL( b3Body_GetAngularVelocity( bodyB ).z - 1.5f, 1.0e-3f );
	b3Vec3 relative = b3Sub( b3Body_GetLinearVelocity( bodyB ), b3Body_GetLinearVelocity( bodyA ) );
	ENSURE_SMALL( b3Length( relative ) - 1.5f * 2.0f, 0.02f );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// Count the bodies a boulder ends up as after a drop from the given height.
static int DropBoulder( float height, int* removedCount )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	b3StructureData* structure = CreateBoulderStructure( 0.8f, 0.5f, NULL, 3 );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.position = (b3Pos){ 0.0f, height + 1.0f, 0.0f };
	def.anchorToStatic = false;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	*removedCount = 0;
	for ( int i = 0; i < 300; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );

		b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
		for ( int j = 0; j < events.fragmentCount; ++j )
		{
			*removedCount += events.fragmentEvents[j].type == b3_fragmentRemoved ? 1 : 0;
		}
	}

	int bodyCount = b3Destructible_GetBodyCount( destructibleId );
	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return bodyCount;
}

// A boulder survives a short drop and shatters into its debris after a long one.
static int BoulderTest( void )
{
	int removedCount = 0;
	ENSURE( DropBoulder( 2.0f, &removedCount ) == 1 );
	ENSURE( removedCount == 0 );

	ENSURE( DropBoulder( 20.0f, &removedCount ) > 3 );
	ENSURE( removedCount == 1 );
	return 0;
}

// A slab on four columns. Crushing two columns on one side must bring the slab down, and it must
// come down because the bonds were overloaded, not because anything was scripted.
static int FrameCollapseTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	BuildingDef buildingDef = DefaultBuildingDef();
	buildingDef.floorCount = 1;
	buildingDef.bayCountX = 1;
	buildingDef.bayCountZ = 1;
	buildingDef.enableWalls = false;
	buildingDef.fracture.childSize = 0.0f;
	b3StructureData* structure = CreateBuildingStructure( &buildingDef );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.enableBondEvents = true;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	// Everything hangs together on the static body
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	// It stands on its own
	StepWorld( worldId, 120 );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == b3Destructible_GetBondCount( destructibleId ) );

	// Crush the fragments of the two columns at negative x
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	int crushed = 0;
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + i;
		bool isColumn = fragment->aabb.upperBound.y < buildingDef.floorHeight - buildingDef.slabThickness + 0.01f;
		if ( isColumn && fragment->center.x < 0.0f && fragment->center.y < 1.5f )
		{
			b3Destructible_DamageFragment( destructibleId, i, 10.0f * fragment->health );
			crushed += 1;
		}
	}
	ENSURE( crushed >= 2 );

	// The slab now hangs off the two remaining columns. Its bonds overload, and it falls.
	int stressBreaks = 0;
	int firstStressBreakStep = -1;
	for ( int i = 0; i < 360; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );

		b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
		if ( i > 0 && events.bondCount > 0 )
		{
			stressBreaks += events.bondCount;
			if ( firstStressBreakStep == -1 )
			{
				firstStressBreakStep = i;
			}
		}
	}

	// Bonds broke after the step that crushed the columns: those came from the stress solve
	ENSURE( stressBreaks > 0 );
	ENSURE( firstStressBreakStep >= 1 );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) > 1 );

	// Something that used to be at slab height is now on the ground
	b3BodyId bodies[256];
	int bodyCount = b3Destructible_GetBodies( destructibleId, bodies, 256 );
	float lowest = 100.0f;
	for ( int i = 0; i < bodyCount; ++i )
	{
		if ( b3Body_GetType( bodies[i] ) == b3_dynamicBody )
		{
			lowest = b3MinFloat( lowest, (float)b3Body_GetWorldCenter( bodies[i] ).y );
		}
	}
	ENSURE( lowest < 1.5f );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// Hash the transform of every body in the world
static uint32_t HashWorld( b3WorldId worldId, b3DestructibleId destructibleId )
{
	b3BodyId bodies[4096];
	int bodyCount = b3Destructible_GetBodies( destructibleId, bodies, 4096 );
	uint32_t hash = B3_HASH_INIT;
	hash = b3Hash( hash, (const uint8_t*)&bodyCount, sizeof( int ) );
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3WorldTransform transform = b3Body_GetTransform( bodies[i] );
		hash = b3Hash( hash, (const uint8_t*)&transform, sizeof( b3WorldTransform ) );
	}

	int bondCount = b3Destructible_GetIntactBondCount( destructibleId );
	hash = b3Hash( hash, (const uint8_t*)&bondCount, sizeof( int ) );
	MAYBE_UNUSED( worldId );
	return hash;
}

static uint32_t RunCollapse( int workerCount, bool enableStress, int* bodyCount, b3Recording* recording )
{
	b3WorldId worldId = CreateTestWorld( workerCount );
	CreateGround( worldId );

	BuildingDef buildingDef = DefaultBuildingDef();
	buildingDef.floorCount = 2;
	buildingDef.bayCountX = 2;
	buildingDef.bayCountZ = 1;
	buildingDef.fracture.cellSize = 1.25f;
	buildingDef.fracture.childSize = 0.6f;
	b3StructureData* structure = CreateBuildingStructure( &buildingDef );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.enableStress = enableStress;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	if ( recording != NULL )
	{
		b3World_StartRecording( worldId, recording );
	}

	for ( int i = 0; i < 240; ++i )
	{
		if ( i == 20 )
		{
			ApplyBlast( worldId, (b3Pos){ -5.0f, 1.0f, -2.5f }, 3.0f, 4.0e6f, 4000.0f );
			ApplyBlast( worldId, (b3Pos){ -5.0f, 1.0f, 2.5f }, 3.0f, 4.0e6f, 4000.0f );
		}

		b3World_Step( worldId, s_timeStep, s_subStepCount );
	}

	if ( recording != NULL )
	{
		b3World_StopRecording( worldId );
	}

	*bodyCount = b3Destructible_GetBodyCount( destructibleId );
	uint32_t hash = HashWorld( worldId, destructibleId );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return hash;
}

// A collapse must give the same result every time, on any number of workers.
static int CollapseDeterminismTest( void )
{
	// With and without the stress solve
	for ( int mode = 0; mode < 2; ++mode )
	{
		bool enableStress = mode == 0;
		int bodyCount1 = 0, bodyCount2 = 0, bodyCount4 = 0;
		uint32_t hash1 = RunCollapse( 1, enableStress, &bodyCount1, NULL );
		uint32_t hash2 = RunCollapse( 1, enableStress, &bodyCount2, NULL );
		uint32_t hash4 = RunCollapse( 4, enableStress, &bodyCount4, NULL );

		// The blast must have done something for this to mean anything
		ENSURE( bodyCount1 > 4 );
		ENSURE( bodyCount1 == bodyCount2 );
		ENSURE( hash1 == hash2 );
		ENSURE( bodyCount1 == bodyCount4 );
		ENSURE( hash1 == hash4 );
	}

	return 0;
}

// The primary scene: a building stands until charges take out its ground floor columns, then it comes
// down and ends as rubble on the ground. This is the benchmark scene at a small scale.
static int BuildingDemolitionTest( void )
{
	b3WorldId worldId = CreateTestWorld( 4 );
	SetBuildingCollapseScale( 1, 2 );
	CreateBuildingCollapse( worldId );

	b3DestructibleId buildingId = GetBuildingCollapseBuilding( 0 );
	ENSURE( b3Destructible_IsValid( buildingId ) );

	const b3StructureData* structure = b3Destructible_GetStructure( buildingId );
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	int bondCount = b3Destructible_GetBondCount( buildingId );
	float height = structure->aabb.upperBound.y;
	ENSURE( structure->supportCount > 500 );

	// The building carries itself: nothing breaks while it waits for the charges
	int stepIndex = 0;
	for ( ; stepIndex < 30; ++stepIndex )
	{
		StepBuildingCollapse( worldId, stepIndex );
		b3World_Step( worldId, s_timeStep, s_subStepCount );
	}

	ENSURE( b3Destructible_GetIntactBondCount( buildingId ) == bondCount );
	ENSURE( b3Destructible_GetBodyCount( buildingId ) == 1 );

	// The charges go off and the building comes down
	float maxStepTime = 0.0f;
	for ( ; stepIndex < 450; ++stepIndex )
	{
		StepBuildingCollapse( worldId, stepIndex );
		b3World_Step( worldId, s_timeStep, s_subStepCount );

		b3DestructionCounters counters = b3World_GetDestructionCounters( worldId );
		ENSURE( counters.bodyCount == b3Destructible_GetBodyCount( buildingId ) );
		ENSURE( counters.bondCount == b3Destructible_GetIntactBondCount( buildingId ) );
		maxStepTime = b3MaxFloat( maxStepTime, b3World_GetDestructionProfile( worldId ).total );
	}

	// It broke into many bodies and most of its bonds are gone
	int bodyCount = b3Destructible_GetBodyCount( buildingId );
	int intactCount = b3Destructible_GetIntactBondCount( buildingId );
	ENSURE( bodyCount > 500 );
	ENSURE( intactCount < bondCount / 2 );
	ENSURE( b3World_GetDestructionCounters( worldId ).pendingSplitCount == 0 );

	// The building is down: nearly all of its mass lies low, on the ground or on the pile. A strip of
	// wall may still stand and a fragment may still be in the air, so this looks at the mass and not
	// at the highest point. No fragment left the scene or fell through the ground.
	float totalMass = 0.0f;
	float highMass = 0.0f;
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		b3BodyId bodyId = b3Destructible_GetFragmentBody( buildingId, i );
		if ( B3_IS_NULL( bodyId ) )
		{
			continue;
		}

		b3Pos center = b3TransformWorldPoint( b3Body_GetTransform( bodyId ), fragments[i].center );
		ENSURE( b3IsValidFloat( (float)center.x ) && b3IsValidFloat( (float)center.y ) && b3IsValidFloat( (float)center.z ) );
		ENSURE( (float)center.y > -0.5f );
		ENSURE( b3AbsFloat( (float)center.x ) < 60.0f && b3AbsFloat( (float)center.z ) < 60.0f );

		totalMass += fragments[i].mass;
		if ( (float)center.y > 0.5f * height )
		{
			highMass += fragments[i].mass;
		}
	}

	ENSURE( totalMass > 0.0f );
	ENSURE( highMass < 0.1f * totalMass );
	MAYBE_UNUSED( maxStepTime );

	DestroyBuildingCollapse();
	b3DestroyWorld( worldId );
	return 0;
}

// An impact can only break as much bond area as it has energy for. A plate falls flat onto a narrow
// pedestal, which bends it hard, and the bonds the blows broke are added up against the fall.
static int ImpactEnergyTest( void )
{
	// Tough enough that the fall cannot pay for all the bonds. The stress of the landing alone
	// would break most of them.
	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.toughness = 3.0e5f;
	material.strengthVariation = 0.0f;

	DestructionBuilder* builder = CreateDestructionBuilder( 5 );
	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.0f;
	fracture.materialIndex = AddDestructionMaterial( builder, &material );
	b3Vec3 halfExtents = { 2.0f, 0.15f, 1.0f };
	AddFracturedBox( builder, b3Transform_identity, halfExtents, &fracture );
	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	ENSURE( structure != NULL );

	const b3StructureBond* bonds = b3GetStructureBonds( structure );
	float totalArea = 0.0f;
	for ( int i = 0; i < structure->bondCount; ++i )
	{
		totalArea += bonds[i].area;
	}

	const float pedestalTop = 1.0f;
	const float dropHeight = 2.0f;
	const float startHeight = pedestalTop + halfExtents.y + dropHeight;

	for ( int pass = 0; pass < 2; ++pass )
	{
		b3WorldId worldId = CreateTestWorld( 1 );
		b3BodyId groundId = CreateGround( worldId );

		// The ground body sits one meter below the surface
		b3BoxHull pedestal = b3MakeOffsetBoxHull( 0.25f, 0.5f * pedestalTop, 1.5f, (b3Vec3){ 0.0f, 1.0f + 0.5f * pedestalTop, 0.0f } );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateHullShape( groundId, &shapeDef, &pedestal.base );

		// In the second pass the blows have no energy to spend on bonds
		b3DestructionSettings settings = b3World_GetDestructionSettings( worldId );
		if ( pass == 1 )
		{
			settings.impactFractureScale = 0.0f;
		}
		b3World_SetDestructionSettings( worldId, &settings );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		def.position = (b3Pos){ 0.0f, startHeight, 0.0f };
		def.anchorToStatic = false;
		def.enableBondEvents = true;
		b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
		ENSURE( b3Body_GetType( b3Destructible_GetFragmentBody( destructibleId, 0 ) ) == b3_dynamicBody );

		float impactArea = 0.0f;
		int impactCount = 0;
		for ( int i = 0; i < 180; ++i )
		{
			b3World_Step( worldId, s_timeStep, s_subStepCount );

			b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
			for ( int j = 0; j < events.bondCount; ++j )
			{
				if ( events.bondEvents[j].cause == b3_bondBreakImpact )
				{
					impactArea += events.bondEvents[j].area;
					impactCount += 1;
				}
			}
		}

		if ( pass == 0 )
		{
			// No part of the plate fell further than from where it started to the ground
			float energy = settings.impactFractureScale * structure->mass * 10.0f * ( startHeight + halfExtents.y );
			ENSURE( impactCount > 0 );
			ENSURE( material.toughness * impactArea <= energy );
			ENSURE( impactArea < 0.5f * totalArea );
		}
		else
		{
			ENSURE( impactCount == 0 );
		}

		b3DestroyWorld( worldId );
	}

	b3DestroyStructure( structure );
	return 0;
}

// Debris is removed when its lifetime ends, and says so.
static int DebrisLifetimeTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 3 );
	ENSURE( structure != NULL );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.debrisLifetime = 0.5f;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	StepWorld( worldId, 10 );
	ApplyBlast( worldId, (b3Pos){ 0.0f, 1.5f, 0.4f }, 1.2f, 2.0e5f, 8000.0f );

	int peakBodyCount = 0;
	int removedCount = 0;
	for ( int i = 0; i < 240; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );
		peakBodyCount = b3MaxInt( peakBodyCount, b3Destructible_GetBodyCount( destructibleId ) );

		b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
		for ( int j = 0; j < events.fragmentCount; ++j )
		{
			const b3FragmentEvent* event = events.fragmentEvents + j;
			if ( event->type == b3_fragmentRemoved )
			{
				// A removed fragment has no body left
				ENSURE( B3_IS_NULL( b3Destructible_GetFragmentBody( destructibleId, event->fragmentIndex ) ) );
				removedCount += 1;
			}
		}
	}

	// The blast made debris and the debris is gone again. The wall around the hole is not debris.
	ENSURE( peakBodyCount > 10 );
	ENSURE( removedCount > 10 );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) < peakBodyCount / 2 );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) >= 1 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// The strength scale makes a structure weaker at run time. A frame that stands at full strength
// fails under its own weight once the scale is low enough.
static int StrengthScaleTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	BuildingDef buildingDef = DefaultBuildingDef();
	buildingDef.floorCount = 2;
	buildingDef.bayCountX = 2;
	buildingDef.bayCountZ = 1;
	buildingDef.fracture.cellSize = 1.25f;
	buildingDef.fracture.childSize = 0.0f;
	b3StructureData* structure = CreateBuildingStructure( &buildingDef );
	ENSURE( structure != NULL );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	int bondCount = b3Destructible_GetBondCount( destructibleId );

	StepWorld( worldId, 120 );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	b3Destructible_SetStrengthScale( destructibleId, 0.01f );
	ENSURE( b3Destructible_GetStrengthScale( destructibleId ) == 0.01f );

	StepWorld( worldId, 240 );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) < bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) > 1 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// A destructible on a moving host goes where the host goes. A steady ride loads nothing, a hard
// acceleration acts like extra weight and breaks what plain gravity could not.
static int HostMotionTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );

	b3BodyDef hostDef = b3DefaultBodyDef();
	hostDef.type = b3_kinematicBody;
	hostDef.position = (b3Pos){ 0.0f, 5.0f, 0.0f };
	b3BodyId hostId = b3CreateBody( worldId, &hostDef );
	b3Body_SetLinearVelocity( hostId, (b3Vec3){ 2.0f, 0.0f, 0.0f } );

	// A short stiff cantilever that carries itself with a wide margin
	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.strengthVariation = 0.0f;
	b3StructureData* structure = CreateBeamStructure( 2.0f, 0.4f, 0.4f, 4, true, &material );
	ENSURE( structure != NULL );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.position = hostDef.position;
	def.hostBodyId = hostId;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	int bondCount = b3Destructible_GetBondCount( destructibleId );

	const int rideSteps = 90;
	StepWorld( worldId, rideSteps );

	// It followed the host and nothing broke
	b3BodyId anchorId = b3Destructible_GetFragmentBody( destructibleId, 0 );
	ENSURE( b3Body_GetType( anchorId ) == b3_kinematicBody );
	b3WorldTransform transform = b3Body_GetTransform( anchorId );
	float traveled = 2.0f * (float)rideSteps * s_timeStep;
	ENSURE( b3AbsFloat( (float)transform.p.x - traveled ) < 0.1f );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	// The host shoots upward. The beam feels many times its weight and snaps.
	b3Vec3 velocity = { 2.0f, 0.0f, 0.0f };
	for ( int i = 0; i < 60; ++i )
	{
		velocity.y += 150.0f * s_timeStep;
		b3Body_SetLinearVelocity( hostId, velocity );
		b3World_Step( worldId, s_timeStep, s_subStepCount );
	}

	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) < bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) > 1 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// Find the support fragment closest to a point in structure space
static int FindFragment( const b3StructureData* structure, b3Vec3 point )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	int best = B3_NULL_INDEX;
	float bestDistance = FLT_MAX;
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		if ( fragments[i].parent != B3_NULL_INDEX )
		{
			continue;
		}

		float distance = b3DistanceSquared( fragments[i].center, point );
		if ( distance < bestDistance )
		{
			bestDistance = distance;
			best = i;
		}
	}

	return best;
}

// Shoot a heavy ball at a wall and return the speed it has left along its path
static float RunPunch( float momentumTransferScale, int* brokenCount )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	b3DestructionSettings settings = b3World_GetDestructionSettings( worldId );
	settings.momentumTransferScale = momentumTransferScale;
	b3World_SetDestructionSettings( worldId, &settings );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.0f;
	b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 11 );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	int bondCount = b3Destructible_GetBondCount( destructibleId );

	StepWorld( worldId, 5 );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 0.0f, 1.5f, 4.0f };
	bodyDef.linearVelocity = (b3Vec3){ 0.0f, 0.0f, -40.0f };
	bodyDef.isBullet = true;
	b3BodyId ballId = b3CreateBody( worldId, &bodyDef );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 7800.0f;
	b3Sphere sphere = { b3Vec3_zero, 0.4f };
	b3CreateSphereShape( ballId, &shapeDef, &sphere );

	StepWorld( worldId, 20 );

	float speed = -b3Body_GetLinearVelocity( ballId ).z;
	*brokenCount = bondCount - b3Destructible_GetIntactBondCount( destructibleId );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return speed;
}

// The rigid body solver stops a ball against a wall as if the wall were whole. The bonds that
// broke under the blow could not have resisted that much, so the solver hands the surplus back
// and the ball punches through instead of stopping dead.
static int MomentumTransferTest( void )
{
	int brokenWith = 0, brokenWithout = 0;
	float speedWith = RunPunch( 1.0f, &brokenWith );
	float speedWithout = RunPunch( 0.0f, &brokenWithout );

	ENSURE( brokenWith > 0 );
	ENSURE( brokenWithout > 0 );
	ENSURE( speedWith > speedWithout + 1.0f );
	return 0;
}

// Debris that falls asleep can turn into static rubble, which costs nothing from then on.
static int FreezeDebrisTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 3 );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.freezeSleepingDebris = true;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	b3BodyId anchorBodyId = b3Destructible_GetFragmentBody( destructibleId, 0 );

	StepWorld( worldId, 10 );
	ApplyBlast( worldId, (b3Pos){ 0.0f, 1.5f, 0.4f }, 1.2f, 2.0e5f, 4000.0f );
	StepWorld( worldId, 900 );

	b3BodyId bodies[2048];
	int bodyCount = b3Destructible_GetBodies( destructibleId, bodies, 2048 );
	int frozenCount = 0;
	for ( int i = 0; i < bodyCount; ++i )
	{
		if ( B3_ID_EQUALS( bodies[i], anchorBodyId ) == false && b3Body_GetType( bodies[i] ) == b3_staticBody )
		{
			frozenCount += 1;
		}
	}

	ENSURE( frozenCount > 10 );

	// Frozen rubble is still the fragment's body
	ENSURE( b3World_GetDestructionCounters( worldId ).bodyCount == bodyCount );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// The solver never creates more bodies in a step than it is allowed. A large break is finished over
// the following steps.
static int BodyBudgetTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	b3DestructionSettings settings = b3World_GetDestructionSettings( worldId );
	settings.maxNewBodiesPerStep = 8;
	b3World_SetDestructionSettings( worldId, &settings );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 3 );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	StepWorld( worldId, 10 );
	ApplyBlast( worldId, (b3Pos){ 0.0f, 1.5f, 0.4f }, 1.5f, 2.0e5f, 8000.0f );

	int waitingSteps = 0;
	int createdCount = 0;
	for ( int i = 0; i < 240; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );

		b3DestructionCounters counters = b3World_GetDestructionCounters( worldId );
		ENSURE( counters.newBodyCount <= settings.maxNewBodiesPerStep );
		createdCount += counters.newBodyCount;

		// Something is waiting: a piece that could not be split yet or a fragment that could not
		// crumble yet. Either way the step used its whole allowance.
		if ( counters.newBodyCount == settings.maxNewBodiesPerStep )
		{
			waitingSteps += 1;
		}
	}


	// The break needed far more bodies than one step allows, and it got them in the end
	ENSURE( createdCount > 4 * settings.maxNewBodiesPerStep );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) > 4 * settings.maxNewBodiesPerStep );
	ENSURE( waitingSteps > 3 );
	ENSURE( b3World_GetDestructionCounters( worldId ).pendingSplitCount == 0 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// An impulse on one fragment. A nudge does nothing, a hard knock tears the fragment out of the wall
// and sends it off in the direction of the push.
static int ApplyImpulseTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.0f;
	b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 11 );
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	int bondCount = b3Destructible_GetBondCount( destructibleId );

	int target = FindFragment( structure, (b3Vec3){ 0.0f, 1.5f, 0.0f } );
	ENSURE( target != B3_NULL_INDEX );
	b3BodyId anchorBodyId = b3Destructible_GetFragmentBody( destructibleId, target );
	b3Pos point = b3ToPos( fragments[target].center );
	float mass = fragments[target].mass;

	StepWorld( worldId, 10 );

	// Enough to move the fragment alone at a few centimeters per second
	b3Destructible_ApplyImpulse( destructibleId, target, (b3Vec3){ 0.0f, 0.0f, -0.05f * mass }, point );
	StepWorld( worldId, 30 );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount );
	ENSURE( B3_ID_EQUALS( b3Destructible_GetFragmentBody( destructibleId, target ), anchorBodyId ) );

	// Enough to move it alone at 40 meters per second
	b3Destructible_ApplyImpulse( destructibleId, target, (b3Vec3){ 0.0f, 0.0f, -40.0f * mass }, point );
	StepWorld( worldId, 5 );

	b3BodyId bodyId = b3Destructible_GetFragmentBody( destructibleId, target );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) < bondCount );
	ENSURE( B3_IS_NON_NULL( bodyId ) );
	ENSURE( B3_ID_EQUALS( bodyId, anchorBodyId ) == false );
	ENSURE( b3Body_GetLinearVelocity( bodyId ).z < -5.0f );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// Radial damage can spread along the bonds or straight through space. Two walls stand close
// together without touching. Damage that follows the bonds stays in the wall it hit. Damage that
// goes through space reaches the other wall too.
static int RadialDamageTest( void )
{
	DestructionBuilder* builder = CreateDestructionBuilder( 13 );
	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	b3Vec3 halfExtents = { 2.0f, 1.5f, 0.15f };
	b3Transform nearWall = { { 0.0f, 1.5f, 0.0f }, b3Quat_identity };
	b3Transform farWall = { { 0.0f, 1.5f, 1.0f }, b3Quat_identity };
	AddFracturedBox( builder, nearWall, halfExtents, &fracture );
	AddFracturedBox( builder, farWall, halfExtents, &fracture );
	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	ENSURE( structure != NULL );

	int nearFragment = FindFragment( structure, (b3Vec3){ 0.0f, 1.5f, 0.0f } );
	int farFragment = FindFragment( structure, (b3Vec3){ 0.0f, 1.5f, 1.0f } );
	ENSURE( nearFragment != farFragment );

	for ( int pass = 0; pass < 2; ++pass )
	{
		b3WorldId worldId = CreateTestWorld( 1 );
		CreateGround( worldId );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
		StepWorld( worldId, 5 );

		float nearHealth = b3Destructible_GetFragmentHealth( destructibleId, nearFragment );
		float farHealth = b3Destructible_GetFragmentHealth( destructibleId, farFragment );
		ENSURE( nearHealth > 0.0f && farHealth > 0.0f );

		// Half of what it takes to crush a fragment, so nothing breaks and the health tells the story
		b3RadialDamageDef damageDef = b3DefaultRadialDamageDef();
		damageDef.position = (b3Pos){ 0.0f, 1.5f, 0.0f };
		damageDef.innerRadius = 0.5f;
		damageDef.outerRadius = 2.5f;
		damageDef.damage = 0.5f * nearHealth;
		damageDef.spreadAlongBonds = pass == 0;
		b3World_ApplyRadialDamage( worldId, &damageDef );
		StepWorld( worldId, 2 );

		float nearAfter = b3Destructible_GetFragmentHealth( destructibleId, nearFragment );
		float farAfter = b3Destructible_GetFragmentHealth( destructibleId, farFragment );

		// The wall that was hit takes damage either way
		ENSURE( nearAfter < nearHealth );

		if ( pass == 0 )
		{
			ENSURE( farAfter == farHealth );
		}
		else
		{
			ENSURE( farAfter < farHealth );
		}

		b3DestroyWorld( worldId );
	}

	b3DestroyStructure( structure );
	return 0;
}

// The application may destroy a fragment shape itself, for example to carve a hole. The fragment
// stays in the structure without its shape and the solver carries on.
static int DestroyFragmentShapeTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 3 );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	int bondCount = b3Destructible_GetBondCount( destructibleId );
	b3BodyId anchorBodyId = b3Destructible_GetFragmentBody( destructibleId, 0 );

	// Find the shape of a fragment in the middle of the wall
	int target = FindFragment( structure, (b3Vec3){ 0.0f, 1.5f, 0.0f } );
	b3ShapeId shapes[1024];
	int shapeCount = b3Body_GetShapes( anchorBodyId, shapes, 1024 );
	b3ShapeId targetShapeId = b3_nullShapeId;
	for ( int i = 0; i < shapeCount; ++i )
	{
		ENSURE( B3_ID_EQUALS( b3Shape_GetDestructible( shapes[i] ), destructibleId ) );
		if ( b3Shape_GetFragmentIndex( shapes[i] ) == target )
		{
			targetShapeId = shapes[i];
		}
	}

	ENSURE( B3_IS_NON_NULL( targetShapeId ) );

	b3DestroyShape( targetShapeId, false );
	ENSURE( b3Shape_IsValid( targetShapeId ) == false );
	ENSURE( b3Body_GetShapeCount( anchorBodyId ) == shapeCount - 1 );

	// Nothing else changed
	StepWorld( worldId, 30 );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	// The wall still breaks like any other, including the fragment that lost its shape
	ApplyBlast( worldId, (b3Pos){ 0.0f, 1.5f, 0.4f }, 1.2f, 2.0e5f, 8000.0f );
	StepWorld( worldId, 120 );
	ENSURE( b3Destructible_IsValid( destructibleId ) );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) < bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) > 1 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// The layer of a bond is the coarsest cluster layer on which its two fragments are in different
// clusters, and the strength of a layer reaches the bonds and the fragments of a destructible.
static int LayerCookTest( void )
{
	DestructionBuilder* builder = CreateDestructionBuilder( 17 );
	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;

	// Two panels side by side, each a cluster
	b3Vec3 halfExtents = { 1.0f, 1.0f, 0.15f };
	b3Transform left = { { -1.0f, 1.0f, 0.0f }, b3Quat_identity };
	b3Transform right = { { 1.0f, 1.0f, 0.0f }, b3Quat_identity };
	fracture.clusters[0] = 0;
	AddFracturedBox( builder, left, halfExtents, &fracture );
	int leftCount = GetFragmentCount( builder );
	fracture.clusters[0] = 1;
	AddFracturedBox( builder, right, halfExtents, &fracture );

	float layerStrength[3] = { 0.5f, 2.0f, 3.0f };
	SetDestructionLayers( builder, 1, layerStrength );
	b3StructureData* structure = BuildStructure( builder );

	// Clustering by a grid of two meter cells gives the same two clusters
	ClusterFragmentsByGrid( builder, 0, 0, (b3Vec3){ 2.0f, 0.0f, 0.0f } );
	b3StructureData* gridStructure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	ENSURE( structure != NULL && gridStructure != NULL );
	ENSURE( gridStructure->bondCount == structure->bondCount );
	for ( int i = 0; i < structure->bondCount; ++i )
	{
		ENSURE( b3GetStructureBonds( gridStructure )[i].layer == b3GetStructureBonds( structure )[i].layer );
	}

	b3DestroyStructure( gridStructure );
	ENSURE( structure->strengthLayerCount == 3 );
	ENSURE( structure->layerStrength[0] == 0.5f && structure->layerStrength[1] == 2.0f && structure->layerStrength[2] == 3.0f );
	ENSURE( b3ValidateStructure( (const uint8_t*)structure, structure->byteCount ) == structure );

	const b3StructureBond* bonds = b3GetStructureBonds( structure );
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	int betweenCount = 0, insideCount = 0;
	for ( int i = 0; i < structure->bondCount; ++i )
	{
		bool isLeftA = bonds[i].fragmentA < leftCount;
		bool isLeftB = bonds[i].fragmentB < leftCount;
		ENSURE( bonds[i].layer == ( isLeftA == isLeftB ? 1 : 0 ) );
		betweenCount += bonds[i].layer == 0;
		insideCount += bonds[i].layer == 1;
	}

	// The panels are four cells high, so about four bonds join them
	ENSURE( betweenCount >= 2 && betweenCount <= 8 );
	ENSURE( insideCount > 4 * betweenCount );

	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	ENSURE( b3Destructible_GetLayerStrength( destructibleId, 0 ) == 0.5f );
	ENSURE( b3Destructible_GetLayerStrength( destructibleId, 1 ) == 2.0f );
	ENSURE( b3Destructible_GetLayerStrength( destructibleId, 2 ) == 3.0f );

	// The structure bonds come first and keep their layer. The anchors to the ground follow and are
	// on no layer.
	int bondCount = b3Destructible_GetBondCount( destructibleId );
	ENSURE( bondCount > structure->bondCount );
	for ( int i = 0; i < bondCount; ++i )
	{
		b3BondState state = b3Destructible_GetBondState( destructibleId, i );
		ENSURE( state.layer == ( i < structure->bondCount ? bonds[i].layer : B3_NULL_INDEX ) );
	}

	// The last layer is the inside of the fragments
	int target = FindFragment( structure, (b3Vec3){ -1.0f, 1.0f, 0.0f } );
	ENSURE( fragments[target].health > 0.0f );
	ENSURE( b3AbsFloat( b3Destructible_GetFragmentHealth( destructibleId, target ) - 3.0f * fragments[target].health ) < 1.0e-3f * fragments[target].health );

	b3Destructible_SetLayerStrength( destructibleId, 2, 6.0f );
	ENSURE( b3Destructible_GetLayerStrength( destructibleId, 2 ) == 6.0f );
	ENSURE( b3AbsFloat( b3Destructible_GetFragmentHealth( destructibleId, target ) - 6.0f * fragments[target].health ) < 1.0e-3f * fragments[target].health );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// A cantilever of eight cells in two clusters. It carries itself until one layer is made weak, and
// then it breaks where that layer is: at the border between the clusters, or at the root inside
// the first cluster, where the bending is the largest.
static int LayerBreakTest( void )
{
	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.strengthVariation = 0.0f;

	enum { cellCount = 8 };
	const float cellLength = 0.375f;

	DestructionBuilder* builder = CreateDestructionBuilder( 1 );
	AddDestructionMaterial( builder, &material );
	for ( int i = 0; i < cellCount; ++i )
	{
		b3Vec3 center = { ( (float)i + 0.5f ) * cellLength, 0.0f, 0.0f };
		b3BoxHull box = b3MakeOffsetBoxHull( 0.5f * cellLength, 0.2f, 0.2f, center );
		int index = AddFragment( builder, &box.base, B3_NULL_INDEX, 0, i == 0 );
		SetFragmentCluster( builder, index, 0, i < cellCount / 2 ? 0 : 1 );
	}

	float layerStrength[3] = { 1.0f, 1.0f, 1.0f };
	SetDestructionLayers( builder, 1, layerStrength );
	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	ENSURE( structure != NULL );

	for ( int pass = 0; pass < 2; ++pass )
	{
		b3WorldId worldId = CreateTestWorld( 1 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		def.position = (b3Pos){ 0.0f, 10.0f, 0.0f };
		b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
		int bondCount = b3Destructible_GetBondCount( destructibleId );
		b3BodyId anchorBodyId = b3Destructible_GetFragmentBody( destructibleId, 0 );

		// It holds as built
		StepWorld( worldId, 60 );
		ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount );

		if ( pass == 0 )
		{
			// Weak between the clusters: the outer cluster drops off in one piece
			b3Destructible_SetLayerStrength( destructibleId, 0, 0.1f );
			StepWorld( worldId, 30 );

			ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount - 1 );
			for ( int i = 0; i < cellCount; ++i )
			{
				bool isHeld = B3_ID_EQUALS( b3Destructible_GetFragmentBody( destructibleId, i ), anchorBodyId );
				ENSURE( isHeld == ( i < cellCount / 2 ) );
			}

			ENSURE( B3_ID_EQUALS( b3Destructible_GetFragmentBody( destructibleId, cellCount / 2 ),
								  b3Destructible_GetFragmentBody( destructibleId, cellCount - 1 ) ) );
		}
		else
		{
			// Weak inside the clusters: the beam snaps at the root and the rest drops off in one
			// piece, the border between the clusters included
			b3Destructible_SetLayerStrength( destructibleId, 1, 0.5f );
			StepWorld( worldId, 30 );

			ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount - 1 );
			ENSURE( B3_ID_EQUALS( b3Destructible_GetFragmentBody( destructibleId, 0 ), anchorBodyId ) );
			for ( int i = 1; i < cellCount; ++i )
			{
				ENSURE( B3_ID_EQUALS( b3Destructible_GetFragmentBody( destructibleId, i ), b3Destructible_GetFragmentBody( destructibleId, 1 ) ) );
			}

			ENSURE( B3_ID_EQUALS( b3Destructible_GetFragmentBody( destructibleId, 1 ), anchorBodyId ) == false );
		}

		b3DestroyWorld( worldId );
	}

	b3DestroyStructure( structure );
	return 0;
}

// Drop a plate of four panels onto a ridge and return the number of support fragments in the
// largest piece that is left, and the number of support fragments that were not crushed.
static int RunPlateDrop( const float* layerStrength, int* survivorCount )
{
	DestructionBuilder* builder = CreateDestructionBuilder( 29 );
	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;

	// Four panels of two by two meters, each a cluster
	b3Vec3 halfExtents = { 1.0f, 0.15f, 1.0f };
	for ( int i = 0; i < 4; ++i )
	{
		b3Transform transform = { { i % 2 == 0 ? -1.0f : 1.0f, 0.0f, i / 2 == 0 ? -1.0f : 1.0f }, b3Quat_identity };
		fracture.clusters[0] = i;
		AddFracturedBox( builder, transform, halfExtents, &fracture );
	}

	SetDestructionLayers( builder, 1, layerStrength );
	b3StructureData* structure = BuildStructure( builder );
	DestroyDestructionBuilder( builder );
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );

	b3WorldId worldId = CreateTestWorld( 1 );
	b3BodyId groundId = CreateGround( worldId );

	// A ridge across the plate, off the border between the panels
	b3BoxHull ridge = b3MakeOffsetBoxHull( 0.2f, 0.5f, 3.0f, (b3Vec3){ 0.5f, 1.5f, 0.0f } );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateHullShape( groundId, &shapeDef, &ridge.base );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	def.position = (b3Pos){ 0.0f, 9.0f, 0.0f };
	def.anchorToStatic = false;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	StepWorld( worldId, 240 );

	b3BodyId bodies[1024];
	b3ShapeId shapes[256];
	int bodyCount = b3Destructible_GetBodies( destructibleId, bodies, 1024 );
	int largest = 0;
	int survivors = 0;
	for ( int i = 0; i < bodyCount; ++i )
	{
		int shapeCount = b3Body_GetShapes( bodies[i], shapes, 256 );
		int size = 0;
		for ( int j = 0; j < shapeCount; ++j )
		{
			int fragmentIndex = b3Shape_GetFragmentIndex( shapes[j] );
			if ( fragmentIndex != B3_NULL_INDEX && fragments[fragmentIndex].parent == B3_NULL_INDEX )
			{
				size += 1;
			}
		}

		largest = b3MaxInt( largest, size );
		survivors += size;
	}

	*survivorCount = survivors;
	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return largest;
}

// The layers decide what a structure breaks into. The same plate falls on the same ridge twice.
// With equal layers it breaks up small and many fragments are crushed. With strong bonds inside the
// panels and tough fragments the panels come through whole.
static int LayerChunkTest( void )
{
	float equal[3] = { 1.0f, 1.0f, 1.0f };
	float strongPanels[3] = { 1.0f, 30.0f, 30.0f };

	int equalSurvivors = 0, panelSurvivors = 0;
	int equalLargest = RunPlateDrop( equal, &equalSurvivors );
	int panelLargest = RunPlateDrop( strongPanels, &panelSurvivors );

	// A panel has sixteen fragments. The border between the panels is no stronger than before, so
	// no piece is larger than a panel.
	ENSURE( panelLargest >= 2 * equalLargest );
	ENSURE( panelLargest <= 16 );
	ENSURE( panelSurvivors >= equalSurvivors );
	return 0;
}

// A recording of a collapse must replay to the same state. The replay world has no destruction solver:
// everything the solver did to the world is in the recording as ordinary operations.
static int CollapseRecordingTest( void )
{
	b3Recording* recording = b3CreateRecording( 0 );
	int bodyCount = 0;
	RunCollapse( 1, true, &bodyCount, recording );
	ENSURE( bodyCount > 4 );
	ENSURE( b3Recording_GetSize( recording ) > 0 );

	bool valid = b3ValidateReplay( b3Recording_GetData( recording ), b3Recording_GetSize( recording ), 1 );
	ENSURE( valid );

	b3DestroyRecording( recording );
	return 0;
}

// b3Shape_SetBody moves a shape without re-creating it and gives the same world as doing it one by one.
static int ShapeSetBodyTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 0.0f, 0.5f, 0.0f };
	b3BodyId bodyA = b3CreateBody( worldId, &bodyDef );
	b3BodyId bodyB = b3CreateBody( worldId, &bodyDef );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BoxHull box1 = b3MakeOffsetBoxHull( 0.5f, 0.5f, 0.5f, (b3Vec3){ 0.0f, 0.0f, 0.0f } );
	b3BoxHull box2 = b3MakeOffsetBoxHull( 0.5f, 0.5f, 0.5f, (b3Vec3){ 1.0f, 0.0f, 0.0f } );
	b3ShapeId shape1 = b3CreateHullShape( bodyA, &shapeDef, &box1.base );
	b3ShapeId shape2 = b3CreateHullShape( bodyA, &shapeDef, &box2.base );
	MAYBE_UNUSED( shape1 );

	// Let the two boxes settle on the ground as one body
	StepWorld( worldId, 30 );
	ENSURE( b3Body_GetContactCapacity( bodyA ) > 0 );
	ENSURE( b3Body_GetShapeCount( bodyA ) == 2 );
	ENSURE( b3Body_GetShapeCount( bodyB ) == 0 );

	// Put body B where body A is, then hand it the second box
	b3Body_SetTransform( bodyB, b3Body_GetPosition( bodyA ), b3Body_GetRotation( bodyA ) );
	b3Shape_SetBody( shape2, bodyB );
	b3Body_ApplyMassFromShapes( bodyA );
	b3Body_ApplyMassFromShapes( bodyB );

	ENSURE( b3Shape_IsValid( shape2 ) );
	ENSURE( B3_ID_EQUALS( b3Shape_GetBody( shape2 ), bodyB ) );
	ENSURE( b3Body_GetShapeCount( bodyA ) == 1 );
	ENSURE( b3Body_GetShapeCount( bodyB ) == 1 );
	ENSURE_SMALL( b3Body_GetMass( bodyA ) - b3Body_GetMass( bodyB ), 1.0e-4f );

	// Both keep resting on the ground: the moved shape found its contacts again
	StepWorld( worldId, 30 );
	ENSURE_SMALL( (float)b3Body_GetPosition( bodyA ).y - 0.5f, 0.02f );
	ENSURE_SMALL( (float)b3Body_GetPosition( bodyB ).y - 0.5f, 0.02f );
	ENSURE( b3Body_GetContactCapacity( bodyB ) > 0 );

	b3DestroyWorld( worldId );
	return 0;
}

// Fragments that break off a static structure start exactly face to face with the neighbors they were
// bonded to, and a blast makes them fast enough for continuous collision. That is the hardest input
// the collision code gets from destruction. So punch a hole in a large wall with a small, violent
// blast: fast debris right next to a structure that is still standing. Leaning cuts and several
// workers. In a debug build this runs under the engine's validation checks.
static int BlastedWallTest( void )
{
	b3WorldId worldId = CreateTestWorld( 2 );
	CreateGround( worldId );

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	fracture.tilt = 0.2f;
	b3StructureData* structure = CreateWallStructure( 10.0f, 5.0f, 0.3f, &fracture, 21 );
	ENSURE( structure != NULL );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );
	b3BodyId anchorBodyId = b3Destructible_GetFragmentBody( destructibleId, 0 );

	StepWorld( worldId, 10 );

	// The blast sits in front of the wall on the +z side, so it must throw debris toward -z
	b3Pos blastCenter = { 0.0f, 2.5f, 0.4f };
	ApplyBlast( worldId, blastCenter, 1.5f, 2.0e5f, 20000.0f );

	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	float maxSpeed = 0.0f;
	float momentumZ = 0.0f;
	int dynamicCount = 0;

	for ( int i = 0; i < 180; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );

		b3BodyId bodies[2048];
		int bodyCount = b3Destructible_GetBodies( destructibleId, bodies, 2048 );
		float stepMomentumZ = 0.0f;
		int stepDynamicCount = 0;
		for ( int j = 0; j < bodyCount; ++j )
		{
			if ( b3Body_GetType( bodies[j] ) != b3_dynamicBody )
			{
				continue;
			}

			b3Vec3 v = b3Body_GetLinearVelocity( bodies[j] );
			ENSURE( b3IsValidVec3( v ) );
			maxSpeed = b3MaxFloat( maxSpeed, b3Length( v ) );
			stepMomentumZ += b3Body_GetMass( bodies[j] ) * v.z;
			stepDynamicCount += 1;
		}

		// Sample the momentum a few steps after the blast, once every deferred body exists
		if ( i == 3 )
		{
			momentumZ = stepMomentumZ;
			dynamicCount = stepDynamicCount;
		}
	}

	// Debris was thrown, fast, and away from the blast
	ENSURE( dynamicCount > 10 );
	ENSURE( maxSpeed > 5.0f );
	ENSURE( momentumZ < 0.0f );

	// Nothing was flung away at an absurd speed, which is what a bad split or a bad impulse looks like
	ENSURE( maxSpeed < 200.0f );

	// The blast makes a hole, it does not bring the wall down: most of the wall is still where it was
	// built. A part of it may have come off its footing and stand there as a free panel, so this
	// looks at where the fragments are and not at which body holds them.
	int standing = 0;
	int anchored = 0;
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		if ( fragments[i].parent != B3_NULL_INDEX )
		{
			continue;
		}

		b3BodyId bodyId = b3Destructible_GetFragmentBody( destructibleId, i );
		if ( B3_IS_NULL( bodyId ) )
		{
			continue;
		}

		// The destructible sits at the origin, so the structure space center is the built position
		b3Pos center = b3TransformWorldPoint( b3Body_GetTransform( bodyId ), fragments[i].center );
		b3Vec3 moved = { (float)center.x - fragments[i].center.x, (float)center.y - fragments[i].center.y,
						 (float)center.z - fragments[i].center.z };
		if ( b3Length( moved ) < 0.25f )
		{
			standing += 1;
		}

		if ( B3_ID_EQUALS( bodyId, anchorBodyId ) )
		{
			anchored += 1;
		}
	}

	ENSURE( standing > structure->supportCount / 2 );
	ENSURE( standing < structure->supportCount );
	ENSURE( anchored > 0 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// A foundation sunk into the ground must still anchor. Its lowest fragment has faces inside the
// static shape instead of resting on it.
static int EmbeddedAnchorTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.strengthVariation = 0.0f;
	const int count = 5;

	// The bottom cube reaches 0.3 m below the ground surface
	b3StructureData* structure = CreateStack( count, -0.3f, false, &material );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

	// Anchors were found, so the whole stack is one static piece
	ENSURE( b3Destructible_GetBondCount( destructibleId ) > count - 1 );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );
	ENSURE( b3Body_GetType( b3Destructible_GetFragmentBody( destructibleId, count - 1 ) ) == b3_staticBody );

	// Only the bottom cube is anchored
	for ( int i = count - 1; i < b3Destructible_GetBondCount( destructibleId ); ++i )
	{
		b3BondState state = b3Destructible_GetBondState( destructibleId, i );
		ENSURE( state.fragmentB == B3_NULL_INDEX );
		ENSURE( state.fragmentA == 0 );
	}

	StepWorld( worldId, 60 );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == b3Destructible_GetBondCount( destructibleId ) );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// A structure with things resting on it must settle into a solved state and stay there. Resting
// contact impulses jitter from step to step. If that jitter restarts the stress solve every step, a
// large structure never converges and never fails under load. So once the loads have settled the
// solver must go quiet, even though the bodies resting on the structure never sleep.
static int RestingLoadTest( void )
{
	b3WorldId worldId = CreateTestWorld( 1 );
	CreateGround( worldId );

	// A slab on four columns
	BuildingDef buildingDef = DefaultBuildingDef();
	buildingDef.floorCount = 1;
	buildingDef.bayCountX = 1;
	buildingDef.bayCountZ = 1;
	buildingDef.enableWalls = false;
	buildingDef.fracture.childSize = 0.0f;
	b3StructureData* structure = CreateBuildingStructure( &buildingDef );

	b3DestructibleDef def = b3DefaultDestructibleDef();
	def.structure = structure;
	b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
	int bondCount = b3Destructible_GetBondCount( destructibleId );

	// Let the self weight solve finish, then note the largest stress
	StepWorld( worldId, 120 );
	ENSURE( b3World_GetDestructionCounters( worldId ).solvedPieceCount == 0 );

	float unloadedStress = 0.0f;
	for ( int i = 0; i < bondCount; ++i )
	{
		unloadedStress = b3MaxFloat( unloadedStress, b3Destructible_GetBondState( destructibleId, i ).fatalRatio );
	}
	ENSURE( unloadedStress > 0.0f );

	// Blocks set down gently on the slab. They may never sleep, so their contacts stay live and jitter.
	// Each weighs a few times more than 5% of the slab fragment under it, enough to matter to the solve,
	// while all four together are a small part of what the slab can carry.
	float slabTop = buildingDef.floorHeight;
	const float blockHalfWidth = 0.25f;
	b3BoxHull blockHull = b3MakeCubeHull( blockHalfWidth );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 2400.0f;
	b3BodyId blocks[4];
	for ( int i = 0; i < 4; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ -1.5f + 1.0f * (float)i, slabTop + blockHalfWidth + 0.05f, 0.3f * (float)( i % 2 ) };
		blocks[i] = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( blocks[i], &shapeDef, &blockHull.base );
	}

	float blockWeight = 10.0f * b3Body_GetMass( blocks[0] );
	ENSURE( blockWeight > 2000.0f );

	// Settle
	StepWorld( worldId, 240 );

	// The blocks are resting on the slab, awake
	for ( int i = 0; i < 4; ++i )
	{
		ENSURE( b3Body_IsAwake( blocks[i] ) );
		ENSURE_SMALL( (float)b3Body_GetPosition( blocks[i] ).y - ( slabTop + blockHalfWidth ), 0.05f );
		ENSURE( b3Length( b3Body_GetLinearVelocity( blocks[i] ) ) < 0.05f );
	}

	// From here on the structure should not even be scheduled for a solve in most steps. Count the
	// scheduling and not the iterations: a small structure that is already solved passes a restart
	// with zero iterations, so iterations alone would not show a solve that restarts every step.
	int busySteps = 0;
	const int watchSteps = 120;
	for ( int i = 0; i < watchSteps; ++i )
	{
		b3World_Step( worldId, s_timeStep, s_subStepCount );
		b3DestructionCounters counters = b3World_GetDestructionCounters( worldId );
		busySteps += counters.solvedPieceCount > 0 ? 1 : 0;
	}

	ENSURE( busySteps <= watchSteps / 10 );

	// The structure feels the blocks: stress went up, and nothing broke
	float loadedStress = 0.0f;
	for ( int i = 0; i < bondCount; ++i )
	{
		loadedStress = b3MaxFloat( loadedStress, b3Destructible_GetBondState( destructibleId, i ).fatalRatio );
	}

	ENSURE( loadedStress > 1.03f * unloadedStress );
	ENSURE( loadedStress < 1.0f );
	ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == bondCount );
	ENSURE( b3Destructible_GetBodyCount( destructibleId ) == 1 );

	b3DestroyWorld( worldId );
	b3DestroyStructure( structure );
	return 0;
}

// Everything the solver allocates must be returned.
static int DestructionLifetimeTest( void )
{
	int64_t baseBytes = b3GetByteCount();

	FractureDef fracture = DefaultFractureDef();
	fracture.cellSize = 0.5f;
	fracture.childSize = 0.25f;
	b3StructureData* structure = CreateWallStructure( 4.0f, 3.0f, 0.3f, &fracture, 11 );

	{
		// Destroy the destructible while it is falling apart
		b3WorldId worldId = CreateTestWorld( 2 );
		CreateGround( worldId );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

		// A second wall from the same structure, standing well clear of the first. Two destructibles
		// must not be created inside each other.
		def.position = (b3Pos){ 0.0f, 0.0f, 10.0f };
		b3DestructibleId secondId = b3CreateDestructible( worldId, &def );
		ENSURE( b3Destructible_GetStructure( destructibleId ) == structure );
		ENSURE( b3Destructible_GetStructure( secondId ) == structure );

		StepWorld( worldId, 10 );
		ApplyBlast( worldId, (b3Pos){ 0.0f, 1.0f, 0.5f }, 2.0f, 1.0e6f, 2000.0f );
		StepWorld( worldId, 40 );
		ENSURE( b3Destructible_GetBodyCount( destructibleId ) > 1 );

		b3BodyId bodies[512];
		int bodyCount = b3Destructible_GetBodies( destructibleId, bodies, 512 );
		b3DestroyDestructible( destructibleId );
		for ( int i = 0; i < bodyCount; ++i )
		{
			ENSURE( b3Body_IsValid( bodies[i] ) == false );
		}

		// The other one is untouched by that and keeps simulating. Break it up as well.
		ENSURE( b3Destructible_IsValid( secondId ) );
		ENSURE( b3Destructible_GetBodyCount( secondId ) == 1 );
		ApplyBlast( worldId, (b3Pos){ 0.0f, 1.0f, 10.5f }, 2.0f, 1.0e6f, 2000.0f );
		StepWorld( worldId, 40 );

		// Destroying a body behind the solver's back is allowed
		bodyCount = b3Destructible_GetBodies( secondId, bodies, 512 );
		ENSURE( bodyCount > 1 );
		b3DestroyBody( bodies[bodyCount - 1] );
		ENSURE( b3Destructible_GetBodyCount( secondId ) == bodyCount - 1 );
		StepWorld( worldId, 10 );

		// Leave the second destructible for the world to clean up
		b3DestroyWorld( worldId );
	}

	{
		// An id must read as invalid once its world is gone, without tripping anything. Use a world
		// that is not in slot zero: a freed slot zero looks like a live world zero to a naive check.
		b3WorldId firstWorldId = CreateTestWorld( 1 );
		b3WorldId secondWorldId = CreateTestWorld( 1 );
		ENSURE( secondWorldId.index1 != firstWorldId.index1 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		def.anchorToStatic = false;
		b3DestructibleId firstId = b3CreateDestructible( firstWorldId, &def );
		b3DestructibleId secondId = b3CreateDestructible( secondWorldId, &def );
		ENSURE( b3Destructible_IsValid( firstId ) );
		ENSURE( b3Destructible_IsValid( secondId ) );
		ENSURE( b3Destructible_IsValid( b3_nullDestructibleId ) == false );

		b3DestroyWorld( secondWorldId );
		ENSURE( b3Destructible_IsValid( secondId ) == false );
		ENSURE( b3Destructible_IsValid( firstId ) );

		b3DestroyWorld( firstWorldId );
		ENSURE( b3Destructible_IsValid( firstId ) == false );
	}

	b3DestroyStructure( structure );
	ENSURE( b3GetByteCount() == baseBytes );
	return 0;
}

// Find a support fragment whose bounding box starts at or above a height
static int FindFragmentAbove( const b3StructureData* structure, float height )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		if ( fragments[i].parent == B3_NULL_INDEX && fragments[i].aabb.lowerBound.y >= height )
		{
			return i;
		}
	}

	return -1;
}

// Crush the lower fragments of the columns of a one bay frame on one side of x = 0
static int CrushFrameColumns( b3DestructibleId destructibleId, const b3StructureData* structure, const BuildingDef* buildingDef,
							  float side )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	int crushed = 0;
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + i;
		bool isColumn = fragment->aabb.upperBound.y < buildingDef->floorHeight - buildingDef->slabThickness + 0.01f;
		if ( isColumn && side * fragment->center.x > 0.0f && fragment->center.y < 1.5f )
		{
			b3Destructible_DamageFragment( destructibleId, i, 10.0f * fragment->health );
			crushed += 1;
		}
	}

	return crushed;
}

// Without stress a structure carries any load. It only breaks where something hits or damages it,
// a piece that loses its last bond to the anchors still falls, and the solver does no work while
// nothing happens. Stress can be turned on and off at any time, per destructible and per world.
static int StressModeTest( void )
{
	b3StructureMaterial material = b3DefaultStructureMaterial();
	material.strengthVariation = 0.0f;

	// A cantilever long enough to snap at its root under its own weight, see CantileverTest. The
	// first pass turns stress off on the destructible, the second on the world.
	int cellCount = 14;
	b3StructureData* beam = CreateBeamStructure( 0.5f * (float)cellCount, 0.3f, 0.3f, cellCount, true, &material );

	for ( int pass = 0; pass < 2; ++pass )
	{
		b3WorldId worldId = CreateTestWorld( 2 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = beam;
		def.position = (b3Pos){ 0.0f, 10.0f, 0.0f };
		def.anchorToStatic = false;
		def.enableBondEvents = true;
		def.enableStress = pass == 1;
		b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
		ENSURE( b3Destructible_IsStressEnabled( destructibleId ) == ( pass == 1 ) );

		b3DestructionSettings settings = b3World_GetDestructionSettings( worldId );
		if ( pass == 1 )
		{
			settings.enableStress = false;
			b3World_SetDestructionSettings( worldId, &settings );
		}

		for ( int i = 0; i < 240; ++i )
		{
			b3World_Step( worldId, s_timeStep, s_subStepCount );

			b3DestructionCounters counters = b3World_GetDestructionCounters( worldId );
			ENSURE( counters.solvedPieceCount == 0 );
			ENSURE( counters.iterationCount == 0 );
			ENSURE( b3World_GetDestructionEvents( worldId ).bondCount == 0 );
		}

		ENSURE( b3Destructible_GetIntactBondCount( destructibleId ) == b3Destructible_GetBondCount( destructibleId ) );
		ENSURE( b3Destructible_GetBondState( destructibleId, 1 ).fatalRatio == 0.0f );
		int64_t bytesWithoutStress = b3World_GetDestructionCounters( worldId ).byteCount;

		// With stress the beam snaps at its root
		if ( pass == 0 )
		{
			b3Destructible_EnableStress( destructibleId, true );
			ENSURE( b3Destructible_IsStressEnabled( destructibleId ) );
		}
		else
		{
			settings.enableStress = true;
			b3World_SetDestructionSettings( worldId, &settings );
		}

		int firstBroken = -1;
		for ( int i = 0; i < 240 && firstBroken == -1; ++i )
		{
			b3World_Step( worldId, s_timeStep, s_subStepCount );

			// The stress system is only built once stress is on
			if ( i == 0 )
			{
				ENSURE( b3World_GetDestructionCounters( worldId ).byteCount > bytesWithoutStress );
			}

			b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
			if ( events.bondCount > 0 )
			{
				firstBroken = events.bondEvents[0].bondIndex;
				ENSURE( events.bondEvents[0].cause == b3_bondBreakOverload || events.bondEvents[0].cause == b3_bondBreakCreep );
			}
		}

		ENSURE( firstBroken == 1 );
		b3DestroyWorld( worldId );
	}

	b3DestroyStructure( beam );

	// A slab on four columns. Taking away the columns on one side leaves it hanging on the other two,
	// which holds without stress. Taking those away as well leaves nothing to hold it, and it falls.
	{
		b3WorldId worldId = CreateTestWorld( 1 );
		CreateGround( worldId );

		BuildingDef buildingDef = DefaultBuildingDef();
		buildingDef.floorCount = 1;
		buildingDef.bayCountX = 1;
		buildingDef.bayCountZ = 1;
		buildingDef.enableWalls = false;
		b3StructureData* structure = CreateBuildingStructure( &buildingDef );

		// The debris of a crushed fragment is too small to simulate, so a crushed column is gone
		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		def.enableStress = false;
		def.enableBondEvents = true;
		def.minFragmentRadius = 100.0f;
		b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );

		float slabBottom = buildingDef.floorHeight - buildingDef.slabThickness;
		int slabFragment = FindFragmentAbove( structure, slabBottom - 0.01f );
		ENSURE( slabFragment >= 0 );

		ENSURE( CrushFrameColumns( destructibleId, structure, &buildingDef, -1.0f ) >= 2 );
		b3World_Step( worldId, s_timeStep, s_subStepCount );
		ENSURE( b3World_GetDestructionEvents( worldId ).bondCount > 0 );

		for ( int i = 0; i < 240; ++i )
		{
			b3World_Step( worldId, s_timeStep, s_subStepCount );
			ENSURE( b3World_GetDestructionEvents( worldId ).bondCount == 0 );
		}

		b3BodyId slabBody = b3Destructible_GetFragmentBody( destructibleId, slabFragment );
		ENSURE( b3Body_GetType( slabBody ) == b3_staticBody );

		ENSURE( CrushFrameColumns( destructibleId, structure, &buildingDef, 1.0f ) >= 2 );
		StepWorld( worldId, 120 );

		slabBody = b3Destructible_GetFragmentBody( destructibleId, slabFragment );
		ENSURE( b3Body_GetType( slabBody ) == b3_dynamicBody );
		// Bodies of a destructible use structure space as their frame
		b3Pos slabPosition = b3Body_GetWorldPoint( slabBody, b3GetStructureFragments( structure )[slabFragment].center );
		ENSURE( slabPosition.y < slabBottom - 0.5f );

		b3DestroyWorld( worldId );
		b3DestroyStructure( structure );
	}

	// A fast ball still breaks a wall without stress, and only the blow breaks bonds
	{
		b3WorldId worldId = CreateTestWorld( 2 );
		CreateGround( worldId );

		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.5f;
		fracture.childSize = 0.0f;
		b3StructureData* structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 11 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = structure;
		def.enableStress = false;
		def.enableBondEvents = true;
		b3CreateDestructible( worldId, &def );

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ 0.0f, 1.5f, 4.0f };
		bodyDef.linearVelocity = (b3Vec3){ 0.0f, 0.0f, -40.0f };
		bodyDef.isBullet = true;
		b3BodyId ballId = b3CreateBody( worldId, &bodyDef );

		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.density = 7800.0f;
		b3Sphere sphere = { b3Vec3_zero, 0.4f };
		b3CreateSphereShape( ballId, &shapeDef, &sphere );

		int impactCount = 0;
		int stressCount = 0;
		for ( int i = 0; i < 120; ++i )
		{
			b3World_Step( worldId, s_timeStep, s_subStepCount );

			b3DestructionEvents events = b3World_GetDestructionEvents( worldId );
			for ( int j = 0; j < events.bondCount; ++j )
			{
				b3BondBreakCause cause = events.bondEvents[j].cause;
				impactCount += cause == b3_bondBreakImpact ? 1 : 0;
				stressCount += cause == b3_bondBreakOverload || cause == b3_bondBreakCreep ? 1 : 0;
			}
		}

		ENSURE( impactCount > 0 );
		ENSURE( stressCount == 0 );

		b3DestroyWorld( worldId );
		b3DestroyStructure( structure );
	}

	return 0;
}

int DestructionTest( void )
{
	RUN_SUBTEST( StructureCookTest );
	RUN_SUBTEST( ShapeSetBodyTest );
	RUN_SUBTEST( ColumnStressTest );
	RUN_SUBTEST( CantileverTest );
	RUN_SUBTEST( SeveredPieceTest );
	RUN_SUBTEST( SplitMomentumTest );
	RUN_SUBTEST( BoulderTest );
	RUN_SUBTEST( FrameCollapseTest );
	RUN_SUBTEST( BuildingDemolitionTest );
	RUN_SUBTEST( ImpactEnergyTest );
	RUN_SUBTEST( DebrisLifetimeTest );
	RUN_SUBTEST( StrengthScaleTest );
	RUN_SUBTEST( HostMotionTest );
	RUN_SUBTEST( MomentumTransferTest );
	RUN_SUBTEST( FreezeDebrisTest );
	RUN_SUBTEST( BodyBudgetTest );
	RUN_SUBTEST( ApplyImpulseTest );
	RUN_SUBTEST( RadialDamageTest );
	RUN_SUBTEST( DestroyFragmentShapeTest );
	RUN_SUBTEST( LayerCookTest );
	RUN_SUBTEST( LayerBreakTest );
	RUN_SUBTEST( LayerChunkTest );
	RUN_SUBTEST( CollapseDeterminismTest );
	RUN_SUBTEST( CollapseRecordingTest );
	RUN_SUBTEST( BlastedWallTest );
	RUN_SUBTEST( EmbeddedAnchorTest );
	RUN_SUBTEST( RestingLoadTest );
	RUN_SUBTEST( DestructionLifetimeTest );
	RUN_SUBTEST( StressModeTest );
	return 0;
}
