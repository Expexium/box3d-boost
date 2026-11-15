// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "arena_allocator.h"
#include "container.h"

#include "box3d/collision.h"
#include "box3d/types.h"

typedef struct b3Shape b3Shape;
typedef struct b3World b3World;

typedef union b3ContactCache
{
	b3SATCache satCache;
	b3SimplexCache simplexCache;
} b3ContactCache;

typedef struct b3TriangleCache
{
	int triangleIndex;
	b3ContactCache cache;
} b3TriangleCache;

b3DeclareArray( b3TriangleCache );

enum b3ContactFlags
{
	// Set when the solid shapes are touching.
	b3_contactTouchingFlag = 0x00000001,

	// Contact has a hit event
	b3_contactHitEventFlag = 0x00000002,

	// This contact wants contact events
	b3_contactEnableContactEvents = 0x00000004,

	// This is contact is between a dynamic and static body
	b3_contactStaticFlag = 0x00000008,

	b3_contactRecycleFlag = 0x00000010,

	// Set when the shapes are touching
	b3_simTouchingFlag = 0x00010000,

	// This contact no longer has overlapping AABBs
	b3_simDisjoint = 0x00020000,

	// This contact started touching
	b3_simStartedTouching = 0x00040000,

	// This contact stopped touching
	b3_simStoppedTouching = 0x00080000,

	// This contact has a hit event
	b3_simEnableHitEvent = 0x00100000,

	// This contact wants pre-solve events
	b3_simEnablePreSolveEvents = 0x00200000,

	// This is a mesh contact
	b3_simMeshContact = 0x00400000,

	// Relative transform is cached for contact recycling
	b3_relativeTransformValid = 0x00800000,

	// Enable speculative contact points
	b3_enableSpeculativePoints = 0x01000000,
};

// A contact edge is used to connect bodies and contacts together
// in a contact graph where each body is a node and each contact
// is an edge. A contact edge belongs to a doubly linked list
// maintained in each attached body. Each contact has two contact
// edges, one for each attached body.
typedef struct b3ContactEdge
{
	int bodyId;
	int prevKey;
	int nextKey;
} b3ContactEdge;

typedef struct b3MeshContact
{
	b3Array( b3TriangleCache ) triangleCache;
	b3AABB queryBounds;
} b3MeshContact;

typedef struct b3ConvexContact
{
	b3ContactCache cache;
} b3ConvexContact;

// Represents the persistent interaction between two shapes
// The fields are grouped by who reads them. The narrow phase and the contact solver preparation only
// touch the first two cache lines of a contact. Contacts are stored in a cache aligned array.
typedef struct b3Contact
{
	// First cache line

	_Alignas( B3_CACHE_LINE_SIZE ) int shapeIdA;
	int shapeIdB;

	// Encoded body sim indices to avoid a cache miss accessing the body in the narrow phase.
	int encodedBodySimA;
	int encodedBodySimB;

	// b3ContactFlags
	uint32_t flags;

	int manifoldCount;

	// todo embed single manifold with ability to allocate overfloat, use union.
	b3Manifold* manifolds;

	// Cache for contact recycling.
	b3Quat cachedRotationA;
	b3Quat cachedRotationB;

	// Second cache line

	b3Transform cachedRelativePose;

	// Mixed friction and restitution
	float friction;
	float restitution;
	float rollingResistance;
	b3Vec3 tangentVelocity;

	// Back index into b3World::contacts
	int contactId;

	// index into the constraint graph color array
	// B3_NULL_INDEX for non-touching or sleeping contacts
	// B3_NULL_INDEX when slot is free
	int colorIndex;

	// contact index within set or graph color
	// B3_NULL_INDEX when slot is free
	int localIndex;

	// The rest is only needed when a manifold is computed or the contact changes state

	// Usage determined by b3_simMeshContact in simFlags
	union
	{
		b3ConvexContact convexContact;
		b3MeshContact meshContact;
	};

	int childIndex;

	// index of simulation set stored in b3World
	// B3_NULL_INDEX when slot is free
	int setIndex;

	// A contact only belongs to an island if touching, otherwise B3_NULL_INDEX.
	int islandId;

	// Index into the island's contacts array for O(1) swap-removal.
	// B3_NULL_INDEX when not in an island.
	int islandIndex;

	// This is monotonically advanced when a contact is allocated in this slot
	// Used to check for invalid b3ContactId
	uint32_t generation;

	b3ContactEdge edges[2];
} b3Contact;

typedef struct b3ContactSpec
{
	int contactId;

	// Start of the global manifold constraint array
	int manifoldStart;
	uint16_t manifoldCount;
} b3ContactSpec;

b3DeclareArray( b3ContactSpec );

void b3InitializeContactRegisters( void );

void b3CreateContact( b3World* world, b3Shape* shapeA, b3Shape* shapeB, int childIndex );
void b3DestroyContact( b3World* world, b3Contact* contact, bool wakeBodies );

// The most lanes any wide narrow phase function uses
#define B3_COLLIDE_MAX_LANES 8

// Input of the wide contact recycling test, one contact per lane. The number of lanes is the SIMD width.
// Every lane must be filled, repeat a contact to pad a partial group.
typedef struct b3RecycleLanes
{
	const b3Contact* contacts[B3_COLLIDE_MAX_LANES];
	const struct b3BodySim* simsA[B3_COLLIDE_MAX_LANES];
	const struct b3BodySim* simsB[B3_COLLIDE_MAX_LANES];

	// The recycling tolerance, which depends on the contact touching
	_Alignas( 32 ) float tolerances[B3_COLLIDE_MAX_LANES];

	// The larger extent of the two bodies, where a static body has no extent
	float extentX[B3_COLLIDE_MAX_LANES];
	float extentY[B3_COLLIDE_MAX_LANES];
	float extentZ[B3_COLLIDE_MAX_LANES];
} b3RecycleLanes;

// Output of the wide contact recycling test
typedef struct b3RecycleResults
{
	// Pose of body B relative to body A, the same value b3InvMulWorldTransforms gives
	_Alignas( 32 ) float px[B3_COLLIDE_MAX_LANES];
	float py[B3_COLLIDE_MAX_LANES];
	float pz[B3_COLLIDE_MAX_LANES];
	float qx[B3_COLLIDE_MAX_LANES];
	float qy[B3_COLLIDE_MAX_LANES];
	float qz[B3_COLLIDE_MAX_LANES];
	float qs[B3_COLLIDE_MAX_LANES];

	// Intermediate values of the test, only meaningful for a contact that can be recycled
	float slack[B3_COLLIDE_MAX_LANES];
	float arcSquared[B3_COLLIDE_MAX_LANES];
	float angularDistance[B3_COLLIDE_MAX_LANES];
} b3RecycleResults;

// Input of the wide capsule separation test, one pair of capsule shapes per lane. Every lane must be filled.
typedef struct b3CapsuleLanes
{
	const b3Shape* shapesA[B3_COLLIDE_MAX_LANES];
	const b3Shape* shapesB[B3_COLLIDE_MAX_LANES];

	// Pose of body B relative to body A
	_Alignas( 32 ) float px[B3_COLLIDE_MAX_LANES];
	float py[B3_COLLIDE_MAX_LANES];
	float pz[B3_COLLIDE_MAX_LANES];
	float qx[B3_COLLIDE_MAX_LANES];
	float qy[B3_COLLIDE_MAX_LANES];
	float qz[B3_COLLIDE_MAX_LANES];
	float qs[B3_COLLIDE_MAX_LANES];
} b3CapsuleLanes;

// See collide_wide.inl. These return a bit per lane.
int b3TestContactRecyclingW4( const b3RecycleLanes* lanes, b3RecycleResults* results );
int b3TestCapsuleSeparationW4( const b3CapsuleLanes* lanes );
#if defined( B3_SIMD_HAS_WIDTH_8 )
int b3TestContactRecyclingW8( const b3RecycleLanes* lanes, b3RecycleResults* results );
int b3TestCapsuleSeparationW8( const b3CapsuleLanes* lanes );
#endif

// Refresh the manifold separations of a recycled contact from the current body transforms. On a step
// that runs the solver, the narrow phase only sets the refresh flag of the contact and the contact
// solver preparation calls this, because it loads the manifolds anyway. Otherwise the narrow phase
// calls this itself. See b3ContactSkip.
void b3RefreshRecycledContact( b3World* world, b3Contact* contact );

// The body sims of a contact, whatever solver set the bodies are in
void b3GetContactBodySims( b3World* world, const b3Contact* contact, const struct b3BodySim** bodySimA,
						   const struct b3BodySim** bodySimB );

// The local centers are passed by pointer because they are only read when the contact is touching. They
// are not in the part of the body sim the narrow phase has already loaded.
bool b3UpdateContact( b3World* world, int workerIndex, b3Contact* contact, b3Shape* shapeA, const b3Vec3* localCenterA,
					  b3WorldTransform xfA, b3Shape* shapeB, const b3Vec3* localCenterB, b3WorldTransform xfB, bool isFast,
					  b3Arena arena );

bool b3ComputeMeshManifolds( b3World* world, int workerIndex, b3Contact* contact, const b3Shape* shapeA,
							 const uint16_t* materialMap, b3WorldTransform xfA, const b3Shape* shapeB, b3WorldTransform xfB,
							 bool isFast, b3Arena arena );
