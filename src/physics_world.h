// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "arena_allocator.h"
#include "bitset.h"
#include "block_allocator.h"
#include "broad_phase.h"
#include "constraint_graph.h"
#include "id_pool.h"
#include "name_cache.h"

#include "box3d/constants.h"
#include "box3d/types.h"

#define B3_DEBUG_POINT_CAPACITY 64
#define B3_DEBUG_LINE_CAPACITY 64

typedef struct b3Body b3Body;
typedef struct b3Recording b3Recording;
typedef struct b3Contact b3Contact;
typedef struct b3Island b3Island;
typedef struct b3Joint b3Joint;
typedef struct b3Sensor b3Sensor;
typedef struct b3SensorTaskContext b3SensorTaskContext;
typedef struct b3SensorHit b3SensorHit;
typedef struct b3Shape b3Shape;
typedef struct b3SolverSet b3SolverSet;

b3DeclareArray( b3BlockAllocator );
b3DeclareArray( b3Body );
b3DeclareArray( b3SolverSet );
b3DeclareArray( b3Joint );
b3DeclareArray( b3Contact );
b3DeclareArray( b3Island );
b3DeclareArray( b3AABB );
b3DeclareArray( b3Shape );
b3DeclareArray( b3Sensor );
b3DeclareArray( b3SensorTaskContext );
b3DeclareArray( b3SensorHit );
b3DeclareArray( b3BodyMoveEvent );
b3DeclareArray( b3SensorBeginTouchEvent );
b3DeclareArray( b3ContactBeginTouchEvent );
b3DeclareArray( b3SensorEndTouchEvent );
b3DeclareArray( b3ContactEndTouchEvent );
b3DeclareArray( b3ContactHitEvent );
b3DeclareArray( b3JointEvent );

enum b3SetType
{
	b3_staticSet = 0,
	b3_disabledSet = 1,
	b3_awakeSet = 2,
	b3_firstSleepingSet = 3,
};

typedef struct b3DebugPoint
{
	b3Pos p;
	int label;
	float value;
	b3HexColor color;
} b3DebugPoint;

typedef struct b3DebugLine
{
	b3Pos p1, p2;
	int label;
	b3HexColor color;
} b3DebugLine;

// Per thread task storage
typedef struct b3TaskContext
{
	b3Arena arena;

	// Collect per thread sensor continuous hit events.
	b3Array( b3SensorHit ) sensorHits;

	// Broad-phase pairs.
	b3Array( uint64_t ) pairKeys;

	// These bits align with the b3ConstraintGraph::contactBlocks and signal a change in contact status
	b3BitSet contactStateBitSet;

	// These bits align with the joint id capacity and signal a change in contact status
	b3BitSet jointStateBitSet;

	// These bits align with the contact id capacity and signal a hit event.
	b3BitSet hitEventBitSet;

	// Fast-path flag: true when this worker set at least one bit in hitEventBitSet this step.
	bool hasHitEvents;

	// Used to put islands to sleep
	b3BitSet awakeIslandBitSet;

	// Per worker split island candidate
	float splitSleepTime;
	int splitIslandId;

	// Profiling
	int satCallCount;
	int satCacheHitCount;
	int distanceIterations;
	int pushBackIterations;
	int rootIterations;

	// Number of contacts recycled this step (collide pass).
	int recycledContactCount;

	b3DebugPoint points[B3_DEBUG_POINT_CAPACITY];
	int pointCount;

	b3DebugLine lines[B3_DEBUG_LINE_CAPACITY];
	int lineCount;

	int manifoldCounts[B3_CONTACT_MANIFOLD_COUNT_BUCKETS];
	// Prevent false sharing
	char cacheLine[64];
} b3TaskContext;

b3DeclareArray( b3TaskContext );

// Contact skip cache
//
// Most contacts in a pile are not touching, their shapes merely have overlapping bounding boxes. Each
// step the narrow phase still has to load such a contact, both shapes, and both bodies only to discover
// that nothing moved enough to matter (the contact recycling test passes and there is no manifold to
// refresh). The skip cache lets the narrow phase reach the same decision from 16 bytes per contact.
//
// Every awake body has a motion epoch. The epoch advances when the body leaves a small window around the
// pose it had when the epoch started, when one of its bounding boxes is enlarged, when it is fast, or
// has its mass data or shapes modified. So while the epoch holds, the body has barely moved.
//
// When the narrow phase finds a contact that passes the recycling test with enough room to spare for the
// worst motion two epoch windows allow, it stores both body epochs in the skip record. On later steps,
// if both epochs still match, the recycling test is guaranteed to pass again, so the contact is skipped
// without being loaded. This never changes simulation results.
//
// A recycled contact that is touching still needs its manifold separations refreshed from the current
// body transforms. On a step that runs the solver that work is deferred to the contact solver
// preparation, which has to load the contact, manifold, and bodies anyway. The narrow phase only raises
// a flag, both for a contact it skipped and for one it recycled with the regular test.
//
// A contact that has little room left in the recycling test needs a small window, while a large window
// survives more motion. So each body keeps several epochs, one per window size, and a skip record refers
// to the largest window the contact can afford.
//
// Epochs live in slots. Slot i + 1 belongs to the awake body sim with index i and remembers which body
// it tracks. When a different body shows up at that index the slot starts over with new epochs, so the
// cache needs no upkeep as bodies move around the awake set, sleep, or get destroyed. Slot 0 is shared
// by all static bodies. They never move on their own, so one slot that advances on the rare edit of a
// static body is enough. This keeps the cache proportional to the number of awake bodies.

// Number of window sizes per body
#define B3_SKIP_LEVEL_COUNT 3

// Each level has a window this many times smaller than the previous level
#define B3_SKIP_LEVEL_RATIO 8.0f

// Linear size of the largest body motion window. Usually meters.
#define B3_SKIP_LINEAR_WINDOW ( 0.025f * B3_SPECULATIVE_DISTANCE )

// Largest rotation window, as a distance between unit quaternions (about half the angle in radians).
#define B3_SKIP_MAX_QUAT_WINDOW 5.0e-4f

// The epoch slot shared by all static bodies
#define B3_SKIP_STATIC_SLOT 0

// Pose a body had when a motion epoch last advanced
typedef struct b3BodyPoseRef
{
	b3Pos p;
	b3Quat q;
} b3BodyPoseRef;

// Motion tracking for one awake body sim index
typedef struct b3BodyMotionRef
{
	// The body this slot is tracking, to detect a different body moving into this sim index
	int bodyId;

	b3BodyPoseRef levels[B3_SKIP_LEVEL_COUNT];
} b3BodyMotionRef;

// The epoch index of body A in a skip record also carries the manifold count bucket of a touching
// contact in its top bits. This lets the narrow phase keep the manifold statistics without loading the
// contact. The bucket is zero for a contact that is not touching.
#define B3_SKIP_BUCKET_SHIFT 28
#define B3_SKIP_INDEX_MASK ( ( 1 << B3_SKIP_BUCKET_SHIFT ) - 1 )

// Indexed by contact id. Valid while both epochs match the slot epochs. A zero epoch never matches.
typedef struct b3ContactSkip
{
	// Index into the flat array of epochs: B3_SKIP_LEVEL_COUNT * slot + level
	// The index of body A has the manifold bucket in the top bits.
	int epochIndexA;
	int epochIndexB;
	uint32_t epochA;
	uint32_t epochB;
} b3ContactSkip;

// One bit per level, set when a body left that window on the current step
typedef uint8_t b3MotionMask;

// Non-zero when the narrow phase recycled or skipped a touching contact, so its manifolds still need a refresh
typedef uint8_t b3RefreshFlag;

b3DeclareArray( b3BodyMotionRef );
b3DeclareArray( b3ContactSkip );
b3DeclareArrayNative( b3MotionMask );
b3DeclareArrayNative( b3RefreshFlag );

// The world struct manages all physics entities, dynamic simulation,  and asynchronous queries.
// The world also contains efficient memory management facilities.
typedef struct b3World
{
	b3Stack stack;
	b3BroadPhase broadPhase;
	b3ConstraintGraph constraintGraph;

	// Manifold allocators have one allocator for each manifold count.
	b3Array( b3BlockAllocator ) manifoldAllocators;
	b3Mutex* manifoldAllocatorMutex;

	// The body id pool is used to allocate and recycle body ids. Body ids
	// provide a stable identifier for users, but incur caches misses when used
	// to access body data. Aligns with b3Body.
	b3IdPool bodyIdPool;

	// This is a sparse array that maps body ids to the body data
	// stored in solver sets. As sims move within a set or across set.
	// Indices come from id pool.
	b3Array( b3Body ) bodies;

	// Provides free list for solver sets.
	b3IdPool solverSetIdPool;

	// Solvers sets allow body sims to be stored in contiguous arrays. The first holds static bodies. The second set
	// holds disabled bodies. The third holds awake bodies. A growable array holds the sleeping islands.
	// This is split up so that the static set can be accessed without being remapped during a static query while
	// the physics step is concurrently putting an island to sleep (race condition).
	b3SolverSet fixedSets[b3_firstSleepingSet];
	b3Array( b3SolverSet ) sleepingSets;

	// Used to create stable ids for joints
	b3IdPool jointIdPool;

	// This is a sparse array that maps joint ids to the joint data stored in the constraint graph
	// or in the solver sets.
	b3Array( b3Joint ) joints;

	// Used to create stable ids for contacts
	b3IdPool contactIdPool;

	// This is a sparse array that maps contact ids to the contact data stored in the constraint graph
	// or in the solver sets.
	b3Array( b3Contact ) contacts;

	// Used to create stable ids for islands
	b3IdPool islandIdPool;

	// This is a sparse array that maps island ids to the island data stored in the solver sets.
	b3Array( b3Island ) islands;

	b3IdPool shapeIdPool;

	// These are sparse arrays that point into the pools above
	b3Array( b3Shape ) shapes;

	// This follows the shape array and is split out for the hot path in the narrow phase.
	b3Array( b3AABB ) fatAABBs;

	// Contact skip cache, see b3ContactSkip. These are sized at the start of the narrow phase.
	// B3_SKIP_LEVEL_COUNT epochs per slot, kept flat and separate for fast lookup in the narrow phase
	b3Array( uint32_t ) bodyEpochs;

	// One per slot. Used to avoid storing skip records for bodies in motion.
	b3Array( b3MotionMask ) bodyMotionMasks;

	// One per slot. The largest rotation window of the body, zero for the static slot.
	b3Array( float ) bodyQuatWindows;

	// One per awake body sim index. The static slot has none.
	b3Array( b3BodyMotionRef ) bodyMotionRefs;

	// Number of awake body sims at the previous narrow phase
	int skipBodyCount;

	// Skip records indexed by contact id
	b3Array( b3ContactSkip ) contactSkips;

	// Deferred manifold refresh flags indexed by contact id. The narrow phase writes these for every
	// touching contact and the contact solver preparation consumes them.
	b3Array( b3RefreshFlag ) contactRefreshFlags;

	// Number of awake bodies that left every motion window on the current step
	b3AtomicInt movingBodyCount;

	// Looking up skip records is wasted work while most bodies are moving, so it is decided per step
	bool useContactSkips;

	// Compound shapes are static only, so only the broad-phase static pass can meet one. This
	// lets a world without compounds skip the shape type read in the pair batch.
	int compoundShapeCount;

	// Reference counted store of shared hull data keyed by content. Shapes hold a
	// pointer to the hull stored in the db. Type erased to avoid leaking the verstable map
	// type into this header.
	void* hullDatabase;

	// Name cache for shape and body names. This works with recording.
	b3NameCache names;

	// This is a dense array of sensor data.
	b3Array( b3Sensor ) sensors;

	// Per thread storage
	b3Array( b3TaskContext ) taskContexts;
	b3Array( b3SensorTaskContext ) sensorTaskContexts;

	b3Array( b3BodyMoveEvent ) bodyMoveEvents;
	b3Array( b3SensorBeginTouchEvent ) sensorBeginEvents;
	b3Array( b3ContactBeginTouchEvent ) contactBeginEvents;

	// End events are double buffered so that the user doesn't need to flush events
	b3Array( b3SensorEndTouchEvent ) sensorEndEvents[2];
	b3Array( b3ContactEndTouchEvent ) contactEndEvents[2];
	int endEventArrayIndex;

	b3Array( b3ContactHitEvent ) contactHitEvents;
	b3Array( b3JointEvent ) jointEvents;

	// Used to track debug draw
	b3BitSet debugBodySet;
	b3BitSet debugJointSet;
	b3BitSet debugContactSet;
	b3BitSet debugIslandSet;
	b3CreateDebugShapeCallback* createDebugShape;
	b3DestroyDebugShapeCallback* destroyDebugShape;
	void* userDebugShapeContext;

	// Id that is incremented every time step
	uint64_t stepIndex;

	// Identify islands for splitting as follows:
	// - I want to split islands so smaller islands can sleep
	// - when a body comes to rest and its sleep timer trips, I can look at the island and flag it for splitting
	//   if it has removed constraints
	// - islands that have removed constraints must be put split first because I don't want to wake bodies incorrectly
	// - otherwise I can use the awake islands that have bodies wanting to sleep as the splitting candidates
	// - if no bodies want to sleep then there is no reason to perform island splitting
	int splitIslandId;

	b3Vec3 gravity;
	float hitEventThreshold;
	float restitutionThreshold;
	int restitutionIterations;
	bool enableRestitutionPropagation;
	float maxLinearSpeed;
	float isotropicInertiaTolerance;
	float contactSpeed;
	float contactHertz;
	float contactDampingRatio;
	float contactRecycleDistance;

	b3FrictionCallback* frictionCallback;
	b3RestitutionCallback* restitutionCallback;

	uint16_t generation;

	b3Profile profile;
	int satCallCount;
	int satCacheHitCount;
	int manifoldCounts[B3_CONTACT_MANIFOLD_COUNT_BUCKETS];

	b3Capacity maxCapacity;

	b3PreSolveFcn* preSolveFcn;
	void* preSolveContext;

	b3CustomFilterFcn* customFilterFcn;
	void* customFilterContext;

	int workerCount;
	int simdWidth;
	b3EnqueueTaskCallback* enqueueTaskFcn;
	b3FinishTaskCallback* finishTaskFcn;
	void* userTaskContext;
	void* userTreeTask;

	struct b3Scheduler* scheduler;

	void* userData;

	// Non-NULL while a recording session is active. Set by b3World_StartRecording,
	// cleared by b3World_StopRecording. Hooks in mutators check this before writing.
	struct b3Recording* recording;

	// latest inverse sub-step
	float inv_h;

	// latest inverse full-step
	float inv_dt;

	int activeTaskCount;
	int taskCount;

	uint16_t worldId;

	bool enableSleep;

	// This indicates there is a world write operation in progress. This is for debugging and
	// not a real mutex. This should have minimal performance impact.
	bool locked;
	bool enableWarmStarting;
	bool enableContinuous;
	bool enableSpeculative;
	bool inUse;
} b3World;

static inline b3SolverSet* b3GetSolverSet( b3World* world, int setIndex )
{
	if ( setIndex < b3_firstSleepingSet )
	{
		B3_ASSERT( 0 <= setIndex );
		return world->fixedSets + setIndex;
	}

	return b3Array_Get( world->sleepingSets, setIndex - b3_firstSleepingSet );
}

static inline b3SolverSet* b3GetFixedSolverSet( b3World* world, int setIndex )
{
	B3_ASSERT( 0 <= setIndex && setIndex < b3_firstSleepingSet );
	return world->fixedSets + setIndex;
}

static inline int b3GetSolverSetCount( const b3World* world )
{
	return b3_firstSleepingSet + world->sleepingSets.count;
}

// Advance all the motion epochs of a slot. This kills every contact skip record that refers to the slot,
// so those contacts go through the regular narrow phase on the next step.
static inline void b3BumpSkipSlot( b3World* world, int slot )
{
	// Slots beyond the array have no skip records yet
	int base = B3_SKIP_LEVEL_COUNT * slot;
	if ( base + B3_SKIP_LEVEL_COUNT <= world->bodyEpochs.count )
	{
		uint32_t* epochs = world->bodyEpochs.data + base;
		for ( int level = 0; level < B3_SKIP_LEVEL_COUNT; ++level )
		{
			uint32_t epoch = epochs[level] + 1;

			// Zero is reserved for invalid skip records
			epochs[level] = epoch == 0 ? 1 : epoch;
		}
	}
}

// Largest rotation window of a body for the contact skip cache. Sized so the farthest point of the body
// sweeps about the same distance as the linear window. Must only depend on the body extent.
static inline float b3GetSkipQuatWindow( b3Vec3 maxExtent )
{
	float extent = b3Length( maxExtent );
	float window = 0.5f * B3_SKIP_LINEAR_WINDOW / ( extent + B3_SKIP_LINEAR_WINDOW );
	return b3MinFloat( window, B3_SKIP_MAX_QUAT_WINDOW );
}

b3World* b3GetUnlockedWorldFromId( b3WorldId id );
b3World* b3GetWorldFromId( b3WorldId id );

b3World* b3GetUnlockedWorld( int index );
b3World* b3GetWorld( int index );

void b3World_RebuildStaticTree( b3WorldId worldId );

void b3ValidateConnectivity( b3World* world );
void b3ValidateSolverSets( b3World* world );
void b3ValidateContacts( b3World* world );

// Register a hull in the world database, returning the owned shared copy. Identical hulls
// share one copy with a reference count. The input may be freed after this call.
const b3HullData* b3AddHullToDatabase( b3World* world, const b3HullData* src );

// Like b3AddHullToDatabase but takes ownership of a heap hull: inserted directly on a miss,
// freed on a hit. Avoids cloning data the caller already allocated.
const b3HullData* b3AddOwnedHullToDatabase( b3World* world, b3HullData* owned );

// Release a reference to a shared hull. The owned copy is freed when the count reaches zero.
void b3RemoveHullFromDatabase( b3World* world, const b3HullData* data );

static inline b3Manifold* b3AllocateManifolds( b3World* world, int count )
{
	if ( count == 0 )
	{
		return NULL;
	}

	int index = count - 1;

	// Need lock because this is called from the parallel narrow phase
	b3LockMutex( world->manifoldAllocatorMutex );
	int currentCount = world->manifoldAllocators.count;
	for ( int i = currentCount; i < count; ++i )
	{
		// The stride is a multiple of the cache line size so every allocation starts on a cache line. The
		// normal and the first point of a manifold are then always the same two cache lines.
		int elementSize = ( i + 1 ) * (int)sizeof( b3Manifold );
		elementSize = ( ( elementSize - 1 ) | ( B3_CACHE_LINE_SIZE - 1 ) ) + 1;
		b3BlockAllocator allocator = b3CreateBlockAllocator( elementSize, 2 * B3_BLOCK_SIZE );
		b3Array_Push( world->manifoldAllocators, allocator );
	}

	b3BlockAllocator* allocator = b3Array_Get( world->manifoldAllocators, index );
	b3Manifold* manifolds = (b3Manifold*)b3AllocateElement( allocator );
	b3UnlockMutex( world->manifoldAllocatorMutex );
	memset( manifolds, 0, count * sizeof( b3Manifold ) );
	return manifolds;
}

static inline void b3FreeManifolds( b3World* world, b3Manifold* manifolds, int count )
{
	if ( count == 0 )
	{
		return;
	}

	int index = count - 1;
	b3LockMutex( world->manifoldAllocatorMutex );
	b3BlockAllocator* allocator = b3Array_Get( world->manifoldAllocators, index );
	b3FreeElement( allocator, manifolds );
	b3UnlockMutex( world->manifoldAllocatorMutex );
}
