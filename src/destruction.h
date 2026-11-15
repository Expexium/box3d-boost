// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

// Destruction solver
//
// A destructible is an instance of a cooked structure (fragments joined by bonds). The solver keeps
// one rigid body per piece, where a piece is a set of fragments joined by intact bonds. The design
// document calls a piece an island. It is renamed here because a Box3D island is something else:
// the set of awake bodies joined by constraints.
//
// Everything is stored in structure space, the local frame shared by all fragments. Every body of a
// destructible uses that frame as its body frame. So fragment centers, bond centroids, and bond
// normals are constants and a split never has to transform geometry.
//
// The work per world step is split in three:
//
// 1. Pre-step (world unlocked): clear events, make anchored pieces follow a moving host.
//
// 2. Analyze (world locked, after the constraint solver, multithreaded): gather the contact impulses
//    the solver flagged, turn them into loads, solve the stress in each piece, and decide which
//    bonds break and which fragments are crushed. This only reads the world.
//
// 3. Commit (world unlocked, serial): break bonds, split pieces, and change bodies. Every change to
//    the world goes through a recorded operation, so a recording replays without the solver.
//
// The stress solve finds the displacement of every fragment, treating fragments as rigid blocks and
// bonds as stiff springs. This is the rigid body spring model. The bond forces follow from the
// displacements. See destruction_stress.c for the details.

#pragma once

#include "container.h"
#include "id_pool.h"

#include "box3d/destruction.h"
#include "box3d/types.h"

typedef struct b3Body b3Body;
typedef struct b3Shape b3Shape;
typedef struct b3StepContext b3StepContext;
typedef struct b3World b3World;

// The stress solver works in double precision. Bending of a slender member makes the system badly
// conditioned (the condition number grows with the fourth power of the fragment count along the
// member), and single precision stalls far above the tolerance needed to trust the stress.
typedef struct b3DVec3
{
	double x, y, z;
} b3DVec3;

// Linear and angular parts of a fragment displacement or of a load on a fragment.
typedef struct b3DVec6
{
	b3DVec3 lin;
	b3DVec3 ang;
} b3DVec6;

enum b3BondFlags
{
	// The bond no longer carries load.
	b3_bondBroken = 0x01,

	// The bond is rigid and never breaks. Used for fragments the structure fixes to the world.
	b3_bondRigid = 0x02,

	// The bond is already in the break list of its piece for this step.
	b3_bondBreaking = 0x04,
};

// Bond state of a destructible. Geometry is in structure space.
typedef struct b3DestructionBond
{
	// Fragment indices. nodeB is B3_NULL_INDEX for a bond to the world.
	int nodeA;
	int nodeB;

	// Lever arms from the fragment centers to the bond centroid.
	b3Vec3 rA;
	b3Vec3 rB;

	// Unit normal from A to B.
	b3Vec3 normal;

	// Undamaged area and the current effective area. Damage shrinks the effective area.
	float area0;
	float area;

	// Inverse distance between the two fragment centers. Sets the bond stiffness.
	float invLength;

	// Random strength factor of this bond.
	float strength;

	// Stress relative to the limits from the latest evaluation. For queries and debug draw.
	float elasticRatio;
	float fatalRatio;

	// Index in the bond array of the stress system that owns this bond, or B3_NULL_INDEX.
	int solverIndex;

	uint16_t materialIndex;

	// b3BondFlags
	uint8_t flags;

	// Strength layer of the bond, B3_ANCHOR_LAYER for a bond to the world
	uint8_t layer;
} b3DestructionBond;

// Bonds to the world are on no strength layer
#define B3_ANCHOR_LAYER 0xFF

// A bond as the stress solver sees it. Kept small because the solver loops over these.
typedef struct b3SolverBond
{
	// Local node indices. The ground node stands in for the world and for clamped neighbors.
	int a;
	int b;
	b3Vec3 rA;
	b3Vec3 rB;

	// Linear and angular stiffness
	float kl;
	float ka;
} b3SolverBond;

// Inverse of the 6x6 stiffness block of one node, stored in a factored form.
typedef struct b3NodePrecond
{
	// Inverse of the Schur complement of the rotation block, symmetric: xx, xy, xz, yy, yz, zz
	double invS[6];

	// Sum of stiffness times lever arm, couples translation and rotation
	b3DVec3 w;

	// Inverse of the summed linear stiffness
	double invK;
} b3NodePrecond;

// A coarse level of the stress solver. The nodes of the finer level are lumped into clusters that
// move as rigid bodies. A bond between two clusters is still a bond, just with longer lever arms,
// so a level is a smaller bond system of the same form and uses the same solver code.
typedef struct b3StressLevel
{
	// Clusters of this level, without the ground
	int nodeCount;
	int bondCount;

	// The bonds of the finer level that join different clusters
	b3SolverBond* bonds;

	// The bond of the finer level each bond came from
	int* sources;

	b3NodePrecond* precond;

	// Residual, correction, and scratch, with one extra entry for the ground
	b3DVec6* r;
	b3DVec6* z;
	b3DVec6* t;

	// Cluster positions in structure space
	b3Vec3* positions;

	// For each node of the finer level: its cluster and its position relative to that cluster
	int* parents;
	b3Vec3* offsets;

	int byteCount;
} b3StressLevel;

// The most coarse levels of a stress system
#define B3_MAX_STRESS_LEVELS 8

// The stress solver state of one piece. All arrays are indexed by local node and have one extra
// entry at the end for the ground node.
typedef struct b3StressSystem
{
	int nodeCount;
	int bondCount;
	int bondCapacity;

	b3SolverBond* bonds;

	// Maps a solver bond to the bond of the destructible
	int* bondIds;

	// Displacement, residual, preconditioned residual, search direction, scratch, and load
	b3DVec6* x;
	b3DVec6* r;
	b3DVec6* z;
	b3DVec6* p;
	b3DVec6* q;
	b3DVec6* f;

	b3NodePrecond* precond;

	// Coarse levels from fine to coarse. They let the solver move whole regions of the piece at
	// once, which a large piece needs to converge in a few iterations.
	b3StressLevel* levels[B3_MAX_STRESS_LEVELS];
	int levelCount;

	// The clusters of the levels match the current bonds. The stiffness may still be stale.
	bool levelsValid;

	// The load vector was built for the current nodes.
	bool loadsValid;

	// The loads in the solver were solved and checked against the bond limits. Until then the
	// solver keeps working on them, even while the real loads move on.
	bool loadsEvaluated;

	// Steps spent on the loads in the solver
	int loadSteps;

	// Bond visits one iteration costs, for the work budget
	int iterationWork;

	// Conjugate gradient scalars
	double rz;
	double loadSqr;

	// Work of the loads, the reference for the error of the solve. @see b3RunConjugateGradient
	double loadWork;

	// The residual and search direction match the current displacement, load, and bonds.
	bool cgValid;

	// The preconditioner matches the current bonds.
	bool precondValid;

	// The bonds were evaluated against the current displacement.
	bool evaluated;

	int byteCount;
} b3StressSystem;

enum b3PieceFlags
{
	// The piece holds an anchor and lives on the anchor body.
	b3_pieceAnchored = 0x0001,

	// Small piece that uses the debris filter and lifetime.
	b3_pieceDebris = 0x0002,

	// Debris that fell asleep and became static rubble. No longer simulated or solved.
	b3_pieceFrozen = 0x0004,

	// Bonds broke or fragments left, the piece may have come apart.
	b3_pieceTopologyDirty = 0x0008,

	// The stress system must be rebuilt from the fragment list.
	b3_pieceSystemDirty = 0x0010,

	// Some bond is between its elastic and fatal limit and is failing over time.
	b3_pieceCreeping = 0x0020,

	// The body may not sleep because the piece is still failing.
	b3_pieceSleepDisabled = 0x0040,

	// The loads on the piece changed since the latest solve.
	b3_pieceLoadDirty = 0x0080,

	// The piece was created in this step.
	b3_pieceNew = 0x0100,

	// The piece is in the list of pieces that need a commit.
	b3_pieceInCommitList = 0x0200,

	// The piece moved from static to dynamic in this step.
	b3_pieceWasStatic = 0x0400,

	// Some fragment of the piece has a contact load in this step.
	b3_pieceHasLoads = 0x0800,

	// Some fragment of the piece took an impact in this step.
	b3_pieceHasImpact = 0x1000,

	// The piece has no load, so its bonds carry nothing and were reset.
	b3_pieceUnloaded = 0x2000,
};

// A bond the stress solve decided to break.
typedef struct b3BondBreak
{
	int bondIndex;

	// Magnitude of the force the bond carried
	float force;

	// Share of that force above what the bond could carry, in [0, 1)
	float excess;

	// b3BondBreakCause
	int cause;
} b3BondBreak;

b3DeclareArray( b3BondBreak );

// A set of fragments joined by intact bonds. Moves as one rigid body.
typedef struct b3DestructionPiece
{
	// World body index or B3_NULL_INDEX for a free slot. Anchored pieces share the anchor body.
	int bodyId;

	// Fragment indices
	int* nodes;
	int nodeCount;
	int nodeCapacity;

	// b3PieceFlags
	uint32_t flags;

	// Mass properties in structure space, inertia about the center
	float mass;
	b3Vec3 center;
	b3Matrix3 inertia;
	b3Matrix3 invInertia;

	// Intact bonds of this piece, including bonds to the world
	int bondCount;

	// Radius of the bounding sphere about the center
	float radius;

	// Seconds this piece has been debris
	float age;

	// NULL for a piece that needs no solve, such as a single fragment
	b3StressSystem* system;

	// Output of the analysis for this step
	b3Array( b3BondBreak ) breaks;

	// Energy in joules the impacts of this step can spend on breaking the bonds of this piece
	float impactEnergy;

	// Share of the impact impulses on this piece that its broken bonds could not resist
	float impactExcess;

	// Scratch for the above: summed force of the bonds that broke around this piece, and the part
	// of it above what they could carry
	float breakForce;
	float breakExcess;

	// The dynamic piece this piece split from in this step, or B3_NULL_INDEX
	int parentPiece;
} b3DestructionPiece;

b3DeclareArray( b3DestructionPiece );

enum b3NodeFlags
{
	// The fragment is in the loaded node list.
	b3_nodeLoaded = 0x01,

	// The fragment took an impact in this step.
	b3_nodeImpact = 0x02,

	// The fragment ran out of health and will be crushed at the commit.
	b3_nodeCrushed = 0x04,

	// The fragment is in a solver scratch set.
	b3_nodeVisited = 0x08,
};

// Fragment states stored in b3Destructible::fragmentPiece for fragments that are not on a piece
#define B3_FRAGMENT_DORMANT ( -1 )
#define B3_FRAGMENT_REMOVED ( -2 )

// Per fragment load state in structure space. Torques are about the fragment center.
typedef struct b3NodeLoad
{
	// Smoothed contact force and torque
	b3Vec3 force;
	b3Vec3 torque;

	// The smoothed force the stress solve last used. The solve restarts when the force drifts away.
	b3Vec3 solvedForce;

	// Contact impulse and angular impulse gathered in this step. After the load update this holds
	// only the part that counts as an impact.
	b3Vec3 impulse;
	b3Vec3 angularImpulse;

	// Impact energy gathered in this step
	float energy;

	// Impulse and angular impulse the application or an explosion applied in this step. Unlike a
	// contact impulse nothing else delivers this to the body, so a fragment that breaks off the
	// anchor body in this step takes it along as momentum.
	b3Vec3 externalImpulse;
	b3Vec3 externalAngularImpulse;
} b3NodeLoad;

// User requests queued until the next step
typedef enum b3DestructionCommandType
{
	b3_commandBreakBond,
	b3_commandDamageFragment,
	b3_commandDamageBond,
	b3_commandImpulse,
} b3DestructionCommandType;

typedef struct b3DestructionCommand
{
	b3DestructionCommandType type;

	// Bond or fragment index
	int index;

	// Damage in joules for a fragment, or the share of its area a bond loses
	float value;

	// Impulse and angular impulse about the fragment center, structure space
	b3Vec3 impulse;
	b3Vec3 angularImpulse;
} b3DestructionCommand;

b3DeclareArray( b3DestructionCommand );

typedef struct b3Destructible
{
	// Index in b3Destruction::destructibles, B3_NULL_INDEX for a free slot
	int id;
	uint16_t generation;

	const b3StructureData* structure;
	void* userData;

	// Creation options
	b3ShapeDef shapeDef;
	b3Filter debrisFilter;
	float debrisRadius;
	float debrisLifetime;
	float minFragmentRadius;
	float strengthScale;

	// Lowest toughness of the structure materials. Zero means some bonds break without energy.
	float minToughness;

	// Strength of the connections on each layer. The last one in use is the inside of the fragments.
	float layerStrength[B3_MAX_STRENGTH_LAYERS];
	int strengthLayerCount;
	bool freezeSleepingDebris;
	bool enableSleep;
	bool enableBondEvents;
	char name[32];

	// The body that holds every anchored piece. Static, or kinematic when following a host.
	int anchorBodyId;

	// Optional host and the pose of structure space relative to it
	b3BodyId hostBodyId;
	b3Transform hostFrame;

	// Host velocity of the previous step, used to estimate the host acceleration
	b3Vec3 hostLinearVelocity;
	b3Vec3 hostAngularVelocity;

	// Acceleration field of the anchor body in structure space: linear at the origin, angular,
	// and the angular velocity for the centripetal term
	b3Vec3 anchorLinearAcceleration;
	b3Vec3 anchorAngularAcceleration;
	b3Vec3 anchorAngularVelocity;

	int fragmentCount;

	// Piece index of each fragment, or B3_FRAGMENT_DORMANT / B3_FRAGMENT_REMOVED
	int* fragmentPiece;

	// Index of each fragment in the node array of its piece. Also the local node of the solver.
	int* fragmentLocal;

	// Remaining health of each fragment in joules
	float* health;

	// b3NodeFlags
	uint8_t* nodeFlags;

	// World shape index of each structure hull or B3_NULL_INDEX
	int* hullShapes;

	// Structure bonds followed by the anchors created with the destructible
	b3DestructionBond* bonds;
	int bondCount;
	int intactBondCount;

	// Bond indices of each fragment, including anchors
	int* adjacencyStarts;
	int* adjacency;
	int adjacencyCount;

	b3Array( b3DestructionPiece ) pieces;
	b3Array( int ) freePieces;

	// Contact loads per fragment and the fragments that currently have one
	b3NodeLoad* loads;
	b3Array( int ) loadedNodes;

	// Displacement of each fragment from the latest solve. Carries the warm start across splits.
	b3DVec6* displacements;

	// Fragments to crush and pieces that changed, handled at the commit
	b3Array( int ) crushedNodes;
	b3Array( int ) commitPieces;

	b3Array( b3DestructionCommand ) commands;

	// Number of bodies owned, including the anchor body
	int bodyCount;

	// Pieces that could not be split yet because the step ran out of new bodies
	int pendingSplitCount;

	// Debris pieces that are still simulated. They need a look every step for lifetime and freezing.
	int debrisCount;

	// Gravity changed or the host accelerated, so anchored loads must be rebuilt
	bool anchorLoadsDirty;
	b3Vec3 gravity;

	int byteCount;
} b3Destructible;

b3DeclareArray( b3Destructible );

// Maps a world shape to the fragment it belongs to.
typedef struct b3ShapeFragment
{
	int destructibleId;
	int fragmentIndex;

	// The fragment is loose and has nothing to crumble into, so it can never carry a load or be
	// crushed again. Contacts between inert shapes are of no interest to the solver.
	bool isInert;
} b3ShapeFragment;

b3DeclareArray( b3ShapeFragment );

// One side of a solved contact that touches a fragment.
typedef struct b3ContactLoad
{
	int destructibleId;
	int fragmentIndex;

	// The body on the other side of the contact, with its generation to detect a recycled slot
	int otherBodyId;
	uint16_t otherGeneration;

	// Impulse on the fragment over the step, world space
	b3Vec3 impulse;

	// Point of application relative to the center of mass of the fragment's body, world space
	b3Vec3 anchor;

	// World point of application, filled in when the load is handed to its fragment
	b3Pos point;

	// Speed the shapes approached each other before the solve
	float approachSpeed;
} b3ContactLoad;

b3DeclareArray( b3ContactLoad );

// A piece that needs work in the analysis
typedef struct b3PieceTask
{
	int destructibleId;
	int pieceIndex;

	// Solver iterations used, written by the worker
	int iterations;
} b3PieceTask;

b3DeclareArray( b3PieceTask );
b3DeclareArray( b3FragmentEvent );
b3DeclareArray( b3BondEvent );
b3DeclareArray( b3DestructionBodyEvent );

// Destruction state of a world
typedef struct b3Destruction
{
	b3DestructionSettings settings;

	b3IdPool destructibleIdPool;
	b3Array( b3Destructible ) destructibles;

	// Follows the world shape array
	b3Array( b3ShapeFragment ) shapeFragments;

	// Follows the world body array. Destructible index of each body or B3_NULL_INDEX.
	b3Array( int ) bodyOwners;

	b3Array( b3FragmentEvent ) fragmentEvents;
	b3Array( b3BondEvent ) bondEvents;
	b3Array( b3DestructionBodyEvent ) bodyEvents;

	// Contact loads of the current step, two slots per flagged contact
	b3Array( b3ContactLoad ) contactLoads;

	b3Array( b3PieceTask ) tasks;

	// Scratch arrays for splitting a piece
	b3Array( int ) splitLabels;
	b3Array( int ) splitOrder;
	b3Array( int ) splitShapes;

	b3DestructionProfile profile;

	// Counters of the latest step
	int solvedPieceCount;
	int iterationCount;
	int brokenBondCount;
	int newBodyCount;
	int pendingSplitCount;

	// Latest time step
	float timeStep;
	float inverseTimeStep;
	int subStepCount;

	// True while the solver itself destroys bodies and shapes, so the destroy hooks stay quiet.
	bool isMutating;
} b3Destruction;

// Shared helpers, see destruction.c

b3Destructible* b3GetDestructibleFromId( b3World* world, b3DestructibleId id );
b3DestructibleId b3MakeDestructibleId( const b3World* world, const b3Destructible* destructible );

// Allocate a piece. This can move the piece array, so do not hold piece pointers across this call.
int b3AllocPiece( b3Destructible* destructible );
void b3FreePiece( b3Destructible* destructible, int pieceIndex );

// Create or destroy a body owned by a destructible. These go through the public body functions so they
// show up in a recording.
b3BodyId b3CreateDestructionBody( b3World* world, b3Destructible* destructible, b3BodyType type, b3WorldTransform transform,
								  b3Vec3 originVelocity, b3Vec3 angularVelocity );
void b3DestroyDestructionBody( b3World* world, b3Destructible* destructible, int bodyId );

// Create or destroy the shapes of a fragment.
void b3CreateFragmentShapes( b3World* world, b3Destructible* destructible, int fragmentIndex, b3BodyId bodyId, bool isDebris );
void b3DestroyFragmentShapes( b3World* world, b3Destructible* destructible, int fragmentIndex );

void b3PushFragmentEvent( b3World* world, const b3Destructible* destructible, int fragmentIndex, b3FragmentEventType type,
						  int bodyId );

// World hooks, see physics_world.c
void b3DestroyDestruction( b3World* world );
void b3DestructionPreStep( b3World* world, float timeStep );
void b3DestructionAnalyze( b3World* world, b3StepContext* context );
void b3DestructionCommit( b3World* world );
void b3DrawDestruction( b3World* world, b3DebugDraw* draw );
int b3GetDestructionByteCount( b3World* world );

// Called when the application destroys a body or a shape that belongs to a destructible.
void b3DestructionOnBodyDestroyed( b3World* world, b3Body* body );
void b3DestructionOnShapeDestroyed( b3World* world, b3Shape* shape );

// Called by b3World_Explode for every fragment shape in range, including shapes on static bodies. The
// impulse and point are in world space.
void b3DestructionOnExplosion( b3World* world, const b3Shape* shape, b3Vec3 impulse, b3Pos point );

// Stress solver, see destruction_stress.c

typedef struct b3Arena b3Arena;

// Allocate the solver state for a piece with the given number of fragments and bonds.
b3StressSystem* b3CreateStressSystem( int nodeCount, int bondCapacity );
void b3UpdatePieceInert( b3World* world, b3Destructible* destructible, const b3DestructionPiece* piece );

// Strength of a bond relative to its material: the scale of the destructible, the random variation
// of the bond, and the layer the bond is on. This scales the stress limits and the energy to break it.
static inline float b3GetBondStrength( const b3Destructible* destructible, const b3DestructionBond* bond )
{
	float layerStrength = bond->layer < B3_MAX_STRENGTH_LAYERS ? destructible->layerStrength[bond->layer] : 1.0f;
	return destructible->strengthScale * bond->strength * layerStrength;
}

// Strength of the inside of the fragments, which scales the health that keeps them from being crushed
static inline float b3GetCrushStrength( const b3Destructible* destructible )
{
	return destructible->strengthScale * destructible->layerStrength[destructible->strengthLayerCount - 1];
}
void b3DestroyStressSystem( b3StressSystem* system );
int b3GetStressSystemByteCount( const b3StressSystem* system );

// Run the stress solve and the damage evaluation for one piece. Thread-safe across pieces.
// Returns the number of solver iterations used.
int b3SolvePieceStress( b3World* world, b3Destructible* destructible, int pieceIndex, b3Arena arena );

// Mass properties of a set of fragments in structure space.
void b3ComputePieceMass( const b3Destructible* destructible, b3DestructionPiece* piece );

// Stress limits of a bond after the instance scale and the random variation
typedef struct b3BondLimits
{
	float tensileElastic, tensileFatal;
	float compressiveElastic, compressiveFatal;
	float shearElastic, shearFatal;
} b3BondLimits;

b3BondLimits b3GetBondLimits( const b3Destructible* destructible, const b3DestructionBond* bond );

// Linear and angular stiffness of a bond from its effective area
static inline void b3GetBondStiffness( const b3StructureMaterial* material, const b3DestructionBond* bond, float* kl, float* ka )
{
	// A short beam of section A and length L: axial stiffness E A / L and bending stiffness E I / L
	// with I = A^2 / 12 for a square section.
	float base = material->stiffness * bond->area * bond->invLength;
	*kl = base;
	*ka = base * bond->area * ( 1.0f / 12.0f );
}
