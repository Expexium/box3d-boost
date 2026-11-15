// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base.h"
#include "id.h"
#include "math_functions.h"
#include "types.h"

#include <stdbool.h>
#include <stdint.h>

/**
 * @defgroup destruction Destruction
 * @brief Stress driven destruction of pre-fractured structures.
 *
 * A structure is a set of convex fragments joined by bonds. Each bond is a patch of shared surface
 * with an area, a centroid, and a normal. A structure is cooked once into a b3StructureData and can
 * then be instanced many times in a world as a destructible.
 *
 * A destructible keeps one rigid body per piece. A piece is a group of fragments that are still
 * joined by intact bonds. Pieces that hold an anchor live on a static body (or on a kinematic body
 * that follows a host). All other pieces are dynamic bodies.
 *
 * Every world step the destruction solver:
 * - gathers contact impulses, gravity, spin, and user impulses as loads on the fragments
 * - solves the force and moment every bond must carry to keep its piece rigid
 * - converts those into tensile, compressive, and shear stress and compares with material limits
 * - damages or breaks overloaded bonds and crushes fragments that absorbed too much energy
 * - splits pieces that are no longer connected into new bodies
 *
 * All fragments of a destructible are expressed in one local frame called structure space. Every
 * body of the destructible uses that frame as its body frame, so a fragment world transform is
 * simply the transform of the body it is currently attached to.
 *
 * Lengths are in meters, mass in kilograms, stress in pascals, and energy in joules.
 * @{
 */

/// 64-bit structure version. Useful for validating serialized data.
#define B3_STRUCTURE_VERSION 0x5D3A91C27E64B80Full

/// Material used by the destruction solver. Strengths are stresses in pascals (N/m^2). A bond takes
/// no damage below its elastic limit, loses strength between the elastic and the fatal limit, and
/// breaks at once at the fatal limit.
/// @see b3DefaultStructureMaterial
typedef struct b3StructureMaterial
{
	/// Mass density used when a fragment has no explicit mass, usually in kg/m^3.
	float density;

	/// Tensile stress where damage starts.
	float tensileElasticLimit;

	/// Tensile stress where a bond breaks at once.
	float tensileFatalLimit;

	/// Compressive stress where damage starts.
	float compressiveElasticLimit;

	/// Compressive stress where a bond breaks at once.
	float compressiveFatalLimit;

	/// Shear stress where damage starts.
	float shearElasticLimit;

	/// Shear stress where a bond breaks at once.
	float shearFatalLimit;

	/// Relative elastic stiffness, non-dimensional. Stiffer bonds attract more of the load when a
	/// structure has several load paths. Only the ratio between materials matters.
	float stiffness;

	/// Fracture energy per unit of new surface, usually in J/m^2. This sets how much impact energy
	/// or direct damage a fragment absorbs before it is crushed, and how much a bond absorbs.
	float toughness;

	/// Time in seconds an overloaded bond takes to fail when it is held between its elastic and
	/// fatal limit. Smaller is more brittle. This makes sustained damage frame rate independent.
	float creepTime;

	/// Random spread of bond strength in the range [0, 1). A value of 0.2 varies strength by +-20%
	/// so identical instances fail differently.
	float strengthVariation;

	/// User material identifier, passed back with events. Not used internally.
	uint64_t userMaterialId;
} b3StructureMaterial;

/// Use this to initialize your structure material. The default resembles plain concrete.
B3_API b3StructureMaterial b3DefaultStructureMaterial( void );

/// A fragment is one pre-fractured piece of geometry. Fragments form a hierarchy. Fragments without
/// a parent are support fragments. They carry bonds and take part in the stress solve. The children
/// of a fragment are dormant debris that is released when the parent is crushed.
/// @see b3DefaultFragmentDef
typedef struct b3FragmentDef
{
	/// The convex hulls of this fragment. The hulls are cloned. A fragment needs at least one hull.
	const b3HullData* const* hulls;

	/// The number of hulls.
	int hullCount;

	/// Transform from hull space into structure space.
	b3Transform transform;

	/// Index of the parent fragment or B3_NULL_INDEX for a support fragment.
	int parentIndex;

	/// Index into b3StructureDef::materials.
	int materialIndex;

	/// Optional mass properties in structure space. If NULL they are computed from the hulls and
	/// the material density. Supply these when the hulls overestimate a concave fragment.
	const b3MassData* massData;

	/// Energy in joules this fragment absorbs before it is crushed. If zero this is computed from
	/// the material toughness and the surface that crushing would create.
	float health;

	/// Fix this fragment to the world. Only applies to support fragments.
	bool isAnchored;

	/// Use this to store application specific fragment data, for example a render bone index.
	uint64_t userData;
} b3FragmentDef;

/// Use this to initialize your fragment definition
B3_API b3FragmentDef b3DefaultFragmentDef( void );

/// An explicit bond between two support fragments, or between a fragment and the world.
typedef struct b3BondDef
{
	/// Index of the first fragment.
	int fragmentA;

	/// Index of the second fragment, or B3_NULL_INDEX to anchor fragment A to the world.
	int fragmentB;

	/// Centroid of the bonded surface in structure space.
	b3Vec3 centroid;

	/// Unit normal of the bonded surface in structure space, pointing from fragment A to fragment B.
	b3Vec3 normal;

	/// Area of the bonded surface, usually in m^2.
	float area;
} b3BondDef;

/// Used to cook a structure. Must be initialized using b3DefaultStructureDef.
typedef struct b3StructureDef
{
	/// The fragments.
	const b3FragmentDef* fragments;

	/// The number of fragments.
	int fragmentCount;

	/// Optional explicit bonds. A generated bond is skipped when an explicit bond joins the same pair.
	const b3BondDef* bonds;

	/// The number of explicit bonds.
	int bondCount;

	/// The materials referenced by the fragments. If NULL every fragment uses the default material.
	const b3StructureMaterial* materials;

	/// The number of materials.
	int materialCount;

	/// Generate bonds wherever two support fragments share a surface.
	bool generateBonds;

	/// The largest gap between two surfaces that can still be bonded, usually in meters.
	float bondGap;

	/// Shared surfaces smaller than this are not bonded, usually in m^2.
	float minBondArea;

	/// Used internally to detect a valid definition. DO NOT SET.
	int internalValue;
} b3StructureDef;

/// Use this to initialize your structure definition
B3_API b3StructureDef b3DefaultStructureDef( void );

/// Fragment flags
typedef enum b3FragmentFlags
{
	/// This fragment is fixed to the world by the structure definition.
	b3_fragmentAnchored = 0x00000001,
} b3FragmentFlags;

/// A cooked fragment. Geometry is in structure space.
typedef struct b3StructureFragment
{
	/// Center of mass.
	b3Vec3 center;

	/// Mass, usually in kilograms.
	float mass;

	/// Rotational inertia about the center of mass.
	b3Matrix3 inertia;

	/// Bounding box of the hulls.
	b3AABB aabb;

	/// Total hull volume, usually in m^3.
	float volume;

	/// Energy absorbed before the fragment is crushed, usually in joules.
	float health;

	/// Radius of the bounding sphere about the center of mass.
	float radius;

	/// Parent fragment or B3_NULL_INDEX for a support fragment.
	int32_t parent;

	/// First entry in the child index array.
	int32_t firstChild;

	/// The number of children.
	int32_t childCount;

	/// First entry in the hull array.
	int32_t firstHull;

	/// The number of hulls.
	int32_t hullCount;

	/// First entry in the adjacency array, a list of bond indices.
	int32_t firstBond;

	/// The number of bonds on this fragment.
	int32_t bondCount;

	/// Material index.
	int32_t materialIndex;

	/// b3FragmentFlags
	uint32_t flags;

	/// Explicit padding for determinism.
	uint32_t padding;

	/// Application specific fragment data.
	uint64_t userData;
} b3StructureFragment;

/// Bond flags
typedef enum b3StructureBondFlags
{
	/// This bond never breaks. Used for fragments the structure definition fixes to the world.
	b3_structureBondRigid = 0x00000001,
} b3StructureBondFlags;

/// A cooked bond. Geometry is in structure space.
typedef struct b3StructureBond
{
	/// First fragment.
	int32_t fragmentA;

	/// Second fragment or B3_NULL_INDEX for a bond to the world.
	int32_t fragmentB;

	/// Centroid of the bonded surface.
	b3Vec3 centroid;

	/// Unit normal pointing from fragment A to fragment B.
	b3Vec3 normal;

	/// Area of the bonded surface.
	float area;

	/// b3StructureBondFlags
	uint32_t flags;
} b3StructureBond;

/// A cooked structure. Created with b3CreateStructure.
/// @note This data structure has data hanging off the end and cannot be directly copied. It holds no
/// pointers, so the full byteCount may be written to a file and loaded later.
typedef struct b3StructureData
{
	/// Version must be first and match B3_STRUCTURE_VERSION
	uint64_t version;

	/// Hash of this structure (this field is zero when the hash is computed).
	uint64_t hash;

	/// Bounding box in structure space.
	b3AABB aabb;

	/// Total mass of the support fragments.
	float mass;

	/// The total number of bytes for this structure.
	int32_t byteCount;

	/// The number of fragments on all levels.
	int32_t fragmentCount;

	/// The number of support fragments.
	int32_t supportCount;

	/// Offset of the fragment array in bytes from the struct address.
	int32_t fragmentOffset;

	/// The number of bonds.
	int32_t bondCount;

	/// Offset of the bond array.
	int32_t bondOffset;

	/// The number of adjacency entries.
	int32_t adjacencyCount;

	/// Offset of the adjacency array. Each fragment owns a range of bond indices.
	int32_t adjacencyOffset;

	/// The number of child index entries.
	int32_t childCount;

	/// Offset of the child index array. Each fragment owns a range of child fragment indices.
	int32_t childOffset;

	/// The number of hulls.
	int32_t hullCount;

	/// Offset of the hull offset array. Each entry is the offset of a hull from the struct address.
	int32_t hullOffset;

	/// The number of materials.
	int32_t materialCount;

	/// Offset of the material array.
	int32_t materialOffset;

	/// Any padding must be explicit.
} b3StructureData;

/// Cook a structure. This computes mass properties, the fragment hierarchy, and the bond graph.
/// This does not access any world and may be called from any thread.
/// @return the cooked structure or NULL if the definition is invalid.
B3_API b3StructureData* b3CreateStructure( const b3StructureDef* def );

/// Destroy a structure.
/// @warning every destructible created from this structure must be destroyed first
B3_API void b3DestroyStructure( b3StructureData* structure );

/// Validate structure bytes, for example after loading them from a file.
/// @return the structure or NULL if the bytes are not a compatible structure.
B3_API const b3StructureData* b3ValidateStructure( const uint8_t* bytes, int byteCount );

/// Get read only fragments.
B3_INLINE const b3StructureFragment* b3GetStructureFragments( const b3StructureData* structure )
{
	return (const b3StructureFragment*)( (intptr_t)structure + structure->fragmentOffset );
}

/// Get read only bonds.
B3_INLINE const b3StructureBond* b3GetStructureBonds( const b3StructureData* structure )
{
	return (const b3StructureBond*)( (intptr_t)structure + structure->bondOffset );
}

/// Get the adjacency array. This holds the bond indices of each fragment.
/// @see b3StructureFragment::firstBond
B3_INLINE const int32_t* b3GetStructureAdjacency( const b3StructureData* structure )
{
	return (const int32_t*)( (intptr_t)structure + structure->adjacencyOffset );
}

/// Get the child index array. This holds the child fragment indices of each fragment.
/// @see b3StructureFragment::firstChild
B3_INLINE const int32_t* b3GetStructureChildren( const b3StructureData* structure )
{
	return (const int32_t*)( (intptr_t)structure + structure->childOffset );
}

/// Get read only materials.
B3_INLINE const b3StructureMaterial* b3GetStructureMaterials( const b3StructureData* structure )
{
	return (const b3StructureMaterial*)( (intptr_t)structure + structure->materialOffset );
}

/// Get a hull in structure space.
B3_INLINE const b3HullData* b3GetStructureHull( const b3StructureData* structure, int hullIndex )
{
	const int32_t* offsets = (const int32_t*)( (intptr_t)structure + structure->hullOffset );
	return (const b3HullData*)( (intptr_t)structure + offsets[hullIndex] );
}

/// Used to create a destructible. Must be initialized using b3DefaultDestructibleDef.
typedef struct b3DestructibleDef
{
	/// The cooked structure.
	/// @warning this must remain valid for the lifetime of the destructible
	const b3StructureData* structure;

	/// The world position of the structure space origin.
	b3Pos position;

	/// The world rotation of structure space.
	b3Quat rotation;

	/// Optional host for the anchors, for example a kinematic vehicle. Anchored pieces follow the
	/// host and feel its acceleration. If null the anchored pieces are static.
	b3BodyId hostBodyId;

	/// Template for the fragment shapes. The filter, the material, and the event flags are used. The
	/// density and the mass update option are ignored because fragments carry their own mass.
	b3ShapeDef shapeDef;

	/// Create anchors where support fragments touch static shapes when the destructible is created.
	bool anchorToStatic;

	/// Distance within which a fragment surface counts as touching a static shape, usually in meters.
	float anchorTolerance;

	/// Scales every material strength of this instance.
	float strengthScale;

	/// Seed for the random bond strength variation.
	uint32_t seed;

	/// Dynamic pieces with a bounding radius below this are debris. Debris uses the debris filter.
	float debrisRadius;

	/// Collision filter for debris. Use it to stop debris from colliding with other debris.
	b3Filter debrisFilter;

	/// Seconds a debris body lives. Zero keeps debris forever.
	float debrisLifetime;

	/// Released fragments with a bounding radius below this are never simulated. They are reported
	/// with a b3_fragmentRemoved event so the application can spawn a visual effect instead.
	float minFragmentRadius;

	/// Turn debris into static rubble when it falls asleep. Static rubble never wakes.
	bool freezeSleepingDebris;

	/// Can the dynamic pieces fall asleep?
	bool enableSleep;

	/// Report a b3BondEvent for every bond that breaks.
	bool enableBondEvents;

	/// Optional name for debugging. Applied to the bodies.
	const char* name;

	/// Use this to store application specific data. This is also the user data of every body.
	void* userData;

	/// Used internally to detect a valid definition. DO NOT SET.
	int internalValue;
} b3DestructibleDef;

/// Use this to initialize your destructible definition
B3_API b3DestructibleDef b3DefaultDestructibleDef( void );

/// World wide tuning of the destruction solver.
/// @see b3World_SetDestructionSettings
typedef struct b3DestructionSettings
{
	/// Turn the stress solve on or off. When off only direct damage and b3Destructible_BreakBond
	/// break bonds.
	bool enableStress;

	/// The most solver iterations a piece gets per world step.
	int maxIterations;

	/// Solver work a piece may spend per world step, in bond visits. A piece always gets a few
	/// iterations. A large piece that needs more than its budget keeps solving the same loads over
	/// the following steps and reacts a few steps late, instead of stalling the frame. Raise this
	/// to make large structures react sooner, lower it to save time.
	int workBudget;

	/// Relative error where the solver stops. The error is measured in energy: the work the
	/// unbalanced forces would still do, against the work of the loads.
	float solverTolerance;

	/// Relative error a piece must reach before sustained loads may wear its bonds down. Bonds
	/// that are far over their limit break before that, bonds near their limit wait for it, so a
	/// half converged solution does not break bonds that would have held.
	float damageTolerance;

	/// Time constant in seconds used to smooth contact forces before they load a structure.
	float loadSmoothingTime;

	/// A contact impulse counts as an impact when it could change the velocity of the fragment it
	/// hits by more than this, usually in meters per second.
	float impactVelocityThreshold;

	/// Duration of an impact in seconds. The impact force is the impulse divided by this. Zero
	/// uses the time step.
	float impactDuration;

	/// An impact stresses fragments within this distance of the hit, usually in meters. The
	/// structure beyond that is treated as immovable for the duration of the impact.
	float impactRadius;

	/// Fraction of an impact energy that damages the fragment it hits.
	float impactDamageScale;

	/// Scales the impact energy that can break bonds near the hit. The stress of the impact picks
	/// which bonds fail, this energy decides how many: a bond costs its area times the material
	/// toughness. A crack along one bond takes far less energy than crushing a fragment to rubble,
	/// which is what the toughness is tuned for, so this is above one. Larger values make falling
	/// pieces break up more when they land.
	float impactFractureScale;

	/// Scales the momentum returned to a body whose impact broke the bonds that resisted it.
	/// One conserves momentum. Zero turns this off.
	float momentumTransferScale;

	/// The most bodies the solver may create per world step. Splits beyond this are finished in
	/// the following steps.
	int maxNewBodiesPerStep;
} b3DestructionSettings;

/// Use this to initialize your destruction settings
B3_API b3DestructionSettings b3DefaultDestructionSettings( void );

/// Used to apply damage in a sphere.
/// @see b3World_ApplyRadialDamage
typedef struct b3RadialDamageDef
{
	/// Mask bits to filter shapes
	uint64_t maskBits;

	/// The center of the damage in world space
	b3Pos position;

	/// Fragments within this radius take the full damage
	float innerRadius;

	/// Damage falls to zero at this radius
	float outerRadius;

	/// Damage at the center, usually in joules. Compare with b3StructureFragment::health.
	float damage;

	/// Measure the falloff along bonds from the fragments inside the inner radius instead of
	/// through open space. Use this so that damage does not jump across gaps.
	bool spreadAlongBonds;
} b3RadialDamageDef;

/// Use this to initialize your radial damage definition
B3_API b3RadialDamageDef b3DefaultRadialDamageDef( void );

/// The kinds of fragment events
typedef enum b3FragmentEventType
{
	/// The fragment moved to another body. This is the event to follow for rendering.
	b3_fragmentMoved,

	/// A dormant child fragment was released and now has a body.
	b3_fragmentReleased,

	/// The fragment is no longer simulated. It was crushed into its children, was too small to
	/// simulate, or its debris lifetime ended.
	b3_fragmentRemoved,
} b3FragmentEventType;

/// A fragment changed body or stopped being simulated.
typedef struct b3FragmentEvent
{
	/// The destructible
	b3DestructibleId destructibleId;

	/// The fragment index in the structure
	int fragmentIndex;

	/// The event type
	b3FragmentEventType type;

	/// The body the fragment is now attached to. For b3_fragmentRemoved this is the body the
	/// fragment was last attached to, which may have been destroyed.
	b3BodyId bodyId;

	/// The transform of the fragment at the time of the event
	b3WorldTransform transform;

	/// The fragment user data
	uint64_t userData;
} b3FragmentEvent;

/// Why a bond broke
typedef enum b3BondBreakCause
{
	/// The sustained load went over the fatal limit. The structure gave way.
	b3_bondBreakOverload,

	/// The bond was held between the elastic and the fatal limit for too long
	b3_bondBreakCreep,

	/// A blow broke the bond
	b3_bondBreakImpact,

	/// The application broke the bond or damaged it until it broke
	b3_bondBreakDamage,

	/// A fragment of the bond was crushed or removed
	b3_bondBreakCrush,
} b3BondBreakCause;

/// A bond broke. Useful for sound and dust.
typedef struct b3BondEvent
{
	/// The destructible
	b3DestructibleId destructibleId;

	/// The bond index. @see b3Destructible_GetBondState
	int bondIndex;

	/// The first fragment
	int fragmentA;

	/// The second fragment or B3_NULL_INDEX for an anchor
	int fragmentB;

	/// World centroid of the broken surface
	b3Pos point;

	/// World normal of the broken surface
	b3Vec3 normal;

	/// Area of the broken surface
	float area;

	/// Why the bond broke
	b3BondBreakCause cause;
} b3BondEvent;

/// The destruction solver created a body.
typedef struct b3DestructionBodyEvent
{
	/// The destructible
	b3DestructibleId destructibleId;

	/// The new body
	b3BodyId bodyId;

	/// The body the fragments came from
	b3BodyId parentBodyId;

	/// The number of fragments on the new body
	int fragmentCount;

	/// Is the new body debris?
	bool isDebris;
} b3DestructionBodyEvent;

/// Destruction events are buffered in the world and are available as event arrays after the time
/// step is complete. The event data is transient. Do not store a reference to this data.
typedef struct b3DestructionEvents
{
	/// Array of fragment events
	b3FragmentEvent* fragmentEvents;

	/// Array of bond events
	b3BondEvent* bondEvents;

	/// Array of body events
	b3DestructionBodyEvent* bodyEvents;

	/// The number of fragment events
	int fragmentCount;

	/// The number of bond events
	int bondCount;

	/// The number of body events
	int bodyCount;
} b3DestructionEvents;

/// The current state of a bond
typedef struct b3BondState
{
	/// The first fragment
	int fragmentA;

	/// The second fragment or B3_NULL_INDEX for an anchor
	int fragmentB;

	/// Centroid in structure space
	b3Vec3 centroid;

	/// Unit normal in structure space, pointing from fragment A to fragment B
	b3Vec3 normal;

	/// The undamaged area
	float area;

	/// Remaining strength in the range [0, 1]
	float health;

	/// Stress relative to the elastic limit from the latest solve. Damage starts at one.
	float elasticRatio;

	/// Stress relative to the fatal limit from the latest solve. The bond breaks at one.
	float fatalRatio;

	/// Has this bond broken?
	bool isBroken;
} b3BondState;

/// Destruction profiling data. Times are in milliseconds.
typedef struct b3DestructionProfile
{
	float total;   ///< All destruction work of the latest step
	float loads;   ///< Gathering contact impulses
	float solve;   ///< Stress solve and damage, across all workers
	float fracture; ///< Breaking bonds, splitting pieces, and changing bodies
} b3DestructionProfile;

/// Counters that give details of the destruction size.
typedef struct b3DestructionCounters
{
	int destructibleCount; ///< Destructibles in the world
	int pieceCount;		   ///< Pieces across all destructibles
	int bodyCount;		   ///< Bodies owned by destructibles
	int bondCount;		   ///< Intact bonds across all destructibles
	int solvedPieceCount;  ///< Pieces that ran the stress solve in the latest step
	int iterationCount;	   ///< Solver iterations in the latest step, across all pieces
	int brokenBondCount;   ///< Bonds that broke in the latest step
	int newBodyCount;	   ///< Bodies created in the latest step
	int pendingSplitCount; ///< Pieces still waiting to be split because of the body budget
	int64_t byteCount;	   ///< Bytes held by the destruction solver
} b3DestructionCounters;

/// Create a destructible from a cooked structure. This creates the bodies and the fragment shapes.
/// @warning This function is locked during callbacks.
B3_API b3DestructibleId b3CreateDestructible( b3WorldId worldId, const b3DestructibleDef* def );

/// Destroy a destructible and all of its bodies.
B3_API void b3DestroyDestructible( b3DestructibleId destructibleId );

/// Destructible identifier validation. Provides validation for up to 64K allocations.
B3_API bool b3Destructible_IsValid( b3DestructibleId id );

/// Get the world that owns this destructible
B3_API b3WorldId b3Destructible_GetWorld( b3DestructibleId destructibleId );

/// Get the structure this destructible was created from
B3_API const b3StructureData* b3Destructible_GetStructure( b3DestructibleId destructibleId );

/// Set the user data for a destructible. This does not change the user data of its bodies.
B3_API void b3Destructible_SetUserData( b3DestructibleId destructibleId, void* userData );

/// Get the user data stored in a destructible
B3_API void* b3Destructible_GetUserData( b3DestructibleId destructibleId );

/// Get the number of bodies owned by this destructible
B3_API int b3Destructible_GetBodyCount( b3DestructibleId destructibleId );

/// Get the bodies owned by this destructible, up to the provided capacity.
/// @returns the number of body ids stored in the user array
B3_API int b3Destructible_GetBodies( b3DestructibleId destructibleId, b3BodyId* bodyArray, int capacity );

/// Get the body a fragment is attached to. Returns a null id if the fragment is not simulated.
B3_API b3BodyId b3Destructible_GetFragmentBody( b3DestructibleId destructibleId, int fragmentIndex );

/// Get the remaining health of a fragment in joules.
B3_API float b3Destructible_GetFragmentHealth( b3DestructibleId destructibleId, int fragmentIndex );

/// Get the number of bonds. This counts the structure bonds followed by the anchors created with
/// the destructible, broken or not.
B3_API int b3Destructible_GetBondCount( b3DestructibleId destructibleId );

/// Get the number of bonds that are not broken
B3_API int b3Destructible_GetIntactBondCount( b3DestructibleId destructibleId );

/// Get the state of a bond
B3_API b3BondState b3Destructible_GetBondState( b3DestructibleId destructibleId, int bondIndex );

/// Break a bond. The split happens in the next world step.
B3_API void b3Destructible_BreakBond( b3DestructibleId destructibleId, int bondIndex );

/// Damage a fragment directly, for example from a bullet. A fragment that runs out of health is
/// crushed in the next world step: its bonds break and its children are released as debris.
/// @param destructibleId The destructible
/// @param fragmentIndex The fragment index in the structure
/// @param damage The energy to remove from the fragment health, usually in joules
B3_API void b3Destructible_DamageFragment( b3DestructibleId destructibleId, int fragmentIndex, float damage );

/// Apply an impulse to a fragment. Unlike b3Body_ApplyLinearImpulse this also stresses the bonds
/// around the fragment as an impact, so it can break a structure that is anchored.
/// @param destructibleId The destructible
/// @param fragmentIndex The fragment index in the structure
/// @param impulse The world impulse vector, usually in N*s or kg*m/s
/// @param point The world position of the point of application
B3_API void b3Destructible_ApplyImpulse( b3DestructibleId destructibleId, int fragmentIndex, b3Vec3 impulse, b3Pos point );

/// Scale every material strength of this destructible. One is the cooked strength.
B3_API void b3Destructible_SetStrengthScale( b3DestructibleId destructibleId, float scale );

/// Get the strength scale
B3_API float b3Destructible_GetStrengthScale( b3DestructibleId destructibleId );

/// Get the destructible that owns a shape. Returns a null id if the shape is not a fragment.
/// This is useful when you get a shape id from a ray cast or an event.
B3_API b3DestructibleId b3Shape_GetDestructible( b3ShapeId shapeId );

/// Get the fragment a shape belongs to. Returns B3_NULL_INDEX if the shape is not a fragment.
B3_API int b3Shape_GetFragmentIndex( b3ShapeId shapeId );

/// Get the destructible that owns a body. Returns a null id if the body is not owned by one.
B3_API b3DestructibleId b3Body_GetDestructible( b3BodyId bodyId );

/// Adjust the destruction solver for a world
B3_API void b3World_SetDestructionSettings( b3WorldId worldId, const b3DestructionSettings* settings );

/// Get the destruction solver settings of a world
B3_API b3DestructionSettings b3World_GetDestructionSettings( b3WorldId worldId );

/// Get the destruction events for the current time step. The event data is transient. Do not
/// store a reference to this data.
B3_API b3DestructionEvents b3World_GetDestructionEvents( b3WorldId worldId );

/// Damage every fragment and bond in a sphere. The damage is applied in the next world step.
/// Pair this with b3World_Explode to also push the pieces.
B3_API void b3World_ApplyRadialDamage( b3WorldId worldId, const b3RadialDamageDef* def );

/// Get the destruction profile of the latest step
B3_API b3DestructionProfile b3World_GetDestructionProfile( b3WorldId worldId );

/// Get destruction counters and sizes
B3_API b3DestructionCounters b3World_GetDestructionCounters( b3WorldId worldId );

/**@}*/ // destruction
