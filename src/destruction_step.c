// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

// The per step work of the destruction solver: loads in, fractures out.
// See destruction.h for the overall flow.

#include "arena_allocator.h"
#include "bitset.h"
#include "body.h"
#include "contact.h"
#include "core.h"
#include "ctz.h"
#include "destruction.h"
#include "math_internal.h"
#include "parallel_for.h"
#include "physics_world.h"
#include "recording.h"
#include "shape.h"
#include "solver.h"

#include "box3d/box3d.h"
#include "box3d/destruction.h"

#include <float.h>
#include <string.h>

// A fragment ignores impacts that carry less than this share of its health. Keeps the countless small
// bumps of resting contact from wearing a structure down.
#define B3_MIN_IMPACT_HEALTH_FRACTION 0.05f

// A dynamic piece spinning with a rim speed above this gets a stress solve even without contacts.
#define B3_SPIN_SPEED_THRESHOLD 10.0f

static inline bool b3IsLiveDestructible( const b3Destructible* destructible )
{
	return destructible->id != B3_NULL_INDEX && destructible->structure != NULL;
}

void b3DestructionPreStep( b3World* world, float timeStep )
{
	b3Destruction* destruction = world->destruction;

	b3Array_Clear( destruction->fragmentEvents );
	b3Array_Clear( destruction->bondEvents );
	b3Array_Clear( destruction->bodyEvents );
	b3Array_Clear( destruction->contactLoads );
	b3Array_Clear( destruction->tasks );

	destruction->profile = (b3DestructionProfile){ 0 };
	destruction->solvedPieceCount = 0;
	destruction->iterationCount = 0;
	destruction->brokenBondCount = 0;
	destruction->newBodyCount = 0;
	destruction->timeStep = timeStep;
	destruction->inverseTimeStep = timeStep > 0.0f ? 1.0f / timeStep : 0.0f;

	// The constraint solver sizes these when it runs. Left empty, the analysis sees no contacts.
	for ( int i = 0; i < world->taskContexts.count; ++i )
	{
		world->taskContexts.data[i].destructionBitSet.blockCount = 0;
	}

	if ( timeStep <= 0.0f )
	{
		return;
	}

	// Make anchored pieces follow their host
	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		b3Destructible* destructible = destruction->destructibles.data + i;
		if ( b3IsLiveDestructible( destructible ) == false || destructible->anchorBodyId == B3_NULL_INDEX ||
			 B3_IS_NULL( destructible->hostBodyId ) )
		{
			continue;
		}

		b3BodyId anchorBodyId = b3MakeBodyId( world, destructible->anchorBodyId );

		if ( b3Body_IsValid( destructible->hostBodyId ) == false )
		{
			// The host is gone. Stay where we are.
			destructible->hostBodyId = b3_nullBodyId;
			b3Body_SetLinearVelocity( anchorBodyId, b3Vec3_zero );
			b3Body_SetAngularVelocity( anchorBodyId, b3Vec3_zero );
			destructible->anchorLinearAcceleration = b3Vec3_zero;
			destructible->anchorAngularAcceleration = b3Vec3_zero;
			destructible->anchorAngularVelocity = b3Vec3_zero;
			destructible->anchorLoadsDirty = true;
			continue;
		}

		// Aim at where the host will be at the end of this step, so the anchor body moves in lock step
		// and any drift is removed.
		b3WorldTransform hostTransform = b3Body_GetTransform( destructible->hostBodyId );
		b3Vec3 hostVelocity = b3Body_GetLocalPointVelocity( destructible->hostBodyId, b3Vec3_zero );
		b3Vec3 hostAngularVelocity = b3Body_GetAngularVelocity( destructible->hostBodyId );

		b3WorldTransform predicted;
		predicted.p = b3OffsetPos( hostTransform.p, b3MulSV( timeStep, hostVelocity ) );
		predicted.q = b3IntegrateRotation( hostTransform.q, b3MulSV( timeStep, hostAngularVelocity ) );

		b3WorldTransform target = b3MulWorldTransforms( predicted, destructible->hostFrame );
		b3Body_SetTargetTransform( anchorBodyId, target, timeStep, true );

		// The acceleration of the anchor frame loads the structure like extra gravity.
		b3Vec3 linearVelocity = b3Body_GetLinearVelocity( anchorBodyId );
		b3Vec3 angularVelocity = b3Body_GetAngularVelocity( anchorBodyId );
		b3Quat rotation = b3Body_GetRotation( anchorBodyId );
		float inverseTimeStep = 1.0f / timeStep;

		b3Vec3 linearAcceleration = b3MulSV( inverseTimeStep, b3Sub( linearVelocity, destructible->hostLinearVelocity ) );
		b3Vec3 angularAcceleration = b3MulSV( inverseTimeStep, b3Sub( angularVelocity, destructible->hostAngularVelocity ) );
		destructible->hostLinearVelocity = linearVelocity;
		destructible->hostAngularVelocity = angularVelocity;

		b3Vec3 a0 = b3InvRotateVector( rotation, linearAcceleration );
		b3Vec3 alpha = b3InvRotateVector( rotation, angularAcceleration );
		b3Vec3 w = b3InvRotateVector( rotation, angularVelocity );

		float change = b3DistanceSquared( a0, destructible->anchorLinearAcceleration ) +
					   b3DistanceSquared( alpha, destructible->anchorAngularAcceleration ) +
					   b3DistanceSquared( w, destructible->anchorAngularVelocity );
		if ( change > 1.0e-6f )
		{
			destructible->anchorLoadsDirty = true;
		}

		destructible->anchorLinearAcceleration = a0;
		destructible->anchorAngularAcceleration = alpha;
		destructible->anchorAngularVelocity = w;
	}
}

// Turn one solved contact into the load on the fragment at either end.
// Does the fragment of a shape have any use for the load of a contact?
static inline bool b3WantsContactLoad( const b3Destruction* destruction, b3ShapeFragment link, float approachSpeed )
{
	const b3Destructible* destructible = destruction->destructibles.data + link.destructibleId;
	int pieceIndex = destructible->fragmentPiece[link.fragmentIndex];
	if ( pieceIndex < 0 )
	{
		return false;
	}

	const b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	if ( piece->flags & b3_pieceFrozen )
	{
		return false;
	}

	// A loose fragment has no bonds to load. Only a blow matters to it, for the energy that can
	// crush it. Most contacts in a pile of rubble end here.
	return piece->bondCount > 0 || approachSpeed > destruction->settings.impactVelocityThreshold;
}

static void b3GatherContact( b3World* world, int contactId, int subStepCount, b3ContactLoad* loads )
{
	const b3Destruction* destruction = world->destruction;
	loads[0].destructibleId = B3_NULL_INDEX;
	loads[1].destructibleId = B3_NULL_INDEX;

	const b3Contact* contact = world->contacts.data + contactId;
	const b3Shape* shapeA = world->shapes.data + contact->shapeIdA;
	const b3Shape* shapeB = world->shapes.data + contact->shapeIdB;

	b3ShapeFragment none = { B3_NULL_INDEX, B3_NULL_INDEX, false };
	b3ShapeFragment linkA = ( shapeA->flags & b3_destructibleShape ) != 0 ? destruction->shapeFragments.data[contact->shapeIdA] : none;
	b3ShapeFragment linkB = ( shapeB->flags & b3_destructibleShape ) != 0 ? destruction->shapeFragments.data[contact->shapeIdB] : none;
	if ( linkA.destructibleId == B3_NULL_INDEX && linkB.destructibleId == B3_NULL_INDEX )
	{
		return;
	}

	// Sum the impulse body B received over the step. The normal impulse is the total across the
	// sub-steps. The friction impulse is per sub-step.
	b3Vec3 impulse = b3Vec3_zero;
	b3Vec3 anchorA = b3Vec3_zero;
	b3Vec3 anchorB = b3Vec3_zero;
	float normalSum = 0.0f;
	float approachSpeed = 0.0f;

	for ( int i = 0; i < contact->manifoldCount; ++i )
	{
		const b3Manifold* manifold = contact->manifolds + i;
		float manifoldSum = 0.0f;

		for ( int j = 0; j < manifold->pointCount; ++j )
		{
			const b3ManifoldPoint* mp = manifold->points + j;
			float normalImpulse = mp->totalNormalImpulse;
			if ( normalImpulse <= 0.0f )
			{
				// A speculative point that did not touch
				continue;
			}

			manifoldSum += normalImpulse;
			anchorA = b3MulAdd( anchorA, normalImpulse, mp->anchorA );
			anchorB = b3MulAdd( anchorB, normalImpulse, mp->anchorB );
			approachSpeed = b3MaxFloat( approachSpeed, -mp->normalVelocity );
		}

		if ( manifoldSum > 0.0f )
		{
			impulse = b3MulAdd( impulse, manifoldSum, manifold->normal );
			impulse = b3MulAdd( impulse, (float)subStepCount, manifold->frictionImpulse );
			normalSum += manifoldSum;
		}
	}

	if ( normalSum <= 0.0f )
	{
		return;
	}

	float inverseSum = 1.0f / normalSum;
	const b3Body* bodyA = world->bodies.data + shapeA->bodyId;
	const b3Body* bodyB = world->bodies.data + shapeB->bodyId;

	if ( linkA.destructibleId != B3_NULL_INDEX && b3WantsContactLoad( destruction, linkA, approachSpeed ) )
	{
		b3ContactLoad* load = loads + 0;
		load->destructibleId = linkA.destructibleId;
		load->fragmentIndex = linkA.fragmentIndex;
		load->otherBodyId = bodyB->id;
		load->otherGeneration = bodyB->generation;
		load->impulse = b3Neg( impulse );
		load->anchor = b3MulSV( inverseSum, anchorA );
		load->approachSpeed = approachSpeed;
	}

	if ( linkB.destructibleId != B3_NULL_INDEX && b3WantsContactLoad( destruction, linkB, approachSpeed ) )
	{
		b3ContactLoad* load = loads + 1;
		load->destructibleId = linkB.destructibleId;
		load->fragmentIndex = linkB.fragmentIndex;
		load->otherBodyId = bodyA->id;
		load->otherGeneration = bodyA->generation;
		load->impulse = impulse;
		load->anchor = b3MulSV( inverseSum, anchorB );
		load->approachSpeed = approachSpeed;
	}
}

typedef struct b3GatherContext
{
	b3World* world;
	const uint64_t* bits;

	// Number of flagged contacts before each word. Gives every contact a fixed slot, so the result
	// does not depend on which worker handles which word.
	const int* slotStarts;
	b3ContactLoad* loads;
	int subStepCount;
} b3GatherContext;

static void b3GatherContactsTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	B3_UNUSED( workerIndex );
	b3GatherContext* gatherContext = context;

	for ( int wordIndex = startIndex; wordIndex < endIndex; ++wordIndex )
	{
		uint64_t word = gatherContext->bits[wordIndex];
		int slot = gatherContext->slotStarts[wordIndex];

		while ( word != 0 )
		{
			uint32_t ctz = b3CTZ64( word );
			int contactId = (int)( 64 * wordIndex + ctz );
			b3GatherContact( gatherContext->world, contactId, gatherContext->subStepCount, gatherContext->loads + 2 * slot );
			slot += 1;

			// Clear the smallest set bit
			word = word & ( word - 1 );
		}
	}
}

// Gather the loads of every contact the constraint solver flagged in this step.
static void b3GatherContactLoads( b3World* world, int subStepCount )
{
	b3Destruction* destruction = world->destruction;

	// Union the per worker bits into worker 0, like the hit events
	b3BitSet* bitSet = &world->taskContexts.data[0].destructionBitSet;
	if ( bitSet->blockCount == 0 )
	{
		// The constraint solver did not run
		return;
	}

	for ( int i = 1; i < world->workerCount; ++i )
	{
		b3InPlaceUnion( bitSet, &world->taskContexts.data[i].destructionBitSet );
	}

	int wordCount = (int)bitSet->blockCount;
	int* slotStarts = b3StackAlloc( &world->stack, wordCount * sizeof( int ), "destruction slots" );
	int contactCount = 0;
	for ( int i = 0; i < wordCount; ++i )
	{
		slotStarts[i] = contactCount;
		contactCount += b3PopCount64( bitSet->bits[i] );
	}

	if ( contactCount > 0 )
	{
		b3Array_Resize( destruction->contactLoads, 2 * contactCount );

		b3GatherContext gatherContext = { world, bitSet->bits, slotStarts, destruction->contactLoads.data, subStepCount };
		b3ParallelFor( world, b3GatherContactsTask, wordCount, 64, &gatherContext, "destruction loads" );
	}

	b3StackFree( &world->stack, slotStarts );
}

static inline void b3AddLoadedNode( b3Destructible* destructible, int node )
{
	if ( ( destructible->nodeFlags[node] & b3_nodeLoaded ) == 0 )
	{
		destructible->nodeFlags[node] |= b3_nodeLoaded;
		b3Array_Push( destructible->loadedNodes, node );
	}
}

static inline void b3AddCommitPiece( b3Destructible* destructible, int pieceIndex )
{
	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	if ( ( piece->flags & b3_pieceInCommitList ) == 0 )
	{
		piece->flags |= b3_pieceInCommitList;
		b3Array_Push( destructible->commitPieces, pieceIndex );
	}
}

// Queue a bond break outside of the stress solve
static void b3RequestBondBreak( b3Destructible* destructible, int bondIndex )
{
	b3DestructionBond* bond = destructible->bonds + bondIndex;
	if ( bond->flags & ( b3_bondBroken | b3_bondBreaking ) )
	{
		return;
	}

	int pieceIndex = destructible->fragmentPiece[bond->nodeA];
	if ( pieceIndex < 0 )
	{
		return;
	}

	bond->flags |= b3_bondBreaking;
	b3BondBreak bondBreak = { bondIndex, 0.0f, 0.0f, b3_bondBreakDamage };
	b3Array_Push( destructible->pieces.data[pieceIndex].breaks, bondBreak );
	b3AddCommitPiece( destructible, pieceIndex );
}

static void b3RequestCrush( b3Destructible* destructible, int node )
{
	if ( destructible->fragmentPiece[node] < 0 || ( destructible->nodeFlags[node] & b3_nodeCrushed ) != 0 )
	{
		return;
	}

	destructible->nodeFlags[node] |= b3_nodeCrushed;
	b3Array_Push( destructible->crushedNodes, node );
}

// Apply the requests the application made since the previous step.
static void b3ApplyCommands( b3Destructible* destructible )
{
	for ( int i = 0; i < destructible->commands.count; ++i )
	{
		const b3DestructionCommand* command = destructible->commands.data + i;

		switch ( command->type )
		{
			case b3_commandBreakBond:
				b3RequestBondBreak( destructible, command->index );
				break;

			case b3_commandDamageFragment:
			{
				int node = command->index;
				if ( destructible->fragmentPiece[node] < 0 )
				{
					break;
				}

				destructible->health[node] -= command->value;
				if ( destructible->health[node] <= 0.0f )
				{
					b3RequestCrush( destructible, node );
				}
			}
			break;

			case b3_commandDamageBond:
			{
				b3DestructionBond* bond = destructible->bonds + command->index;
				if ( bond->flags & ( b3_bondBroken | b3_bondBreaking | b3_bondRigid ) )
				{
					break;
				}

				bond->area *= 1.0f - b3ClampFloat( command->value, 0.0f, 1.0f );
				if ( bond->area < 0.05f * bond->area0 )
				{
					b3RequestBondBreak( destructible, command->index );
					break;
				}

				// The stiffness changed, so the piece must rebuild its solver arrays
				int pieceIndex = destructible->fragmentPiece[bond->nodeA];
				if ( pieceIndex >= 0 )
				{
					destructible->pieces.data[pieceIndex].flags |= b3_pieceSystemDirty;
				}
			}
			break;

			case b3_commandImpulse:
			{
				int node = command->index;
				if ( destructible->fragmentPiece[node] < 0 )
				{
					break;
				}

				b3NodeLoad* load = destructible->loads + node;
				load->impulse = b3Add( load->impulse, command->impulse );
				load->angularImpulse = b3Add( load->angularImpulse, command->angularImpulse );
				load->externalImpulse = b3Add( load->externalImpulse, command->impulse );
				load->externalAngularImpulse = b3Add( load->externalAngularImpulse, command->angularImpulse );

				// The push wants to move this fragment alone, the bonds make it take the rest of the
				// piece along. The kinetic energy lost in between is what can break those bonds, the
				// same as in a collision where the two bodies stick together.
				const b3DestructionPiece* piece = destructible->pieces.data + destructible->fragmentPiece[node];
				float mass = b3GetStructureFragments( destructible->structure )[node].mass;
				float carried = 1.0f;
				if ( ( piece->flags & b3_pieceAnchored ) == 0 )
				{
					carried = piece->mass > mass ? 1.0f - mass / piece->mass : 0.0f;
				}

				if ( mass > 0.0f )
				{
					load->energy += carried * 0.5f * b3LengthSquared( command->impulse ) / mass;
				}

				b3AddLoadedNode( destructible, node );
			}
			break;

			default:
				B3_ASSERT( false );
				break;
		}
	}

	b3Array_Clear( destructible->commands );
}

// Hand the contact loads of this step to their fragments, in structure space.
static void b3DistributeContactLoads( b3World* world )
{
	b3Destruction* destruction = world->destruction;
	float impactSpeed = destruction->settings.impactVelocityThreshold;

	for ( int i = 0; i < destruction->contactLoads.count; ++i )
	{
		b3ContactLoad* contactLoad = destruction->contactLoads.data + i;
		if ( contactLoad->destructibleId == B3_NULL_INDEX )
		{
			continue;
		}

		b3Destructible* destructible = destruction->destructibles.data + contactLoad->destructibleId;
		int node = contactLoad->fragmentIndex;
		int pieceIndex = destructible->fragmentPiece[node];
		if ( pieceIndex < 0 )
		{
			contactLoad->destructibleId = B3_NULL_INDEX;
			continue;
		}

		const b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
		if ( piece->flags & b3_pieceFrozen )
		{
			contactLoad->destructibleId = B3_NULL_INDEX;
			continue;
		}

		b3Body* body = world->bodies.data + piece->bodyId;
		const b3BodySim* bodySim = b3GetBodySim( world, body );
		b3Quat rotation = bodySim->transform.q;
		const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + node;

		// The body frame is structure space, so going local is a single rotation.
		b3Vec3 localImpulse = b3InvRotateVector( rotation, contactLoad->impulse );
		b3Vec3 localPoint = b3Add( bodySim->localCenter, b3InvRotateVector( rotation, contactLoad->anchor ) );
		contactLoad->point = b3OffsetPos( bodySim->center, contactLoad->anchor );

		b3NodeLoad* load = destructible->loads + node;
		load->impulse = b3Add( load->impulse, localImpulse );
		load->angularImpulse = b3Add( load->angularImpulse, b3Cross( b3Sub( localPoint, fragment->center ), localImpulse ) );

		// Energy the impact can spend on breaking the fragment and the bonds around it. Half the
		// impulse times the approach speed adds up to the kinetic energy lost, however the impulse is
		// shared between contacts. Two destructibles that hit each other split that energy.
		if ( contactLoad->approachSpeed > impactSpeed )
		{
			int otherBodyId = contactLoad->otherBodyId;
			bool isShared = otherBodyId < destruction->bodyOwners.count && destruction->bodyOwners.data[otherBodyId] != B3_NULL_INDEX;
			float share = isShared ? 0.5f : 1.0f;
			load->energy += share * 0.5f * b3Length( contactLoad->impulse ) * contactLoad->approachSpeed;
		}

		b3AddLoadedNode( destructible, node );
	}
}

// Update the smoothed loads of a destructible and find impacts and crushed fragments.
static void b3UpdateLoads( b3World* world, b3Destructible* destructible )
{
	const b3Destruction* destruction = world->destruction;
	const b3DestructionSettings* settings = &destruction->settings;
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );

	float timeStep = destruction->timeStep;
	float inverseTimeStep = destruction->inverseTimeStep;
	float blend = timeStep / ( settings->loadSmoothingTime + timeStep );
	float gravityLength = b3Length( world->gravity );

	// Flags that only live for one step
	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		b3DestructionPiece* piece = destructible->pieces.data + i;
		piece->flags &= ~( b3_pieceHasLoads | b3_pieceHasImpact | b3_pieceNew | b3_pieceWasStatic );
		piece->parentPiece = B3_NULL_INDEX;
		piece->impactExcess = 0.0f;
		piece->impactEnergy = 0.0f;
	}

	if ( b3DistanceSquared( world->gravity, destructible->gravity ) > 0.0f )
	{
		destructible->gravity = world->gravity;
		destructible->anchorLoadsDirty = true;
	}

	int keepCount = 0;
	for ( int i = 0; i < destructible->loadedNodes.count; ++i )
	{
		int node = destructible->loadedNodes.data[i];
		b3NodeLoad* load = destructible->loads + node;
		destructible->nodeFlags[node] &= ~b3_nodeImpact;

		int pieceIndex = destructible->fragmentPiece[node];
		if ( pieceIndex < 0 )
		{
			// The fragment is gone
			*load = (b3NodeLoad){ 0 };
			destructible->nodeFlags[node] &= ~b3_nodeLoaded;
			continue;
		}

		b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
		const b3StructureFragment* fragment = fragments + node;
		float weight = fragment->mass * gravityLength;

		// The part of this step's impulse above the smoothed force is a blow. It counts as an impact
		// when it could change the velocity of the fragment by more than the threshold.
		b3Vec3 excess = b3MulSub( load->impulse, timeStep, load->force );
		b3Vec3 angularExcess = b3MulSub( load->angularImpulse, timeStep, load->torque );
		float threshold = fragment->mass * settings->impactVelocityThreshold + 0.5f * timeStep * b3Length( load->force );
		bool isImpact = b3LengthSquared( excess ) > threshold * threshold;
		bool hadImpulse = b3LengthSquared( load->impulse ) > 0.0f;

		// Smooth the contact force. Contact impulses jitter from step to step, the structure should not.
		load->force = b3Lerp( load->force, b3MulSV( inverseTimeStep, load->impulse ), blend );
		load->torque = b3Lerp( load->torque, b3MulSV( inverseTimeStep, load->angularImpulse ), blend );

		if ( isImpact )
		{
			load->impulse = excess;
			load->angularImpulse = angularExcess;
			destructible->nodeFlags[node] |= b3_nodeImpact;
			piece->flags |= b3_pieceHasImpact;
		}
		else
		{
			load->impulse = b3Vec3_zero;
			load->angularImpulse = b3Vec3_zero;
		}

		// Impact energy wears the fragment down and pays for the bonds the impact breaks
		float energy = settings->impactDamageScale * load->energy;
		piece->impactEnergy += settings->impactFractureScale * load->energy;
		load->energy = 0.0f;
		float fullHealth = destructible->strengthScale * fragment->health;
		if ( energy > B3_MIN_IMPACT_HEALTH_FRACTION * fullHealth )
		{
			destructible->health[node] -= energy;
			if ( destructible->health[node] <= 0.0f )
			{
				b3RequestCrush( destructible, node );
			}
		}

		// Drop the load once the contact is gone and the smoothed force has died out. A push that is
		// still waiting for a deferred crush or split keeps the load alive.
		bool hasPush = b3LengthSquared( load->externalImpulse ) + b3LengthSquared( load->externalAngularImpulse ) > 0.0f;
		bool isSpent = hadImpulse == false && hasPush == false && b3LengthSquared( load->force ) < 1.0e-6f * weight * weight;

		// A loose fragment has no bonds for a force to load. The blow was all that mattered to it.
		if ( piece->bondCount == 0 && hasPush == false )
		{
			isSpent = true;
		}

		if ( isSpent )
		{
			if ( b3LengthSquared( load->solvedForce ) > 0.0f )
			{
				piece->flags |= b3_pieceLoadDirty;
			}

			*load = (b3NodeLoad){ 0 };
			destructible->nodeFlags[node] &= ~b3_nodeLoaded;
			continue;
		}

		piece->flags |= b3_pieceHasLoads;

		// Restart the solve of the piece once the smoothed force has drifted from what it last used
		float reference = b3MaxFloat( b3Length( load->solvedForce ), weight );
		if ( b3DistanceSquared( load->force, load->solvedForce ) > 0.0025f * reference * reference )
		{
			piece->flags |= b3_pieceLoadDirty;
		}

		destructible->loadedNodes.data[keepCount++] = node;
	}

	destructible->loadedNodes.count = keepCount;

	if ( destructible->anchorLoadsDirty )
	{
		for ( int i = 0; i < destructible->pieces.count; ++i )
		{
			b3DestructionPiece* piece = destructible->pieces.data + i;
			if ( piece->bodyId != B3_NULL_INDEX && ( piece->flags & b3_pieceAnchored ) != 0 )
			{
				piece->flags |= b3_pieceLoadDirty;
			}
		}

		destructible->anchorLoadsDirty = false;
	}
}

// Pick the pieces that need a stress solve in this step.
static void b3BuildTasks( b3World* world, b3Destructible* destructible )
{
	b3Destruction* destruction = world->destruction;

	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		const b3DestructionPiece* piece = destructible->pieces.data + i;
		if ( piece->bodyId == B3_NULL_INDEX )
		{
			continue;
		}

		// A piece that is waiting to be split may be in several parts. Solving it would be meaningless.
		if ( piece->flags & ( b3_pieceFrozen | b3_pieceTopologyDirty ) )
		{
			continue;
		}

		bool isDirty = ( piece->flags & b3_pieceSystemDirty ) != 0;
		if ( isDirty == false && piece->bondCount == 0 )
		{
			continue;
		}

		bool needsSolve;
		if ( piece->flags & b3_pieceAnchored )
		{
			needsSolve = isDirty || ( piece->flags & ( b3_pieceLoadDirty | b3_pieceCreeping | b3_pieceHasImpact ) ) != 0 ||
						 piece->system == NULL || piece->system->evaluated == false;
		}
		else
		{
			b3Body* body = world->bodies.data + piece->bodyId;
			if ( body->setIndex != b3_awakeSet )
			{
				// Nothing changes while the body sleeps
				continue;
			}

			needsSolve = isDirty || ( piece->flags & ( b3_pieceHasLoads | b3_pieceCreeping | b3_pieceHasImpact ) ) != 0 ||
						 ( piece->flags & b3_pieceUnloaded ) == 0;

			if ( needsSolve == false )
			{
				// A fast spin pulls a piece apart even in free flight
				const b3BodyState* state = b3GetBodyState( world, body );
				float rimSpeed = b3Length( state->angularVelocity ) * piece->radius;
				needsSolve = rimSpeed > B3_SPIN_SPEED_THRESHOLD;
			}
		}

		if ( needsSolve )
		{
			b3PieceTask task = { destructible->id, i, 0 };
			b3Array_Push( destruction->tasks, task );
		}
	}
}

static void b3SolvePiecesTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	b3World* world = context;
	b3Destruction* destruction = world->destruction;
	b3Arena arena = world->taskContexts.data[workerIndex].arena;

	for ( int i = startIndex; i < endIndex; ++i )
	{
		b3PieceTask* task = destruction->tasks.data + i;
		b3Destructible* destructible = destruction->destructibles.data + task->destructibleId;
		task->iterations = b3SolvePieceStress( world, destructible, task->pieceIndex, arena );
	}
}

void b3DestructionAnalyze( b3World* world, b3StepContext* context )
{
	b3Destruction* destruction = world->destruction;
	destruction->subStepCount = context->subStepCount;

	b3TracyCZoneNC( destruction_analyze, "Destruction", b3_colorOrangeRed, true );

	uint64_t ticks = b3GetTicks();

	b3GatherContactLoads( world, context->subStepCount );

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		b3Destructible* destructible = destruction->destructibles.data + i;
		if ( b3IsLiveDestructible( destructible ) )
		{
			b3ApplyCommands( destructible );
		}
	}

	b3DistributeContactLoads( world );

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		b3Destructible* destructible = destruction->destructibles.data + i;
		if ( b3IsLiveDestructible( destructible ) )
		{
			b3UpdateLoads( world, destructible );
			b3BuildTasks( world, destructible );
		}
	}

	destruction->profile.loads = b3GetMillisecondsAndReset( &ticks );

	// The pieces are independent, so they solve in parallel. Each piece gives the same result no
	// matter which worker runs it.
	int taskCount = destruction->tasks.count;
	if ( taskCount > 0 )
	{
		b3ParallelFor( world, b3SolvePiecesTask, taskCount, 1, world, "destruction solve" );
	}

	for ( int i = 0; i < taskCount; ++i )
	{
		const b3PieceTask* task = destruction->tasks.data + i;
		b3Destructible* destructible = destruction->destructibles.data + task->destructibleId;
		destruction->iterationCount += task->iterations;

		if ( destructible->pieces.data[task->pieceIndex].breaks.count > 0 )
		{
			b3AddCommitPiece( destructible, task->pieceIndex );
		}
	}

	destruction->solvedPieceCount = taskCount;
	destruction->profile.solve = b3GetMilliseconds( ticks );

	b3TracyCZoneEnd( destruction_analyze );
}

// Mark a bond broken and report it.
static void b3BreakBond( b3World* world, b3Destructible* destructible, int bondIndex, b3BondBreakCause cause )
{
	b3Destruction* destruction = world->destruction;
	b3DestructionBond* bond = destructible->bonds + bondIndex;
	if ( bond->flags & b3_bondBroken )
	{
		return;
	}

	bond->flags |= b3_bondBroken;
	bond->flags &= ~b3_bondBreaking;
	bond->solverIndex = B3_NULL_INDEX;
	bond->elasticRatio = 0.0f;
	bond->fatalRatio = 0.0f;
	destructible->intactBondCount -= 1;
	destruction->brokenBondCount += 1;

	int pieceIndex = destructible->fragmentPiece[bond->nodeA];
	if ( pieceIndex < 0 )
	{
		return;
	}

	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	piece->bondCount -= 1;
	piece->flags |= b3_pieceTopologyDirty | b3_pieceSystemDirty;

	if ( destructible->enableBondEvents )
	{
		const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + bond->nodeA;
		b3WorldTransform transform = b3GetBodyTransform( world, piece->bodyId );

		b3BondEvent event = { 0 };
		event.destructibleId = b3MakeDestructibleId( world, destructible );
		event.bondIndex = bondIndex;
		event.fragmentA = bond->nodeA;
		event.fragmentB = bond->nodeB;
		event.point = b3TransformWorldPoint( transform, b3Add( fragment->center, bond->rA ) );
		event.normal = b3RotateVector( transform.q, bond->normal );
		event.area = bond->area0;
		event.cause = cause;
		b3Array_Push( destruction->bondEvents, event );
	}
}

// Move the shapes of some fragments from one body to another.
static void b3MoveFragmentShapes( b3World* world, const int* shapeIds, int count, int sourceBodyId, int targetBodyId )
{
	if ( count == 0 )
	{
		return;
	}

	if ( world->recording != NULL )
	{
		// A recording replays this as single shape moves
		b3BodyId targetId = b3MakeBodyId( world, targetBodyId );
		for ( int i = 0; i < count; ++i )
		{
			const b3Shape* shape = world->shapes.data + shapeIds[i];
			b3ShapeId shapeId = { shapeIds[i] + 1, world->worldId, shape->generation };
			B3_REC( world, ShapeSetBody, shapeId, targetId );
		}
	}

	world->locked = true;
	b3MoveShapes( world, shapeIds, count, sourceBodyId, targetBodyId );
	world->locked = false;
}

static void b3PushBodyEvent( b3World* world, const b3Destructible* destructible, int bodyId, int parentBodyId, int fragmentCount,
							 bool isDebris )
{
	b3DestructionBodyEvent event = { 0 };
	event.destructibleId = b3MakeDestructibleId( world, destructible );
	event.bodyId = b3MakeBodyId( world, bodyId );
	event.parentBodyId = b3MakeBodyId( world, parentBodyId );
	event.fragmentCount = fragmentCount;
	event.isDebris = isDebris;
	b3Array_Push( world->destruction->bodyEvents, event );
}

// The motion a body gives to fragments that leave it: the velocity of its origin and its spin.
static void b3GetBodyMotion( b3World* world, int bodyId, b3Vec3* originVelocity, b3Vec3* angularVelocity )
{
	*originVelocity = b3Vec3_zero;
	*angularVelocity = b3Vec3_zero;

	b3Body* body = world->bodies.data + bodyId;
	b3BodyState* state = b3GetBodyState( world, body );
	if ( state == NULL || body->type == b3_staticBody )
	{
		return;
	}

	// The state holds the velocity of the center of mass. Shift it to the body origin.
	const b3BodySim* bodySim = b3GetBodySim( world, body );
	b3Vec3 offset = b3SubPos( bodySim->transform.p, bodySim->center );
	*originVelocity = b3Add( state->linearVelocity, b3Cross( state->angularVelocity, offset ) );
	*angularVelocity = state->angularVelocity;
}

// Give debris shapes the debris filter after they moved to their own body.
static void b3ApplyDebrisFilter( b3World* world, b3Destructible* destructible, const b3DestructionPiece* piece )
{
	if ( memcmp( &destructible->debrisFilter, &destructible->shapeDef.filter, sizeof( b3Filter ) ) == 0 )
	{
		return;
	}

	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );
	for ( int i = 0; i < piece->nodeCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + piece->nodes[i];
		for ( int j = 0; j < fragment->hullCount; ++j )
		{
			int rawId = destructible->hullShapes[fragment->firstHull + j];
			if ( rawId != B3_NULL_INDEX )
			{
				const b3Shape* shape = world->shapes.data + rawId;
				b3ShapeId shapeId = { rawId + 1, world->worldId, shape->generation };
				b3Shape_SetFilter( shapeId, destructible->debrisFilter, true );
			}
		}
	}
}

// Stop simulating a piece: its fragments are reported removed and its body is destroyed.
static void b3RemovePiece( b3World* world, b3Destructible* destructible, int pieceIndex )
{
	b3Destruction* destruction = world->destruction;
	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );
	int bodyId = piece->bodyId;

	for ( int i = 0; i < piece->nodeCount; ++i )
	{
		int node = piece->nodes[i];

		for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
		{
			b3BreakBond( world, destructible, destructible->adjacency[j], b3_bondBreakCrush );
		}

		b3PushFragmentEvent( world, destructible, node, b3_fragmentRemoved, bodyId );

		const b3StructureFragment* fragment = fragments + node;
		for ( int j = 0; j < fragment->hullCount; ++j )
		{
			int hullIndex = fragment->firstHull + j;
			int rawId = destructible->hullShapes[hullIndex];
			if ( rawId != B3_NULL_INDEX )
			{
				// The shape goes with the body
				destruction->shapeFragments.data[rawId] = (b3ShapeFragment){ B3_NULL_INDEX, B3_NULL_INDEX, false };
				destructible->hullShapes[hullIndex] = B3_NULL_INDEX;
			}
		}

		destructible->fragmentPiece[node] = B3_FRAGMENT_REMOVED;
	}

	if ( ( piece->flags & b3_pieceDebris ) != 0 && ( piece->flags & b3_pieceFrozen ) == 0 )
	{
		destructible->debrisCount -= 1;
	}

	B3_ASSERT( bodyId != destructible->anchorBodyId );
	b3DestroyDestructionBody( world, destructible, bodyId );
	b3FreePiece( destructible, pieceIndex );
}

// Create a piece for a single released fragment on a new dynamic body. The impulse is a world space
// push through the center of the fragment, on top of the motion it inherits.
static void b3ReleaseFragment( b3World* world, b3Destructible* destructible, int node, int parentBodyId, b3WorldTransform transform,
							   b3Vec3 originVelocity, b3Vec3 angularVelocity, b3Vec3 impulse )
{
	const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + node;

	int pieceIndex = b3AllocPiece( destructible );
	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	if ( piece->nodeCapacity < 1 )
	{
		piece->nodes = b3GrowAlloc( piece->nodes, piece->nodeCapacity * (int)sizeof( int ), 4 * (int)sizeof( int ) );
		piece->nodeCapacity = 4;
	}

	piece->nodes[0] = node;
	piece->nodeCount = 1;
	piece->flags = b3_pieceNew;
	b3ComputePieceMass( destructible, piece );

	bool isDebris = fragment->radius < destructible->debrisRadius;
	if ( isDebris )
	{
		piece->flags |= b3_pieceDebris;
		destructible->debrisCount += 1;
	}

	b3BodyId bodyId = b3CreateDestructionBody( world, destructible, b3_dynamicBody, transform, originVelocity, angularVelocity );
	piece->bodyId = bodyId.index1 - 1;
	destructible->fragmentPiece[node] = pieceIndex;
	destructible->fragmentLocal[node] = 0;

	b3CreateFragmentShapes( world, destructible, node, bodyId, isDebris );
	b3UpdatePieceInert( world, destructible, piece );

	b3MassData massData = { piece->mass, piece->center, piece->inertia };
	b3Body_SetMassData( bodyId, massData );

	if ( b3LengthSquared( impulse ) > 0.0f )
	{
		b3Body_ApplyLinearImpulseToCenter( bodyId, impulse, true );
	}

	world->destruction->newBodyCount += 1;
	b3PushFragmentEvent( world, destructible, node, b3_fragmentReleased, piece->bodyId );
	b3PushBodyEvent( world, destructible, piece->bodyId, parentBodyId, 1, isDebris );
}

// Crush a fragment: break its bonds and replace it with its children.
// Returns false if the step has no bodies left for the children, the crush then waits for the next step.
static bool b3CrushFragment( b3World* world, b3Destructible* destructible, int node, int* bodyBudget )
{
	int pieceIndex = destructible->fragmentPiece[node];
	if ( pieceIndex < 0 )
	{
		return true;
	}

	const b3StructureData* structure = destructible->structure;
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	const b3StructureFragment* fragment = fragments + node;
	const int* children = b3GetStructureChildren( structure ) + fragment->firstChild;

	if ( fragment->childCount == 0 )
	{
		// Nothing to crumble into. The fragment breaks loose in one piece and cannot be crushed again.
		for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
		{
			b3BreakBond( world, destructible, destructible->adjacency[j], b3_bondBreakCrush );
		}

		destructible->health[node] = FLT_MAX;
		b3AddCommitPiece( destructible, pieceIndex );
		return true;
	}

	int bodyCount = 0;
	for ( int i = 0; i < fragment->childCount; ++i )
	{
		if ( fragments[children[i]].radius >= destructible->minFragmentRadius )
		{
			bodyCount += 1;
		}
	}

	if ( bodyCount > *bodyBudget )
	{
		return false;
	}

	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	int bodyId = piece->bodyId;
	b3WorldTransform transform = b3GetBodyTransform( world, bodyId );
	b3Vec3 originVelocity, angularVelocity;
	b3GetBodyMotion( world, bodyId, &originVelocity, &angularVelocity );

	// A push from a blast or the application reached a dynamic body already. A fragment on a body that
	// does not move kept it, and its children carry it off, shared by mass.
	b3Vec3 externalImpulse = b3Vec3_zero;
	if ( world->bodies.data[bodyId].type != b3_dynamicBody )
	{
		externalImpulse = b3RotateVector( transform.q, destructible->loads[node].externalImpulse );
	}

	for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
	{
		b3BreakBond( world, destructible, destructible->adjacency[j], b3_bondBreakCrush );
	}

	// Take the fragment out of its piece
	int local = destructible->fragmentLocal[node];
	piece->nodeCount -= 1;
	if ( local != piece->nodeCount )
	{
		int moved = piece->nodes[piece->nodeCount];
		piece->nodes[local] = moved;
		destructible->fragmentLocal[moved] = local;
	}

	piece->flags |= b3_pieceTopologyDirty | b3_pieceSystemDirty;
	b3AddCommitPiece( destructible, pieceIndex );

	b3PushFragmentEvent( world, destructible, node, b3_fragmentRemoved, bodyId );
	b3DestroyFragmentShapes( world, destructible, node );
	destructible->fragmentPiece[node] = B3_FRAGMENT_REMOVED;

	bool isEmpty = piece->nodeCount == 0;
	bool ownsBody = bodyId != destructible->anchorBodyId;

	// The children appear where the fragment was and move the way it moved. Do not use the piece
	// pointer below, releasing a fragment can move the piece array.
	for ( int i = 0; i < fragment->childCount; ++i )
	{
		int child = children[i];
		if ( fragments[child].radius < destructible->minFragmentRadius )
		{
			// Too small to simulate. The application may show an effect instead.
			destructible->fragmentPiece[child] = B3_FRAGMENT_REMOVED;
			b3PushFragmentEvent( world, destructible, child, b3_fragmentRemoved, bodyId );
			continue;
		}

		b3Vec3 childImpulse = b3MulSV( fragment->mass > 0.0f ? fragments[child].mass / fragment->mass : 0.0f, externalImpulse );
		b3ReleaseFragment( world, destructible, child, bodyId, transform, originVelocity, angularVelocity, childImpulse );
		*bodyBudget -= 1;
	}

	if ( isEmpty )
	{
		// That was the last fragment of the piece
		piece = destructible->pieces.data + pieceIndex;
		if ( ( piece->flags & b3_pieceDebris ) != 0 && ( piece->flags & b3_pieceFrozen ) == 0 )
		{
			destructible->debrisCount -= 1;
		}

		if ( ownsBody )
		{
			b3DestroyDestructionBody( world, destructible, bodyId );
		}

		b3FreePiece( destructible, pieceIndex );
	}

	return true;
}

// A connected part of a piece that is being split
typedef struct b3SplitPart
{
	// Range in the split order array
	int start;
	int count;

	// Intact bonds, each counted once
	int bondCount;
	float mass;
	bool isAnchored;
} b3SplitPart;

// Check if a piece came apart and give each part that broke off its own piece and body.
static void b3SplitPiece( b3World* world, b3Destructible* destructible, int pieceIndex, int* bodyBudget )
{
	b3Destruction* destruction = world->destruction;
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );
	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
	int nodeCount = piece->nodeCount;

	if ( nodeCount == 0 )
	{
		return;
	}

	bool pieceAnchored = ( piece->flags & b3_pieceAnchored ) != 0;
	int parentBodyId = piece->bodyId;

	// Flood along the intact bonds. The order array ends up holding the fragments grouped by part.
	b3Array_Resize( destruction->splitLabels, nodeCount );
	b3Array_Resize( destruction->splitOrder, nodeCount );
	int* labels = destruction->splitLabels.data;
	int* order = destruction->splitOrder.data;
	memset( labels, 0xFF, nodeCount * sizeof( int ) );

	enum
	{
		localPartCapacity = 32
	};

	b3SplitPart localParts[localPartCapacity];
	b3SplitPart* parts = localParts;
	int partCapacity = localPartCapacity;
	int partCount = 0;
	int orderCount = 0;

	for ( int seed = 0; seed < nodeCount; ++seed )
	{
		if ( labels[seed] != B3_NULL_INDEX )
		{
			continue;
		}

		if ( partCount == partCapacity )
		{
			int newCapacity = 2 * partCapacity;
			b3SplitPart* newParts = b3Alloc( newCapacity * sizeof( b3SplitPart ) );
			memcpy( newParts, parts, partCount * sizeof( b3SplitPart ) );
			if ( parts != localParts )
			{
				b3Free( parts, partCapacity * sizeof( b3SplitPart ) );
			}
			parts = newParts;
			partCapacity = newCapacity;
		}

		b3SplitPart* part = parts + partCount;
		*part = (b3SplitPart){ 0 };
		part->start = orderCount;

		labels[seed] = partCount;
		order[orderCount++] = piece->nodes[seed];

		for ( int head = part->start; head < orderCount; ++head )
		{
			int node = order[head];
			part->mass += fragments[node].mass;

			for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
			{
				const b3DestructionBond* bond = destructible->bonds + destructible->adjacency[j];
				if ( bond->flags & b3_bondBroken )
				{
					continue;
				}

				if ( bond->nodeA == node )
				{
					part->bondCount += 1;
				}

				if ( bond->nodeB == B3_NULL_INDEX )
				{
					part->isAnchored = true;
					continue;
				}

				int other = bond->nodeA == node ? bond->nodeB : bond->nodeA;
				int otherLocal = destructible->fragmentLocal[other];
				B3_ASSERT( destructible->fragmentPiece[other] == pieceIndex );
				if ( labels[otherLocal] == B3_NULL_INDEX )
				{
					labels[otherLocal] = partCount;
					order[orderCount++] = other;
				}
			}
		}

		part->count = orderCount - part->start;
		partCount += 1;
	}

	B3_ASSERT( orderCount == nodeCount );

	// Pick the part that keeps this piece and its body. On the anchor body that is a part that is still
	// anchored, otherwise the heaviest part so the fewest shapes move.
	int keeper = B3_NULL_INDEX;
	if ( pieceAnchored )
	{
		for ( int i = 0; i < partCount; ++i )
		{
			if ( parts[i].isAnchored )
			{
				keeper = i;
				break;
			}
		}
	}
	else
	{
		keeper = 0;
		for ( int i = 1; i < partCount; ++i )
		{
			if ( parts[i].mass > parts[keeper].mass )
			{
				keeper = i;
			}
		}
	}

	if ( partCount == 1 && keeper == 0 )
	{
		// Still in one part. Fragments may have been crushed out of it, so refresh the mass.
		piece->flags &= ~b3_pieceTopologyDirty;
		piece->bondCount = parts[0].bondCount;

		if ( pieceAnchored == false )
		{
			b3UpdatePieceInert( world, destructible, piece );
		}

		if ( pieceAnchored == false && ( piece->flags & b3_pieceFrozen ) == 0 )
		{
			b3ComputePieceMass( destructible, piece );
			b3MassData massData = { piece->mass, piece->center, piece->inertia };
			b3Body_SetMassData( b3MakeBodyId( world, parentBodyId ), massData );
		}

		if ( parts != localParts )
		{
			b3Free( parts, partCapacity * sizeof( b3SplitPart ) );
		}
		return;
	}

	b3WorldTransform transform = b3GetBodyTransform( world, parentBodyId );
	b3Vec3 originVelocity, angularVelocity;
	b3GetBodyMotion( world, parentBodyId, &originVelocity, &angularVelocity );
	bool parentIsStatic = world->bodies.data[parentBodyId].type != b3_dynamicBody;

	// Reset the scratch used to measure how overloaded the broken bonds were
	piece->breakForce = 0.0f;
	piece->breakExcess = 0.0f;

	bool isPending = false;
	int firstNewPiece = destructible->pieces.count;
	B3_UNUSED( firstNewPiece );

	// Mark which parts leave. A part that needs a body can only leave while the step has bodies left.
	// Negative labels are reused here: part index to new piece index.
	b3Array_Resize( destruction->splitShapes, 0 );
	int* partPieces = b3Alloc( partCount * sizeof( int ) );

	for ( int i = 0; i < partCount; ++i )
	{
		partPieces[i] = B3_NULL_INDEX;
		if ( i == keeper )
		{
			partPieces[i] = pieceIndex;
			continue;
		}

		const b3SplitPart* part = parts + i;
		bool staysOnAnchor = pieceAnchored && part->isAnchored;

		if ( staysOnAnchor == false && *bodyBudget <= 0 )
		{
			// Out of bodies for this step. The part stays attached and the split finishes later.
			isPending = true;
			continue;
		}

		int newIndex = b3AllocPiece( destructible );

		// The piece array may have moved
		piece = destructible->pieces.data + pieceIndex;
		b3DestructionPiece* newPiece = destructible->pieces.data + newIndex;
		partPieces[i] = newIndex;

		if ( newPiece->nodeCapacity < part->count )
		{
			newPiece->nodes =
				b3GrowAlloc( newPiece->nodes, newPiece->nodeCapacity * (int)sizeof( int ), part->count * (int)sizeof( int ) );
			newPiece->nodeCapacity = part->count;
		}

		memcpy( newPiece->nodes, order + part->start, part->count * sizeof( int ) );
		newPiece->nodeCount = part->count;
		newPiece->bondCount = part->bondCount;
		newPiece->flags = b3_pieceSystemDirty | b3_pieceLoadDirty | b3_pieceNew;

		for ( int j = 0; j < part->count; ++j )
		{
			int node = newPiece->nodes[j];
			destructible->fragmentPiece[node] = newIndex;
			destructible->fragmentLocal[node] = j;
		}

		b3ComputePieceMass( destructible, newPiece );

		if ( staysOnAnchor )
		{
			// Another anchored part. It stays on the anchor body, so no shape has to move.
			newPiece->bodyId = parentBodyId;
			newPiece->flags |= b3_pieceAnchored;
			continue;
		}

		// A free part: new dynamic body in the same frame, moving the way the parent moved
		b3BodyId newBodyId =
			b3CreateDestructionBody( world, destructible, b3_dynamicBody, transform, originVelocity, angularVelocity );
		*bodyBudget -= 1;
		destruction->newBodyCount += 1;
		newPiece->bodyId = newBodyId.index1 - 1;

		// A part that comes free starts its solve from rest. The displacement it had while it was
		// held says nothing about how it is loaded now.
		for ( int j = 0; j < newPiece->nodeCount; ++j )
		{
			destructible->displacements[newPiece->nodes[j]] = (b3DVec6){ 0 };
		}

		if ( parentIsStatic )
		{
			newPiece->flags |= b3_pieceWasStatic;
		}
		else
		{
			newPiece->parentPiece = pieceIndex;
		}

		b3Array_Clear( destruction->splitShapes );
		for ( int j = 0; j < part->count; ++j )
		{
			const b3StructureFragment* fragment = fragments + newPiece->nodes[j];
			for ( int k = 0; k < fragment->hullCount; ++k )
			{
				int rawId = destructible->hullShapes[fragment->firstHull + k];
				if ( rawId != B3_NULL_INDEX )
				{
					b3Array_Push( destruction->splitShapes, rawId );
				}
			}
		}

		b3MoveFragmentShapes( world, destruction->splitShapes.data, destruction->splitShapes.count, parentBodyId,
							  newPiece->bodyId );

		b3MassData massData = { newPiece->mass, newPiece->center, newPiece->inertia };
		b3Body_SetMassData( newBodyId, massData );

		if ( parentIsStatic )
		{
			// A push from a blast or the application never reached these fragments while they were on a
			// body that does not move. They take it along now.
			b3Vec3 linearImpulse = b3Vec3_zero;
			b3Vec3 angularImpulse = b3Vec3_zero;
			for ( int j = 0; j < part->count; ++j )
			{
				int node = newPiece->nodes[j];
				const b3NodeLoad* load = destructible->loads + node;
				b3Vec3 arm = b3Sub( fragments[node].center, newPiece->center );
				linearImpulse = b3Add( linearImpulse, load->externalImpulse );
				angularImpulse = b3Add( angularImpulse, b3Add( load->externalAngularImpulse, b3Cross( arm, load->externalImpulse ) ) );
			}

			if ( b3LengthSquared( linearImpulse ) > 0.0f )
			{
				b3Body_ApplyLinearImpulseToCenter( newBodyId, b3RotateVector( transform.q, linearImpulse ), true );
			}

			if ( b3LengthSquared( angularImpulse ) > 0.0f )
			{
				b3Body_ApplyAngularImpulse( newBodyId, b3RotateVector( transform.q, angularImpulse ), true );
			}
		}

		b3UpdatePieceInert( world, destructible, newPiece );

		bool isDebris = newPiece->radius < destructible->debrisRadius;
		if ( isDebris )
		{
			newPiece->flags |= b3_pieceDebris;
			destructible->debrisCount += 1;
			b3ApplyDebrisFilter( world, destructible, newPiece );
		}

		b3PushBodyEvent( world, destructible, newPiece->bodyId, parentBodyId, part->count, isDebris );
		for ( int j = 0; j < part->count; ++j )
		{
			b3PushFragmentEvent( world, destructible, newPiece->nodes[j], b3_fragmentMoved, newPiece->bodyId );
		}
	}

	// What remains on this piece: the keeper part plus any part that could not leave yet
	piece = destructible->pieces.data + pieceIndex;
	{
		int remaining = 0;
		int remainingBonds = 0;
		for ( int i = 0; i < partCount; ++i )
		{
			if ( partPieces[i] != pieceIndex && partPieces[i] != B3_NULL_INDEX )
			{
				continue;
			}

			const b3SplitPart* part = parts + i;
			remainingBonds += part->bondCount;
			for ( int j = 0; j < part->count; ++j )
			{
				int node = order[part->start + j];
				piece->nodes[remaining] = node;
				destructible->fragmentLocal[node] = remaining;
				remaining += 1;
			}
		}

		piece->nodeCount = remaining;
		piece->bondCount = remainingBonds;

		if ( pieceAnchored == false )
		{
			b3UpdatePieceInert( world, destructible, piece );
		}
	}

	// Measure how overloaded the bonds were that broke between the parts. A bond that broke in an
	// impact carried more force than it could hold, and that surplus never reached the far side.
	for ( int i = 0; i < piece->breaks.count; ++i )
	{
		const b3BondBreak* bondBreak = piece->breaks.data + i;
		const b3DestructionBond* bond = destructible->bonds + bondBreak->bondIndex;
		int pieceA = destructible->fragmentPiece[bond->nodeA];
		int pieceB = bond->nodeB != B3_NULL_INDEX ? destructible->fragmentPiece[bond->nodeB] : B3_NULL_INDEX;
		if ( pieceA == pieceB )
		{
			continue;
		}

		if ( pieceA >= 0 )
		{
			destructible->pieces.data[pieceA].breakForce += bondBreak->force;
			destructible->pieces.data[pieceA].breakExcess += bondBreak->force * bondBreak->excess;
		}

		if ( pieceB >= 0 )
		{
			destructible->pieces.data[pieceB].breakForce += bondBreak->force;
			destructible->pieces.data[pieceB].breakExcess += bondBreak->force * bondBreak->excess;
		}
	}

	for ( int i = 0; i < partCount; ++i )
	{
		int newIndex = partPieces[i];
		if ( newIndex == B3_NULL_INDEX || newIndex == pieceIndex )
		{
			continue;
		}

		b3DestructionPiece* newPiece = destructible->pieces.data + newIndex;
		newPiece->impactExcess = newPiece->breakForce > 0.0f ? newPiece->breakExcess / newPiece->breakForce : 0.0f;
		newPiece->breakForce = 0.0f;
		newPiece->breakExcess = 0.0f;
	}

	b3Free( partPieces, partCount * sizeof( int ) );

	if ( piece->nodeCount == 0 )
	{
		// Every part left. This only happens on the anchor body, which stays.
		B3_ASSERT( pieceAnchored );
		b3FreePiece( destructible, pieceIndex );
	}
	else
	{
		piece->flags |= b3_pieceSystemDirty | b3_pieceLoadDirty;

		if ( isPending )
		{
			piece->flags |= b3_pieceTopologyDirty;
		}
		else
		{
			piece->flags &= ~b3_pieceTopologyDirty;
		}

		if ( pieceAnchored == false )
		{
			b3ComputePieceMass( destructible, piece );
			b3MassData massData = { piece->mass, piece->center, piece->inertia };
			b3Body_SetMassData( b3MakeBodyId( world, parentBodyId ), massData );
		}
	}

	if ( parts != localParts )
	{
		b3Free( parts, partCapacity * sizeof( b3SplitPart ) );
	}
}

static void b3CommitDestructible( b3World* world, b3Destructible* destructible, int* bodyBudget )
{
	// Crush fragments first: this breaks bonds and can leave pieces in parts
	if ( destructible->crushedNodes.count > 0 )
	{
		int keepCount = 0;
		for ( int i = 0; i < destructible->crushedNodes.count; ++i )
		{
			int node = destructible->crushedNodes.data[i];
			if ( b3CrushFragment( world, destructible, node, bodyBudget ) )
			{
				destructible->nodeFlags[node] &= ~b3_nodeCrushed;
			}
			else
			{
				// No bodies left in this step, try again in the next
				destructible->crushedNodes.data[keepCount++] = node;
			}
		}

		destructible->crushedNodes.count = keepCount;
	}

	// Break the bonds the analysis and the application asked for
	for ( int i = 0; i < destructible->commitPieces.count; ++i )
	{
		int pieceIndex = destructible->commitPieces.data[i];
		b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
		if ( piece->bodyId == B3_NULL_INDEX )
		{
			continue;
		}

		for ( int j = 0; j < piece->breaks.count; ++j )
		{
			b3BreakBond( world, destructible, piece->breaks.data[j].bondIndex, (b3BondBreakCause)piece->breaks.data[j].cause );
		}
	}

	// Split the pieces that came apart
	int keepCount = 0;
	int pendingCount = 0;
	int commitCount = destructible->commitPieces.count;
	for ( int i = 0; i < commitCount; ++i )
	{
		int pieceIndex = destructible->commitPieces.data[i];
		b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
		if ( piece->bodyId == B3_NULL_INDEX )
		{
			// Freed while crushing. A freed piece has no flags left.
			continue;
		}

		if ( piece->flags & b3_pieceTopologyDirty )
		{
			b3SplitPiece( world, destructible, pieceIndex, bodyBudget );
		}

		// Splitting can move the piece array and can free this piece
		piece = destructible->pieces.data + pieceIndex;
		if ( piece->bodyId == B3_NULL_INDEX )
		{
			continue;
		}

		b3Array_Clear( piece->breaks );

		if ( piece->flags & b3_pieceTopologyDirty )
		{
			// Ran out of bodies. Keep the piece in the list so the split continues in the next step.
			destructible->commitPieces.data[keepCount++] = pieceIndex;
			pendingCount += 1;
		}
		else
		{
			piece->flags &= ~b3_pieceInCommitList;
		}
	}

	destructible->commitPieces.count = keepCount;
	destructible->pendingSplitCount = pendingCount;

	// Shapes that leave a body or are destroyed leave its mass to be updated, and the pieces do
	// that for their own bodies above. The anchor body has no mass that matters. It is kinematic
	// when it follows a host though, and then the engine wants to hear that its mass was dealt with.
	if ( destructible->anchorBodyId != B3_NULL_INDEX )
	{
		const b3Body* anchor = world->bodies.data + destructible->anchorBodyId;
		if ( anchor->flags & b3_dirtyMass )
		{
			b3MassData massData = { 0 };
			b3Body_SetMassData( b3MakeBodyId( world, destructible->anchorBodyId ), massData );
		}
	}
}

// Age the debris of a destructible: remove it when its time is up, freeze it when it sleeps.
static void b3UpdateDebris( b3World* world, b3Destructible* destructible )
{
	float timeStep = world->destruction->timeStep;

	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		b3DestructionPiece* piece = destructible->pieces.data + i;
		if ( piece->bodyId == B3_NULL_INDEX || ( piece->flags & b3_pieceDebris ) == 0 || ( piece->flags & b3_pieceFrozen ) != 0 )
		{
			continue;
		}

		piece->age += timeStep;

		if ( destructible->debrisLifetime > 0.0f && piece->age >= destructible->debrisLifetime )
		{
			b3RemovePiece( world, destructible, i );
			continue;
		}

		if ( destructible->freezeSleepingDebris )
		{
			const b3Body* body = world->bodies.data + piece->bodyId;
			if ( body->setIndex >= b3_firstSleepingSet )
			{
				// Static rubble never wakes, which caps the number of active bodies after a collapse
				b3Body_SetType( b3MakeBodyId( world, piece->bodyId ), b3_staticBody );

				piece = destructible->pieces.data + i;
				piece->flags |= b3_pieceFrozen;
				piece->flags &= ~( b3_pieceCreeping | b3_pieceSleepDisabled );
				b3DestroyStressSystem( piece->system );
				piece->system = NULL;
				destructible->debrisCount -= 1;
			}
		}
	}
}

// Give momentum back to bodies whose impact broke the bonds that resisted it.
//
// The constraint solver treated the structure as one rigid body, so it took all of the impactor's
// momentum. But the bonds broke: they could only pass on a part of that impulse before failing. The
// rest is returned to the impactor, so a falling slab keeps going through a floor it breaks instead
// of stopping on it. On the next step it meets the freed fragments as ordinary bodies.
static void b3TransferMomentum( b3World* world )
{
	b3Destruction* destruction = world->destruction;
	float scale = destruction->settings.momentumTransferScale;
	if ( scale <= 0.0f )
	{
		return;
	}

	for ( int i = 0; i < destruction->contactLoads.count; ++i )
	{
		const b3ContactLoad* contactLoad = destruction->contactLoads.data + i;
		if ( contactLoad->destructibleId == B3_NULL_INDEX )
		{
			continue;
		}

		b3Destructible* destructible = destruction->destructibles.data + contactLoad->destructibleId;
		if ( b3IsLiveDestructible( destructible ) == false )
		{
			continue;
		}

		int node = contactLoad->fragmentIndex;
		if ( ( destructible->nodeFlags[node] & b3_nodeImpact ) == 0 )
		{
			continue;
		}

		int pieceIndex = destructible->fragmentPiece[node];
		if ( pieceIndex < 0 )
		{
			continue;
		}

		const b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
		if ( ( piece->flags & b3_pieceNew ) == 0 || ( piece->flags & b3_pieceAnchored ) != 0 || piece->impactExcess <= 0.0f )
		{
			continue;
		}

		// The other body may have been destroyed during this commit and its slot reused
		const b3Body* other = world->bodies.data + contactLoad->otherBodyId;
		if ( other->id != contactLoad->otherBodyId || other->generation != contactLoad->otherGeneration ||
			 other->type != b3_dynamicBody || other->setIndex != b3_awakeSet )
		{
			continue;
		}

		// The load holds the impulse on the fragment. The impactor received the opposite, so this
		// pushes it back along its way.
		float length = b3Length( contactLoad->impulse );
		if ( length <= 0.0f )
		{
			continue;
		}

		b3Vec3 direction = b3MulSV( 1.0f / length, contactLoad->impulse );

		// A dynamic parent received the impulse of the hit as well. What the impactor gets back comes
		// out of the part that kept the body.
		const b3Body* parentBody = NULL;
		if ( piece->parentPiece != B3_NULL_INDEX )
		{
			const b3DestructionPiece* parent = destructible->pieces.data + piece->parentPiece;
			if ( parent->bodyId != B3_NULL_INDEX && ( parent->flags & ( b3_pieceAnchored | b3_pieceFrozen ) ) == 0 )
			{
				parentBody = world->bodies.data + parent->bodyId;
			}
		}

		// The exchange only undoes part of the hit. Before the hit the impactor closed in on the
		// fragment at the approach speed, and afterwards the two must not close in faster than that.
		// The mass that limits this is the smaller of the two sides. Without the limit a small body
		// caught between a heavy falling piece and the ground would be shot away with the momentum
		// of the heavy piece, and so would a small parent.
		const b3BodyState* otherState = b3GetBodyState( world, (b3Body*)other );
		const b3BodyState* pieceState = b3GetBodyState( world, world->bodies.data + piece->bodyId );
		float otherSpeed = otherState != NULL ? b3Dot( otherState->linearVelocity, direction ) : 0.0f;
		float pieceSpeed = pieceState != NULL ? b3Dot( pieceState->linearVelocity, direction ) : 0.0f;
		float closingSpeed = contactLoad->approachSpeed + pieceSpeed - otherSpeed;
		if ( closingSpeed <= 0.0f || other->mass <= 0.0f )
		{
			continue;
		}

		float reducedMass = other->mass;
		if ( parentBody != NULL && parentBody->mass > 0.0f )
		{
			reducedMass = other->mass * parentBody->mass / ( other->mass + parentBody->mass );
		}

		float wanted = scale * b3MinFloat( piece->impactExcess, 1.0f ) * length;
		b3Vec3 impulse = b3MulSV( b3MinFloat( wanted, reducedMass * closingSpeed ), direction );
		b3Body_ApplyLinearImpulse( b3MakeBodyId( world, contactLoad->otherBodyId ), impulse, contactLoad->point, true );

		if ( parentBody != NULL )
		{
			b3Body_ApplyLinearImpulseToCenter( b3MakeBodyId( world, parentBody->id ), b3Neg( impulse ), true );
		}
	}
}

// Forget the pushes of this step. A fragment that is still waiting to be crushed or split off, because
// the step ran out of new bodies, keeps its push until that happens.
static void b3ClearExternalImpulses( b3Destructible* destructible )
{
	for ( int i = 0; i < destructible->loadedNodes.count; ++i )
	{
		int node = destructible->loadedNodes.data[i];
		if ( destructible->nodeFlags[node] & b3_nodeCrushed )
		{
			continue;
		}

		int pieceIndex = destructible->fragmentPiece[node];
		if ( pieceIndex >= 0 && ( destructible->pieces.data[pieceIndex].flags & b3_pieceTopologyDirty ) != 0 )
		{
			continue;
		}

		b3NodeLoad* load = destructible->loads + node;
		load->externalImpulse = b3Vec3_zero;
		load->externalAngularImpulse = b3Vec3_zero;
	}
}

// Keep a body awake while its piece is still failing, so a slow collapse does not stall.
static void b3UpdateSleepLocks( b3World* world )
{
	b3Destruction* destruction = world->destruction;

	for ( int i = 0; i < destruction->tasks.count; ++i )
	{
		const b3PieceTask* task = destruction->tasks.data + i;
		b3Destructible* destructible = destruction->destructibles.data + task->destructibleId;
		if ( b3IsLiveDestructible( destructible ) == false || task->pieceIndex >= destructible->pieces.count )
		{
			continue;
		}

		b3DestructionPiece* piece = destructible->pieces.data + task->pieceIndex;
		if ( piece->bodyId == B3_NULL_INDEX || ( piece->flags & ( b3_pieceAnchored | b3_pieceFrozen ) ) != 0 )
		{
			continue;
		}

		bool isCreeping = ( piece->flags & b3_pieceCreeping ) != 0;
		bool isLocked = ( piece->flags & b3_pieceSleepDisabled ) != 0;

		if ( isCreeping && isLocked == false )
		{
			b3Body_EnableSleep( b3MakeBodyId( world, piece->bodyId ), false );
			piece->flags |= b3_pieceSleepDisabled;
		}
		else if ( isCreeping == false && isLocked )
		{
			b3Body_EnableSleep( b3MakeBodyId( world, piece->bodyId ), destructible->enableSleep );
			piece->flags &= ~b3_pieceSleepDisabled;
		}
	}
}

void b3DestructionCommit( b3World* world )
{
	b3Destruction* destruction = world->destruction;
	B3_ASSERT( world->locked == false );

	uint64_t ticks = b3GetTicks();
	int bodyBudget = destruction->settings.maxNewBodiesPerStep;
	destruction->pendingSplitCount = 0;

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		b3Destructible* destructible = destruction->destructibles.data + i;
		if ( b3IsLiveDestructible( destructible ) == false )
		{
			continue;
		}

		if ( destructible->crushedNodes.count > 0 || destructible->commitPieces.count > 0 )
		{
			b3CommitDestructible( world, destructible, &bodyBudget );
		}

		destruction->pendingSplitCount += destructible->pendingSplitCount;
	}

	b3TransferMomentum( world );
	b3UpdateSleepLocks( world );

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		b3Destructible* destructible = destruction->destructibles.data + i;
		if ( b3IsLiveDestructible( destructible ) == false )
		{
			continue;
		}

		b3ClearExternalImpulses( destructible );

		if ( destructible->debrisCount > 0 )
		{
			b3UpdateDebris( world, destructible );
		}
	}

	destruction->profile.fracture = b3GetMilliseconds( ticks );
	destruction->profile.total = destruction->profile.loads + destruction->profile.solve + destruction->profile.fracture;
}

// A fragment found by a radial damage query
typedef struct b3DamageHit
{
	int destructibleId;
	int fragmentIndex;
	float distance;
} b3DamageHit;

b3DeclareArray( b3DamageHit );

typedef struct b3DamageQueryContext
{
	b3World* world;
	b3Pos position;
	float outerRadius;
	b3Array( b3DamageHit ) hits;
} b3DamageQueryContext;

static bool b3DamageQueryCallback( int proxyId, uint64_t userData, void* context )
{
	B3_UNUSED( proxyId );
	b3DamageQueryContext* queryContext = context;
	b3World* world = queryContext->world;

	int shapeId = (int)userData;
	const b3Shape* shape = world->shapes.data + shapeId;
	if ( ( shape->flags & b3_destructibleShape ) == 0 )
	{
		return true;
	}

	b3ShapeFragment link = world->destruction->shapeFragments.data[shapeId];
	if ( link.destructibleId == B3_NULL_INDEX )
	{
		return true;
	}

	// Distance from the damage center to the shape, measured in the body frame for precision
	b3WorldTransform transform = b3GetBodyTransform( world, shape->bodyId );
	b3Vec3 localPosition = b3InvTransformWorldPoint( transform, queryContext->position );

	b3DistanceInput input;
	input.proxyA = b3MakeShapeProxy( shape );
	input.proxyB = (b3ShapeProxy){ &localPosition, 1, 0.0f };
	input.transform = b3Transform_identity;
	input.useRadii = true;

	b3SimplexCache cache = { 0 };
	b3DistanceOutput output = b3ShapeDistance( &input, &cache, NULL, 0 );
	if ( output.distance > queryContext->outerRadius )
	{
		return true;
	}

	b3DamageHit hit = { link.destructibleId, link.fragmentIndex, output.distance };
	b3Array_Push( queryContext->hits, hit );
	return true;
}

static inline float b3DamageFalloff( const b3RadialDamageDef* def, float distance )
{
	if ( distance <= def->innerRadius )
	{
		return 1.0f;
	}

	if ( distance >= def->outerRadius || def->outerRadius <= def->innerRadius )
	{
		return 0.0f;
	}

	return ( def->outerRadius - distance ) / ( def->outerRadius - def->innerRadius );
}

// Queue the damage of one destructible. The distances array holds the distance of every fragment
// that takes damage and a negative value for the rest.
static void b3QueueRadialDamage( b3Destructible* destructible, const b3RadialDamageDef* def, const float* distances )
{
	const b3StructureMaterial* materials = b3GetStructureMaterials( destructible->structure );

	for ( int node = 0; node < destructible->fragmentCount; ++node )
	{
		float distance = distances[node];
		if ( distance < 0.0f )
		{
			continue;
		}

		float damage = def->damage * b3DamageFalloff( def, distance );
		if ( damage <= 0.0f )
		{
			continue;
		}

		b3DestructionCommand fragmentCommand = { 0 };
		fragmentCommand.type = b3_commandDamageFragment;
		fragmentCommand.index = node;
		fragmentCommand.value = damage;
		b3Array_Push( destructible->commands, fragmentCommand );

		// Bonds take damage as well: breaking a bond costs its area times the toughness. A bond between
		// two damaged fragments takes the larger of the two. Visit it from the closer fragment.
		for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
		{
			int bondIndex = destructible->adjacency[j];
			const b3DestructionBond* bond = destructible->bonds + bondIndex;
			if ( bond->flags & ( b3_bondBroken | b3_bondRigid ) )
			{
				continue;
			}

			if ( bond->nodeB != B3_NULL_INDEX )
			{
				int other = bond->nodeA == node ? bond->nodeB : bond->nodeA;
				float otherDistance = distances[other];
				bool otherIsCloser = otherDistance >= 0.0f && ( otherDistance < distance || ( otherDistance == distance && other < node ) );
				if ( otherIsCloser )
				{
					continue;
				}
			}

			float energy = destructible->strengthScale * bond->strength * materials[bond->materialIndex].toughness * bond->area0;
			if ( energy <= 0.0f )
			{
				continue;
			}

			b3DestructionCommand bondCommand = { 0 };
			bondCommand.type = b3_commandDamageBond;
			bondCommand.index = bondIndex;
			bondCommand.value = damage / energy;
			b3Array_Push( destructible->commands, bondCommand );
		}
	}
}

void b3World_ApplyRadialDamage( b3WorldId worldId, const b3RadialDamageDef* def )
{
	B3_ASSERT( b3IsValidPosition( def->position ) );
	B3_ASSERT( b3IsValidFloat( def->innerRadius ) && def->innerRadius >= 0.0f );
	B3_ASSERT( b3IsValidFloat( def->outerRadius ) && def->outerRadius >= 0.0f );
	B3_ASSERT( b3IsValidFloat( def->damage ) && def->damage >= 0.0f );

	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL || world->destruction == NULL )
	{
		return;
	}

	b3Destruction* destruction = world->destruction;
	float outerRadius = b3MaxFloat( def->outerRadius, def->innerRadius );

	b3DamageQueryContext queryContext = { 0 };
	queryContext.world = world;
	queryContext.position = def->position;
	queryContext.outerRadius = outerRadius;

	b3AABB localBox = { { -outerRadius, -outerRadius, -outerRadius }, { outerRadius, outerRadius, outerRadius } };
	b3AABB aabb = b3OffsetAABB( localBox, def->position );

	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		b3DynamicTree_Query( world->broadPhase.trees + i, aabb, def->maskBits, false, b3DamageQueryCallback, &queryContext );
	}

	// Handle one destructible at a time
	for ( int destructibleId = 0; destructibleId < destruction->destructibles.count; ++destructibleId )
	{
		b3Destructible* destructible = destruction->destructibles.data + destructibleId;
		if ( b3IsLiveDestructible( destructible ) == false )
		{
			continue;
		}

		float* distances = NULL;
		int fragmentCount = destructible->fragmentCount;
		float nearest = FLT_MAX;

		for ( int i = 0; i < queryContext.hits.count; ++i )
		{
			const b3DamageHit* hit = queryContext.hits.data + i;
			if ( hit->destructibleId != destructibleId || destructible->fragmentPiece[hit->fragmentIndex] < 0 )
			{
				continue;
			}

			if ( distances == NULL )
			{
				distances = b3Alloc( fragmentCount * sizeof( float ) );
				for ( int j = 0; j < fragmentCount; ++j )
				{
					distances[j] = -1.0f;
				}
			}

			// A fragment with several hulls is as close as its closest hull
			float current = distances[hit->fragmentIndex];
			if ( current < 0.0f || hit->distance < current )
			{
				distances[hit->fragmentIndex] = hit->distance;
			}

			nearest = b3MinFloat( nearest, hit->distance );
		}

		if ( distances == NULL )
		{
			continue;
		}

		if ( def->spreadAlongBonds )
		{
			// Measure the falloff along the bonds, starting from the fragments inside the inner radius
			// (or from the nearest fragment if none is). Damage then cannot jump across a gap to
			// something that is close in space but far along the structure.
			float seedRadius = b3MaxFloat( def->innerRadius, nearest );
			int* queue = b3Alloc( fragmentCount * sizeof( int ) );
			float* pathDistances = b3Alloc( fragmentCount * sizeof( float ) );
			int queueCount = 0;

			for ( int j = 0; j < fragmentCount; ++j )
			{
				pathDistances[j] = -1.0f;
				if ( distances[j] >= 0.0f && distances[j] <= seedRadius )
				{
					pathDistances[j] = distances[j];
					queue[queueCount++] = j;
				}
			}

			// Relax until no path gets shorter. The work list is a ring that holds each fragment at most
			// once at a time, so it never needs more room than the fragment count.
			for ( int j = 0; j < queueCount; ++j )
			{
				destructible->nodeFlags[queue[j]] |= b3_nodeVisited;
			}

			int head = 0;
			int pending = queueCount;
			int tail = queueCount % fragmentCount;
			while ( pending > 0 )
			{
				int node = queue[head];
				head = head + 1 == fragmentCount ? 0 : head + 1;
				pending -= 1;
				destructible->nodeFlags[node] &= ~b3_nodeVisited;
				float base = pathDistances[node];

				for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
				{
					const b3DestructionBond* bond = destructible->bonds + destructible->adjacency[j];
					if ( ( bond->flags & b3_bondBroken ) != 0 || bond->nodeB == B3_NULL_INDEX )
					{
						continue;
					}

					int other = bond->nodeA == node ? bond->nodeB : bond->nodeA;
					float distance = base + 1.0f / bond->invLength;
					if ( distance >= outerRadius )
					{
						continue;
					}

					if ( pathDistances[other] < 0.0f || distance < pathDistances[other] )
					{
						pathDistances[other] = distance;

						if ( ( destructible->nodeFlags[other] & b3_nodeVisited ) == 0 )
						{
							destructible->nodeFlags[other] |= b3_nodeVisited;
							queue[tail] = other;
							tail = tail + 1 == fragmentCount ? 0 : tail + 1;
							pending += 1;
						}
					}
				}
			}

			for ( int j = 0; j < fragmentCount; ++j )
			{
				destructible->nodeFlags[j] &= ~b3_nodeVisited;
			}

			b3QueueRadialDamage( destructible, def, pathDistances );
			b3Free( pathDistances, fragmentCount * sizeof( float ) );
			b3Free( queue, fragmentCount * sizeof( int ) );
		}
		else
		{
			b3QueueRadialDamage( destructible, def, distances );
		}

		b3Free( distances, fragmentCount * sizeof( float ) );
	}

	b3Array_Destroy( queryContext.hits );
}
