// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

// Destructible lifetime, queries, and the hooks the rest of the engine calls.
// The per step work lives in destruction_step.c and the stress solve in destruction_stress.c.

#include "destruction.h"

#include "body.h"
#include "core.h"
#include "physics_world.h"
#include "shape.h"

#include "box3d/box3d.h"
#include "box3d/destruction.h"

#include <float.h>
#include <string.h>

b3DestructibleDef b3DefaultDestructibleDef( void )
{
	b3DestructibleDef def = { 0 };
	def.position = b3Pos_zero;
	def.rotation = b3Quat_identity;
	def.hostBodyId = b3_nullBodyId;
	def.shapeDef = b3DefaultShapeDef();
	def.anchorToStatic = true;
	def.anchorTolerance = 2.0f * B3_LINEAR_SLOP;
	def.strengthScale = 1.0f;
	def.seed = 1;
	def.debrisRadius = 0.25f * b3GetLengthUnitsPerMeter();
	def.debrisFilter = b3DefaultFilter();
	def.debrisLifetime = 0.0f;
	def.minFragmentRadius = 0.0f;
	def.freezeSleepingDebris = false;
	def.enableSleep = true;
	def.enableStress = true;
	def.enableBondEvents = false;
	def.internalValue = B3_SECRET_COOKIE;
	return def;
}

b3DestructionSettings b3DefaultDestructionSettings( void )
{
	b3DestructionSettings settings = { 0 };
	settings.enableStress = true;
	settings.maxIterations = 32;
	settings.workBudget = 100000;
	settings.solverTolerance = 0.02f;
	settings.damageTolerance = 0.1f;
	settings.loadSmoothingTime = 0.1f;
	settings.impactVelocityThreshold = 1.0f * b3GetLengthUnitsPerMeter();
	settings.impactDuration = 0.0f;
	settings.impactRadius = 3.0f * b3GetLengthUnitsPerMeter();
	settings.impactDamageScale = 1.0f;
	settings.impactFractureScale = 2.0f;
	settings.momentumTransferScale = 1.0f;
	settings.maxNewBodiesPerStep = 256;
	return settings;
}

b3RadialDamageDef b3DefaultRadialDamageDef( void )
{
	b3RadialDamageDef def = { 0 };
	def.maskBits = B3_DEFAULT_MASK_BITS;
	def.position = b3Pos_zero;
	def.spreadAlongBonds = true;
	return def;
}

static b3Destruction* b3EnsureDestruction( b3World* world )
{
	if ( world->destruction != NULL )
	{
		return world->destruction;
	}

	b3Destruction* destruction = b3AllocZero( sizeof( b3Destruction ) );
	destruction->settings = b3DefaultDestructionSettings();
	destruction->destructibleIdPool = b3CreateIdPool();
	world->destruction = destruction;
	return destruction;
}

b3Destructible* b3GetDestructibleFromId( b3World* world, b3DestructibleId id )
{
	B3_ASSERT( b3Destructible_IsValid( id ) );
	return b3Array_Get( world->destruction->destructibles, id.index1 - 1 );
}

b3DestructibleId b3MakeDestructibleId( const b3World* world, const b3Destructible* destructible )
{
	return (b3DestructibleId){ destructible->id + 1, world->worldId, destructible->generation };
}

static inline b3WorldId b3MakeWorldId( const b3World* world )
{
	return (b3WorldId){ (uint16_t)( world->worldId + 1 ), world->generation };
}

int b3AllocPiece( b3Destructible* destructible )
{
	int pieceIndex;
	if ( destructible->freePieces.count > 0 )
	{
		pieceIndex = b3Array_Pop( destructible->freePieces );
	}
	else
	{
		pieceIndex = destructible->pieces.count;
		b3Array_Push( destructible->pieces, (b3DestructionPiece){ 0 } );
	}

	b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;

	// Keep the fragment and break arrays of a recycled slot
	int* nodes = piece->nodes;
	int nodeCapacity = piece->nodeCapacity;
	b3BondBreak* breakData = piece->breaks.data;
	int breakCapacity = piece->breaks.capacity;

	*piece = (b3DestructionPiece){ 0 };
	piece->nodes = nodes;
	piece->nodeCapacity = nodeCapacity;
	piece->breaks.data = breakData;
	piece->breaks.capacity = breakCapacity;
	piece->bodyId = B3_NULL_INDEX;
	piece->parentPiece = B3_NULL_INDEX;
	return pieceIndex;
}

void b3FreePiece( b3Destructible* destructible, int pieceIndex )
{
	b3DestructionPiece* piece = b3Array_Get( destructible->pieces, pieceIndex );
	B3_ASSERT( piece->bodyId != B3_NULL_INDEX );

	b3DestroyStressSystem( piece->system );
	piece->system = NULL;
	piece->bodyId = B3_NULL_INDEX;
	piece->nodeCount = 0;
	piece->bondCount = 0;
	piece->flags = 0;
	b3Array_Clear( piece->breaks );
	b3Array_Push( destructible->freePieces, pieceIndex );
}

b3BodyId b3CreateDestructionBody( b3World* world, b3Destructible* destructible, b3BodyType type, b3WorldTransform transform,
								  b3Vec3 originVelocity, b3Vec3 angularVelocity )
{
	b3Destruction* destruction = world->destruction;

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = type;
	bodyDef.position = transform.p;
	bodyDef.rotation = transform.q;
	bodyDef.linearVelocity = originVelocity;
	bodyDef.angularVelocity = angularVelocity;
	bodyDef.enableSleep = destructible->enableSleep;
	bodyDef.name = destructible->name[0] != 0 ? destructible->name : NULL;
	bodyDef.userData = destructible->userData;

	b3BodyId bodyId = b3CreateBody( b3MakeWorldId( world ), &bodyDef );
	int rawId = bodyId.index1 - 1;

	// The owner array follows the world body array
	while ( destruction->bodyOwners.count < world->bodies.count )
	{
		b3Array_Push( destruction->bodyOwners, B3_NULL_INDEX );
	}

	destruction->bodyOwners.data[rawId] = destructible->id;
	destructible->bodyCount += 1;
	return bodyId;
}

// Mark the shapes of a piece as inert if it is a single loose fragment without children. Bonds never
// form again, so this is for good. Most bodies in a pile of rubble end up like this, and so do most
// contacts.
void b3UpdatePieceInert( b3World* world, b3Destructible* destructible, const b3DestructionPiece* piece )
{
	if ( piece->nodeCount != 1 || piece->bondCount != 0 )
	{
		return;
	}

	const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + piece->nodes[0];
	if ( fragment->childCount != 0 )
	{
		return;
	}

	b3Destruction* destruction = world->destruction;
	for ( int i = 0; i < fragment->hullCount; ++i )
	{
		int shapeId = destructible->hullShapes[fragment->firstHull + i];
		if ( shapeId != B3_NULL_INDEX )
		{
			destruction->shapeFragments.data[shapeId].isInert = true;
		}
	}
}

void b3DestroyDestructionBody( b3World* world, b3Destructible* destructible, int bodyId )
{
	b3Destruction* destruction = world->destruction;
	B3_ASSERT( destruction->bodyOwners.data[bodyId] == destructible->id );

	destruction->bodyOwners.data[bodyId] = B3_NULL_INDEX;
	destructible->bodyCount -= 1;

	// The body hook must not treat this as the application destroying the body
	bool wasMutating = destruction->isMutating;
	destruction->isMutating = true;
	b3DestroyBody( b3MakeBodyId( world, bodyId ) );
	destruction->isMutating = wasMutating;
}

void b3CreateFragmentShapes( b3World* world, b3Destructible* destructible, int fragmentIndex, b3BodyId bodyId, bool isDebris )
{
	b3Destruction* destruction = world->destruction;
	const b3StructureData* structure = destructible->structure;
	const b3StructureFragment* fragment = b3GetStructureFragments( structure ) + fragmentIndex;
	const b3StructureMaterial* material = b3GetStructureMaterials( structure ) + fragment->materialIndex;

	b3ShapeDef shapeDef = destructible->shapeDef;

	// The body mass is set from the fragments. The density only matters if the application later asks
	// Box3D to recompute the mass from the shapes.
	shapeDef.density = material->density;
	shapeDef.updateBodyMass = false;
	shapeDef.isSensor = false;

	if ( isDebris )
	{
		shapeDef.filter = destructible->debrisFilter;
	}

	for ( int i = 0; i < fragment->hullCount; ++i )
	{
		int hullIndex = fragment->firstHull + i;
		B3_ASSERT( destructible->hullShapes[hullIndex] == B3_NULL_INDEX );

		b3ShapeId shapeId = b3CreateHullShape( bodyId, &shapeDef, b3GetStructureHull( structure, hullIndex ) );
		if ( B3_IS_NULL( shapeId ) )
		{
			continue;
		}

		int rawId = shapeId.index1 - 1;
		b3Shape* shape = b3Array_Get( world->shapes, rawId );
		shape->flags |= b3_destructibleShape;

		// The shape map follows the world shape array
		while ( destruction->shapeFragments.count < world->shapes.count )
		{
			b3Array_Push( destruction->shapeFragments, ( (b3ShapeFragment){ B3_NULL_INDEX, B3_NULL_INDEX, false } ) );
		}

		destruction->shapeFragments.data[rawId] = (b3ShapeFragment){ destructible->id, fragmentIndex, false };
		destructible->hullShapes[hullIndex] = rawId;
	}
}

void b3DestroyFragmentShapes( b3World* world, b3Destructible* destructible, int fragmentIndex )
{
	b3Destruction* destruction = world->destruction;
	const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + fragmentIndex;

	bool wasMutating = destruction->isMutating;
	destruction->isMutating = true;

	for ( int i = 0; i < fragment->hullCount; ++i )
	{
		int hullIndex = fragment->firstHull + i;
		int rawId = destructible->hullShapes[hullIndex];
		if ( rawId == B3_NULL_INDEX )
		{
			continue;
		}

		const b3Shape* shape = b3Array_Get( world->shapes, rawId );
		b3ShapeId shapeId = { rawId + 1, world->worldId, shape->generation };

		destruction->shapeFragments.data[rawId] = (b3ShapeFragment){ B3_NULL_INDEX, B3_NULL_INDEX, false };
		destructible->hullShapes[hullIndex] = B3_NULL_INDEX;

		// The caller updates the body mass once all its fragments have changed
		b3DestroyShape( shapeId, false );
	}

	destruction->isMutating = wasMutating;
}

void b3PushFragmentEvent( b3World* world, const b3Destructible* destructible, int fragmentIndex, b3FragmentEventType type,
						  int bodyId )
{
	b3Destruction* destruction = world->destruction;
	const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + fragmentIndex;

	b3FragmentEvent event = { 0 };
	event.destructibleId = b3MakeDestructibleId( world, destructible );
	event.fragmentIndex = fragmentIndex;
	event.type = type;
	event.userData = fragment->userData;
	event.transform = b3WorldTransform_identity;

	if ( bodyId != B3_NULL_INDEX )
	{
		const b3Body* body = b3Array_Get( world->bodies, bodyId );

		// The body may already be gone when a fragment is removed along with it
		if ( body->id == bodyId )
		{
			event.bodyId = b3MakeBodyId( world, bodyId );
			event.transform = b3GetBodyTransform( world, bodyId );
		}
	}

	b3Array_Push( destruction->fragmentEvents, event );
}

// Deterministic random number in [0, 1) from a seed and an index
static inline float b3HashUnitFloat( uint32_t seed, uint32_t index )
{
	uint32_t h = seed ^ ( index * 0x9E3779B9u );
	h ^= h >> 16;
	h *= 0x85EBCA6Bu;
	h ^= h >> 13;
	h *= 0xC2B2AE35u;
	h ^= h >> 16;
	return (float)( h >> 8 ) * ( 1.0f / 16777216.0f );
}

static void b3InitBond( const b3StructureData* structure, b3DestructionBond* bond, const b3StructureBond* source, int bondIndex,
						uint32_t seed )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	const b3StructureMaterial* materials = b3GetStructureMaterials( structure );
	const b3StructureFragment* fragmentA = fragments + source->fragmentA;

	*bond = (b3DestructionBond){ 0 };
	bond->nodeA = source->fragmentA;
	bond->nodeB = source->fragmentB;
	bond->rA = b3Sub( source->centroid, fragmentA->center );
	bond->normal = source->normal;
	bond->area0 = source->area;
	bond->area = source->area;
	bond->solverIndex = B3_NULL_INDEX;
	bond->materialIndex = (uint16_t)fragmentA->materialIndex;
	bond->flags = ( source->flags & b3_structureBondRigid ) != 0 ? b3_bondRigid : 0;
	bond->layer = source->layer >= 0 ? (uint8_t)source->layer : B3_ANCHOR_LAYER;

	// The bond is as stiff as a short beam that spans the two fragment centers.
	float length;
	if ( source->fragmentB != B3_NULL_INDEX )
	{
		const b3StructureFragment* fragmentB = fragments + source->fragmentB;
		bond->rB = b3Sub( source->centroid, fragmentB->center );
		length = b3Distance( fragmentA->center, fragmentB->center );

		// The weaker material decides when the bond breaks
		if ( materials[fragmentB->materialIndex].tensileFatalLimit < materials[fragmentA->materialIndex].tensileFatalLimit )
		{
			bond->materialIndex = (uint16_t)fragmentB->materialIndex;
		}
	}
	else
	{
		// A bond to the world: as if the fragment were mirrored across the bond surface.
		bond->rB = b3Vec3_zero;
		length = 2.0f * b3Length( bond->rA );
	}

	// Guard against fragment centers that sit on top of each other
	float minLength = b3MaxFloat( 0.1f * sqrtf( source->area ), B3_LINEAR_SLOP );
	if ( ( bond->flags & b3_bondRigid ) != 0 )
	{
		minLength = b3MaxFloat( minLength, cbrtf( fragmentA->volume ) );
	}

	bond->invLength = 1.0f / b3MaxFloat( length, minLength );

	float variation = materials[bond->materialIndex].strengthVariation;
	bond->strength = 1.0f + variation * ( 2.0f * b3HashUnitFloat( seed, (uint32_t)bondIndex ) - 1.0f );
}

typedef struct b3AnchorQueryContext
{
	b3World* world;
	bool hit;
} b3AnchorQueryContext;

static bool b3AnchorOverlapCallback( b3ShapeId shapeId, void* context )
{
	b3AnchorQueryContext* queryContext = context;
	const b3Shape* shape = b3Array_Get( queryContext->world->shapes, shapeId.index1 - 1 );

	// Another destructible is not solid ground: its fragments can break away later
	if ( shape->flags & b3_destructibleShape )
	{
		return true;
	}

	queryContext->hit = true;
	return false;
}

b3DeclareArray( b3StructureBond );

// Find the faces of support fragments that rest against static shapes and make an anchor for each.
// A face is sampled at its center and near each vertex. The share of samples that touch static
// geometry gives the anchored share of the face.
//
// A fragment that stands on the ground has a face lying exactly in the ground surface. Asking the
// distance query about two shapes that touch exactly is its degenerate case, so every query here is
// pulled a little way into the fragment and given a radius that reaches back out by the same amount.
// The static shape is then at a clean positive distance and the answer is the same.
static void b3FindStaticAnchors( b3World* world, const b3DestructibleDef* def, b3Array( b3StructureBond ) * anchors )
{
	const b3StructureData* structure = def->structure;
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	b3WorldId worldId = b3MakeWorldId( world );
	float tolerance = b3MaxFloat( def->anchorTolerance, 0.0f );
	float inset = b3MaxFloat( 4.0f * B3_LINEAR_SLOP, tolerance );

	b3QueryFilter filter = b3DefaultQueryFilter();
	filter.categoryBits = def->shapeDef.filter.categoryBits;
	filter.maskBits = def->shapeDef.filter.maskBits;

	b3AnchorQueryContext queryContext = { world, false };
	b3Vec3 worldPoints[B3_MAX_HULL_VERTICES];
	b3Vec3 polygon[B3_MAX_HULL_VERTICES];

	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + i;
		if ( fragment->parent != B3_NULL_INDEX )
		{
			continue;
		}

		for ( int j = 0; j < fragment->hullCount; ++j )
		{
			const b3HullData* hull = b3GetStructureHull( structure, fragment->firstHull + j );
			const b3Vec3* points = b3GetHullPoints( hull );
			const b3Plane* planes = b3GetHullPlanes( hull );
			const b3HullFace* faces = b3GetHullFaces( hull );
			const b3HullHalfEdge* edges = b3GetHullEdges( hull );

			// Most fragments touch nothing static. Reject those with a single query. The hull is shrunk
			// toward its center and the radius covers the farthest any point moved, so this can only
			// let through too much, never too little.
			const float shrink = 0.9f;
			float reach = 0.0f;
			for ( int k = 0; k < hull->vertexCount; ++k )
			{
				b3Vec3 offset = b3Sub( points[k], hull->center );
				reach = b3MaxFloat( reach, b3Length( offset ) );
				worldPoints[k] = b3RotateVector( def->rotation, b3MulAdd( hull->center, shrink, offset ) );
			}

			b3ShapeProxy hullProxy = { worldPoints, hull->vertexCount, tolerance + ( 1.0f - shrink ) * reach };
			queryContext.hit = false;
			b3World_OverlapShape_Static( worldId, def->position, &hullProxy, filter, b3AnchorOverlapCallback, &queryContext );
			if ( queryContext.hit == false )
			{
				continue;
			}

			for ( int faceIndex = 0; faceIndex < hull->faceCount; ++faceIndex )
			{
				int count = 0;
				int startEdge = faces[faceIndex].edge;
				int edge = startEdge;
				do
				{
					polygon[count++] = points[edges[edge].origin];
					edge = edges[edge].next;
				}
				while ( edge != startEdge && count < B3_MAX_HULL_VERTICES );

				if ( count < 3 )
				{
					continue;
				}

				b3Vec3 center = b3Vec3_zero;
				for ( int k = 0; k < count; ++k )
				{
					center = b3Add( center, polygon[k] );
				}
				center = b3MulSV( 1.0f / count, center );

				b3Vec3 areaVector = b3Vec3_zero;
				for ( int k = 0; k < count; ++k )
				{
					b3Vec3 e1 = b3Sub( polygon[k], center );
					b3Vec3 e2 = b3Sub( polygon[k + 1 < count ? k + 1 : 0], center );
					areaVector = b3Add( areaVector, b3Cross( e1, e2 ) );
				}
				float faceArea = 0.5f * b3Length( areaVector );

				int hitCount = 0;
				b3Vec3 hitCenter = b3Vec3_zero;
				int sampleCount = count + 1;
				for ( int k = 0; k < sampleCount; ++k )
				{
					b3Vec3 sample = k == count ? center : b3Lerp( center, polygon[k], 0.7f );

					// Step back from the face into the fragment, see the note above
					b3Vec3 inside = b3MulSub( sample, inset, planes[faceIndex].normal );
					b3Vec3 worldSample = b3RotateVector( def->rotation, inside );
					b3ShapeProxy sampleProxy = { &worldSample, 1, inset + tolerance };

					queryContext.hit = false;
					b3World_OverlapShape_Static( worldId, def->position, &sampleProxy, filter, b3AnchorOverlapCallback,
												 &queryContext );
					if ( queryContext.hit )
					{
						hitCount += 1;
						hitCenter = b3Add( hitCenter, sample );
					}
				}

				if ( hitCount == 0 )
				{
					continue;
				}

				b3StructureBond anchor = { 0 };
				anchor.fragmentA = i;
				anchor.fragmentB = B3_NULL_INDEX;
				anchor.centroid = b3MulSV( 1.0f / hitCount, hitCenter );
				anchor.normal = planes[faceIndex].normal;
				anchor.area = faceArea * (float)hitCount / (float)sampleCount;
				anchor.layer = B3_NULL_INDEX;
				if ( anchor.area > 0.0f )
				{
					b3Array_Push( *anchors, anchor );
				}
			}
		}
	}
}

b3DestructibleId b3CreateDestructible( b3WorldId worldId, const b3DestructibleDef* def )
{
	B3_CHECK_DEF( def );
	B3_ASSERT( def->structure != NULL && def->structure->version == B3_STRUCTURE_VERSION );
	B3_ASSERT( b3IsValidPosition( def->position ) );
	B3_ASSERT( b3IsValidQuat( def->rotation ) );
	B3_ASSERT( b3IsValidFloat( def->strengthScale ) && def->strengthScale > 0.0f );

	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL || def->structure == NULL || def->structure->version != B3_STRUCTURE_VERSION )
	{
		return b3_nullDestructibleId;
	}

	b3Destruction* destruction = b3EnsureDestruction( world );
	const b3StructureData* structure = def->structure;
	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	const b3StructureBond* structureBonds = b3GetStructureBonds( structure );
	int fragmentCount = structure->fragmentCount;

	// Find the anchors before any body of this destructible exists, so the queries cannot hit them.
	b3Array( b3StructureBond ) anchors = { 0 };
	if ( def->anchorToStatic )
	{
		b3FindStaticAnchors( world, def, &anchors );
	}

	int id = b3AllocId( &destruction->destructibleIdPool );
	if ( id == destruction->destructibles.count )
	{
		b3Array_Push( destruction->destructibles, (b3Destructible){ 0 } );
	}

	b3Destructible* destructible = b3Array_Get( destruction->destructibles, id );
	uint16_t generation = destructible->generation;
	*destructible = (b3Destructible){ 0 };
	destructible->id = id;
	destructible->generation = (uint16_t)( generation + 1 );
	destructible->structure = structure;
	destructible->userData = def->userData;
	destructible->shapeDef = def->shapeDef;
	destructible->shapeDef.name = NULL;
	destructible->shapeDef.materials = NULL;
	destructible->shapeDef.materialCount = 0;
	destructible->debrisFilter = def->debrisFilter;
	destructible->debrisRadius = def->debrisRadius;
	destructible->debrisLifetime = def->debrisLifetime;
	destructible->minFragmentRadius = def->minFragmentRadius;
	destructible->strengthScale = def->strengthScale;
	destructible->strengthLayerCount = structure->strengthLayerCount;
	memcpy( destructible->layerStrength, structure->layerStrength, sizeof( destructible->layerStrength ) );
	destructible->freezeSleepingDebris = def->freezeSleepingDebris;

	const b3StructureMaterial* materials = b3GetStructureMaterials( structure );
	destructible->minToughness = structure->materialCount > 0 ? FLT_MAX : 0.0f;
	for ( int i = 0; i < structure->materialCount; ++i )
	{
		destructible->minToughness = b3MinFloat( destructible->minToughness, materials[i].toughness );
	}

	destructible->enableSleep = def->enableSleep;
	destructible->enableStress = def->enableStress;
	destructible->enableBondEvents = def->enableBondEvents;
	destructible->anchorBodyId = B3_NULL_INDEX;
	destructible->hostBodyId = b3_nullBodyId;
	destructible->hostFrame = b3Transform_identity;
	destructible->fragmentCount = fragmentCount;
	destructible->gravity = world->gravity;

	if ( def->name != NULL )
	{
		b3StrCpy( destructible->name, (int)sizeof( destructible->name ), def->name );
	}

	destructible->fragmentPiece = b3Alloc( fragmentCount * sizeof( int ) );
	destructible->fragmentLocal = b3AllocZero( fragmentCount * sizeof( int ) );
	destructible->health = b3Alloc( fragmentCount * sizeof( float ) );
	destructible->nodeFlags = b3AllocZero( fragmentCount * sizeof( uint8_t ) );
	destructible->loads = b3AllocZero( fragmentCount * sizeof( b3NodeLoad ) );
	destructible->displacements = b3AllocZero( fragmentCount * sizeof( b3DVec6 ) );
	destructible->hullShapes = b3Alloc( structure->hullCount * sizeof( int ) );

	for ( int i = 0; i < fragmentCount; ++i )
	{
		destructible->fragmentPiece[i] = B3_FRAGMENT_DORMANT;
		destructible->health[i] = b3GetCrushStrength( destructible ) * fragments[i].health;
	}

	for ( int i = 0; i < structure->hullCount; ++i )
	{
		destructible->hullShapes[i] = B3_NULL_INDEX;
	}

	// Bonds: the structure bonds followed by the anchors found above
	int bondCount = structure->bondCount + anchors.count;
	destructible->bondCount = bondCount;
	destructible->intactBondCount = bondCount;
	destructible->bonds = b3Alloc( b3MaxInt( 1, bondCount ) * sizeof( b3DestructionBond ) );

	for ( int i = 0; i < structure->bondCount; ++i )
	{
		b3InitBond( structure, destructible->bonds + i, structureBonds + i, i, def->seed );
	}

	for ( int i = 0; i < anchors.count; ++i )
	{
		int bondIndex = structure->bondCount + i;
		b3InitBond( structure, destructible->bonds + bondIndex, anchors.data + i, bondIndex, def->seed );
	}

	b3Array_Destroy( anchors );

	// Adjacency: the bonds of each fragment, anchors included
	destructible->adjacencyStarts = b3AllocZero( ( fragmentCount + 1 ) * sizeof( int ) );
	int adjacencyCount = 0;
	for ( int i = 0; i < bondCount; ++i )
	{
		const b3DestructionBond* bond = destructible->bonds + i;
		destructible->adjacencyStarts[bond->nodeA + 1] += 1;
		adjacencyCount += 1;
		if ( bond->nodeB != B3_NULL_INDEX )
		{
			destructible->adjacencyStarts[bond->nodeB + 1] += 1;
			adjacencyCount += 1;
		}
	}

	for ( int i = 0; i < fragmentCount; ++i )
	{
		destructible->adjacencyStarts[i + 1] += destructible->adjacencyStarts[i];
	}

	destructible->adjacency = b3Alloc( b3MaxInt( 1, adjacencyCount ) * sizeof( int ) );
	destructible->adjacencyCount = adjacencyCount;
	{
		int* cursors = b3Alloc( fragmentCount * sizeof( int ) );
		memcpy( cursors, destructible->adjacencyStarts, fragmentCount * sizeof( int ) );
		for ( int i = 0; i < bondCount; ++i )
		{
			const b3DestructionBond* bond = destructible->bonds + i;
			destructible->adjacency[cursors[bond->nodeA]++] = i;
			if ( bond->nodeB != B3_NULL_INDEX )
			{
				destructible->adjacency[cursors[bond->nodeB]++] = i;
			}
		}
		b3Free( cursors, fragmentCount * sizeof( int ) );
	}

	// Group the support fragments into pieces by flooding along the bonds
	int* queue = b3Alloc( fragmentCount * sizeof( int ) );
	bool anyAnchored = false;

	for ( int seed = 0; seed < fragmentCount; ++seed )
	{
		if ( fragments[seed].parent != B3_NULL_INDEX || destructible->fragmentPiece[seed] != B3_FRAGMENT_DORMANT )
		{
			continue;
		}

		int pieceIndex = b3AllocPiece( destructible );
		int count = 0;
		int pieceBondCount = 0;
		bool isAnchored = false;

		queue[count++] = seed;
		destructible->fragmentPiece[seed] = pieceIndex;

		for ( int head = 0; head < count; ++head )
		{
			int node = queue[head];
			destructible->fragmentLocal[node] = head;

			for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
			{
				const b3DestructionBond* bond = destructible->bonds + destructible->adjacency[j];
				if ( bond->nodeA == node )
				{
					pieceBondCount += 1;
				}

				if ( bond->nodeB == B3_NULL_INDEX )
				{
					isAnchored = true;
					continue;
				}

				int other = bond->nodeA == node ? bond->nodeB : bond->nodeA;
				if ( destructible->fragmentPiece[other] == B3_FRAGMENT_DORMANT )
				{
					destructible->fragmentPiece[other] = pieceIndex;
					queue[count++] = other;
				}
			}
		}

		b3DestructionPiece* piece = b3Array_Get( destructible->pieces, pieceIndex );
		if ( piece->nodeCapacity < count )
		{
			piece->nodes = b3GrowAlloc( piece->nodes, piece->nodeCapacity * (int)sizeof( int ), count * (int)sizeof( int ) );
			piece->nodeCapacity = count;
		}

		memcpy( piece->nodes, queue, count * sizeof( int ) );
		piece->nodeCount = count;
		piece->bondCount = pieceBondCount;
		piece->flags = b3_pieceSystemDirty | b3_pieceLoadDirty | ( isAnchored ? b3_pieceAnchored : 0 );
		b3ComputePieceMass( destructible, piece );
		anyAnchored = anyAnchored || isAnchored;
	}

	b3Free( queue, fragmentCount * sizeof( int ) );

	// Bodies. Anchored pieces share one body, either static or following the host.
	b3WorldTransform transform = { def->position, def->rotation };

	if ( anyAnchored )
	{
		b3BodyType anchorType = b3_staticBody;
		if ( B3_IS_NON_NULL( def->hostBodyId ) && b3Body_IsValid( def->hostBodyId ) &&
			 b3Body_GetType( def->hostBodyId ) != b3_staticBody )
		{
			anchorType = b3_kinematicBody;
			destructible->hostBodyId = def->hostBodyId;
			destructible->hostFrame = b3InvMulWorldTransforms( b3Body_GetTransform( def->hostBodyId ), transform );
		}

		b3BodyId anchorBodyId = b3CreateDestructionBody( world, destructible, anchorType, transform, b3Vec3_zero, b3Vec3_zero );
		destructible->anchorBodyId = anchorBodyId.index1 - 1;
	}

	for ( int pieceIndex = 0; pieceIndex < destructible->pieces.count; ++pieceIndex )
	{
		b3BodyId bodyId;
		bool isDebris = false;

		// Creating a body does not move the piece array, so this pointer is safe to hold here
		b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;

		if ( piece->flags & b3_pieceAnchored )
		{
			bodyId = b3MakeBodyId( world, destructible->anchorBodyId );
		}
		else
		{
			bodyId = b3CreateDestructionBody( world, destructible, b3_dynamicBody, transform, b3Vec3_zero, b3Vec3_zero );
			isDebris = piece->radius < destructible->debrisRadius;
			if ( isDebris )
			{
				piece->flags |= b3_pieceDebris;
			}
		}

		piece->bodyId = bodyId.index1 - 1;

		for ( int i = 0; i < piece->nodeCount; ++i )
		{
			b3CreateFragmentShapes( world, destructible, piece->nodes[i], bodyId, isDebris );
		}

		if ( ( piece->flags & b3_pieceAnchored ) == 0 )
		{
			b3MassData massData = { piece->mass, piece->center, piece->inertia };
			b3Body_SetMassData( bodyId, massData );
		}
	}

	if ( anyAnchored )
	{
		// Clears the deferred mass flag. A static or kinematic body has no mass.
		b3Body_ApplyMassFromShapes( b3MakeBodyId( world, destructible->anchorBodyId ) );
	}

	return b3MakeDestructibleId( world, destructible );
}

// Free everything a destructible owns except its bodies.
static void b3FreeDestructible( b3World* world, b3Destructible* destructible )
{
	b3Destruction* destruction = world->destruction;
	const b3StructureData* structure = destructible->structure;
	int fragmentCount = destructible->fragmentCount;

	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		b3DestructionPiece* piece = destructible->pieces.data + i;
		b3DestroyStressSystem( piece->system );
		b3Free( piece->nodes, piece->nodeCapacity * sizeof( int ) );
		b3Array_Destroy( piece->breaks );
	}

	b3Array_Destroy( destructible->pieces );
	b3Array_Destroy( destructible->freePieces );
	b3Array_Destroy( destructible->loadedNodes );
	b3Array_Destroy( destructible->crushedNodes );
	b3Array_Destroy( destructible->commitPieces );
	b3Array_Destroy( destructible->commands );

	b3Free( destructible->fragmentPiece, fragmentCount * sizeof( int ) );
	b3Free( destructible->fragmentLocal, fragmentCount * sizeof( int ) );
	b3Free( destructible->health, fragmentCount * sizeof( float ) );
	b3Free( destructible->nodeFlags, fragmentCount * sizeof( uint8_t ) );
	b3Free( destructible->loads, fragmentCount * sizeof( b3NodeLoad ) );
	b3Free( destructible->displacements, fragmentCount * sizeof( b3DVec6 ) );
	b3Free( destructible->hullShapes, structure->hullCount * sizeof( int ) );
	b3Free( destructible->bonds, b3MaxInt( 1, destructible->bondCount ) * sizeof( b3DestructionBond ) );
	b3Free( destructible->adjacencyStarts, ( fragmentCount + 1 ) * sizeof( int ) );

	b3Free( destructible->adjacency, b3MaxInt( 1, destructible->adjacencyCount ) * sizeof( int ) );

	// Preserve the generation for id validation
	uint16_t generation = destructible->generation;
	int id = destructible->id;
	*destructible = (b3Destructible){ 0 };
	destructible->generation = generation;
	destructible->id = B3_NULL_INDEX;

	b3FreeId( &destruction->destructibleIdPool, id );
}

void b3DestroyDestructible( b3DestructibleId destructibleId )
{
	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL || world->destruction == NULL )
	{
		return;
	}

	b3Destruction* destruction = world->destruction;
	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );

	// Forget the shapes. They are destroyed along with their bodies.
	for ( int i = 0; i < destructible->structure->hullCount; ++i )
	{
		int rawId = destructible->hullShapes[i];
		if ( rawId != B3_NULL_INDEX )
		{
			destruction->shapeFragments.data[rawId] = (b3ShapeFragment){ B3_NULL_INDEX, B3_NULL_INDEX, false };
		}
	}

	// Every body with this owner belongs to the destructible. Anchored pieces share the anchor body, so
	// go by the owner array instead of the pieces.
	for ( int i = 0; i < destruction->bodyOwners.count && destructible->bodyCount > 0; ++i )
	{
		if ( destruction->bodyOwners.data[i] == destructible->id )
		{
			b3DestroyDestructionBody( world, destructible, i );
		}
	}

	b3FreeDestructible( world, destructible );
}

void b3DestroyDestruction( b3World* world )
{
	b3Destruction* destruction = world->destruction;
	if ( destruction == NULL )
	{
		return;
	}

	// The world is going away, so the bodies need no cleanup
	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		b3Destructible* destructible = destruction->destructibles.data + i;
		if ( destructible->id != B3_NULL_INDEX && destructible->structure != NULL )
		{
			b3FreeDestructible( world, destructible );
		}
	}

	b3Array_Destroy( destruction->destructibles );
	b3Array_Destroy( destruction->shapeFragments );
	b3Array_Destroy( destruction->bodyOwners );
	b3Array_Destroy( destruction->fragmentEvents );
	b3Array_Destroy( destruction->bondEvents );
	b3Array_Destroy( destruction->bodyEvents );
	b3Array_Destroy( destruction->contactLoads );
	b3Array_Destroy( destruction->tasks );
	b3Array_Destroy( destruction->splitLabels );
	b3Array_Destroy( destruction->splitOrder );
	b3Array_Destroy( destruction->splitShapes );
	b3DestroyIdPool( &destruction->destructibleIdPool );

	b3Free( destruction, sizeof( b3Destruction ) );
	world->destruction = NULL;
}

void b3DestructionOnBodyDestroyed( b3World* world, b3Body* body )
{
	b3Destruction* destruction = world->destruction;
	if ( destruction->isMutating || body->id >= destruction->bodyOwners.count )
	{
		return;
	}

	int owner = destruction->bodyOwners.data[body->id];
	if ( owner == B3_NULL_INDEX )
	{
		return;
	}

	// The application destroyed a body of a destructible. Drop the fragments it carried.
	b3Destructible* destructible = b3Array_Get( destruction->destructibles, owner );
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );

	for ( int pieceIndex = 0; pieceIndex < destructible->pieces.count; ++pieceIndex )
	{
		b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
		if ( piece->bodyId != body->id )
		{
			continue;
		}

		for ( int i = 0; i < piece->nodeCount; ++i )
		{
			int node = piece->nodes[i];
			const b3StructureFragment* fragment = fragments + node;

			for ( int j = 0; j < fragment->hullCount; ++j )
			{
				int hullIndex = fragment->firstHull + j;
				int rawId = destructible->hullShapes[hullIndex];
				if ( rawId != B3_NULL_INDEX )
				{
					destruction->shapeFragments.data[rawId] = (b3ShapeFragment){ B3_NULL_INDEX, B3_NULL_INDEX, false };
					destructible->hullShapes[hullIndex] = B3_NULL_INDEX;
				}
			}

			for ( int j = destructible->adjacencyStarts[node]; j < destructible->adjacencyStarts[node + 1]; ++j )
			{
				b3DestructionBond* bond = destructible->bonds + destructible->adjacency[j];
				if ( ( bond->flags & b3_bondBroken ) == 0 )
				{
					bond->flags |= b3_bondBroken;
					bond->flags &= ~b3_bondBreaking;
					bond->solverIndex = B3_NULL_INDEX;
					destructible->intactBondCount -= 1;
				}
			}

			destructible->fragmentPiece[node] = B3_FRAGMENT_REMOVED;
			destructible->nodeFlags[node] = 0;
			destructible->loads[node] = (b3NodeLoad){ 0 };
			b3PushFragmentEvent( world, destructible, node, b3_fragmentRemoved, body->id );
		}

		b3FreePiece( destructible, pieceIndex );
	}

	if ( destructible->anchorBodyId == body->id )
	{
		destructible->anchorBodyId = B3_NULL_INDEX;
	}

	destruction->bodyOwners.data[body->id] = B3_NULL_INDEX;
	destructible->bodyCount -= 1;
}

void b3DestructionOnShapeDestroyed( b3World* world, b3Shape* shape )
{
	b3Destruction* destruction = world->destruction;
	if ( destruction == NULL || destruction->isMutating || shape->id >= destruction->shapeFragments.count )
	{
		return;
	}

	b3ShapeFragment link = destruction->shapeFragments.data[shape->id];
	if ( link.destructibleId == B3_NULL_INDEX )
	{
		return;
	}

	// The application destroyed a fragment shape. The fragment stays in the bond graph without it.
	b3Destructible* destructible = b3Array_Get( destruction->destructibles, link.destructibleId );
	const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + link.fragmentIndex;
	for ( int i = 0; i < fragment->hullCount; ++i )
	{
		if ( destructible->hullShapes[fragment->firstHull + i] == shape->id )
		{
			destructible->hullShapes[fragment->firstHull + i] = B3_NULL_INDEX;
		}
	}

	destruction->shapeFragments.data[shape->id] = (b3ShapeFragment){ B3_NULL_INDEX, B3_NULL_INDEX, false };
}

// Convert a world impulse at a world point into an impulse and angular impulse on a fragment in
// structure space, and queue it for the next step.
static void b3QueueFragmentImpulse( b3World* world, b3Destructible* destructible, int fragmentIndex, b3Vec3 impulse, b3Pos point )
{
	int pieceIndex = destructible->fragmentPiece[fragmentIndex];
	if ( pieceIndex < 0 )
	{
		return;
	}

	const b3DestructionPiece* piece = b3Array_Get( destructible->pieces, pieceIndex );
	const b3StructureFragment* fragment = b3GetStructureFragments( destructible->structure ) + fragmentIndex;
	b3WorldTransform transform = b3GetBodyTransform( world, piece->bodyId );

	b3Vec3 localImpulse = b3InvRotateVector( transform.q, impulse );
	b3Vec3 localPoint = b3InvTransformWorldPoint( transform, point );

	b3DestructionCommand command = { 0 };
	command.type = b3_commandImpulse;
	command.index = fragmentIndex;
	command.impulse = localImpulse;
	command.angularImpulse = b3Cross( b3Sub( localPoint, fragment->center ), localImpulse );
	b3Array_Push( destructible->commands, command );
}

void b3DestructionOnExplosion( b3World* world, const b3Shape* shape, b3Vec3 impulse, b3Pos point )
{
	b3Destruction* destruction = world->destruction;
	if ( destruction == NULL || shape->id >= destruction->shapeFragments.count )
	{
		return;
	}

	b3ShapeFragment link = destruction->shapeFragments.data[shape->id];
	if ( link.destructibleId == B3_NULL_INDEX )
	{
		return;
	}

	b3Destructible* destructible = b3Array_Get( destruction->destructibles, link.destructibleId );
	b3QueueFragmentImpulse( world, destructible, link.fragmentIndex, impulse, point );
}

b3WorldId b3Destructible_GetWorld( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3MakeWorldId( world );
}

const b3StructureData* b3Destructible_GetStructure( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->structure;
}

void b3Destructible_SetUserData( b3DestructibleId destructibleId, void* userData )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	b3GetDestructibleFromId( world, destructibleId )->userData = userData;
}

void* b3Destructible_GetUserData( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->userData;
}

int b3Destructible_GetBodyCount( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->bodyCount;
}

int b3Destructible_GetBodies( b3DestructibleId destructibleId, b3BodyId* bodyArray, int capacity )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	const b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );

	int count = 0;
	if ( destructible->anchorBodyId != B3_NULL_INDEX && count < capacity )
	{
		bodyArray[count++] = b3MakeBodyId( world, destructible->anchorBodyId );
	}

	for ( int i = 0; i < destructible->pieces.count && count < capacity; ++i )
	{
		const b3DestructionPiece* piece = destructible->pieces.data + i;
		if ( piece->bodyId == B3_NULL_INDEX || piece->bodyId == destructible->anchorBodyId )
		{
			continue;
		}

		bodyArray[count++] = b3MakeBodyId( world, piece->bodyId );
	}

	return count;
}

b3BodyId b3Destructible_GetFragmentBody( b3DestructibleId destructibleId, int fragmentIndex )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	const b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= fragmentIndex && fragmentIndex < destructible->fragmentCount );

	int pieceIndex = destructible->fragmentPiece[fragmentIndex];
	if ( pieceIndex < 0 )
	{
		return b3_nullBodyId;
	}

	return b3MakeBodyId( world, destructible->pieces.data[pieceIndex].bodyId );
}

float b3Destructible_GetFragmentHealth( b3DestructibleId destructibleId, int fragmentIndex )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	const b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= fragmentIndex && fragmentIndex < destructible->fragmentCount );
	return destructible->health[fragmentIndex];
}

int b3Destructible_GetBondCount( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->bondCount;
}

int b3Destructible_GetIntactBondCount( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->intactBondCount;
}

b3BondState b3Destructible_GetBondState( b3DestructibleId destructibleId, int bondIndex )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	const b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= bondIndex && bondIndex < destructible->bondCount );

	const b3DestructionBond* bond = destructible->bonds + bondIndex;
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );
	bool isBroken = ( bond->flags & b3_bondBroken ) != 0;

	b3BondState state = { 0 };
	state.fragmentA = bond->nodeA;
	state.fragmentB = bond->nodeB;
	state.centroid = b3Add( fragments[bond->nodeA].center, bond->rA );
	state.normal = bond->normal;
	state.area = bond->area0;
	state.health = isBroken ? 0.0f : bond->area / bond->area0;
	state.elasticRatio = isBroken ? 0.0f : bond->elasticRatio;
	state.fatalRatio = isBroken ? 0.0f : bond->fatalRatio;
	state.isBroken = isBroken;
	state.layer = bond->layer == B3_ANCHOR_LAYER ? B3_NULL_INDEX : bond->layer;
	return state;
}

void b3Destructible_BreakBond( b3DestructibleId destructibleId, int bondIndex )
{
	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL )
	{
		return;
	}

	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= bondIndex && bondIndex < destructible->bondCount );

	b3DestructionCommand command = { 0 };
	command.type = b3_commandBreakBond;
	command.index = bondIndex;
	b3Array_Push( destructible->commands, command );
}

void b3Destructible_DamageFragment( b3DestructibleId destructibleId, int fragmentIndex, float damage )
{
	B3_ASSERT( b3IsValidFloat( damage ) && damage >= 0.0f );

	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL )
	{
		return;
	}

	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= fragmentIndex && fragmentIndex < destructible->fragmentCount );

	b3DestructionCommand command = { 0 };
	command.type = b3_commandDamageFragment;
	command.index = fragmentIndex;
	command.value = damage;
	b3Array_Push( destructible->commands, command );
}

void b3Destructible_ApplyImpulse( b3DestructibleId destructibleId, int fragmentIndex, b3Vec3 impulse, b3Pos point )
{
	B3_ASSERT( b3IsValidVec3( impulse ) );
	B3_ASSERT( b3IsValidPosition( point ) );

	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL )
	{
		return;
	}

	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= fragmentIndex && fragmentIndex < destructible->fragmentCount );

	int pieceIndex = destructible->fragmentPiece[fragmentIndex];
	if ( pieceIndex < 0 )
	{
		return;
	}

	b3QueueFragmentImpulse( world, destructible, fragmentIndex, impulse, point );

	// Push the body as well. This has no effect on an anchored piece.
	const b3DestructionPiece* piece = b3Array_Get( destructible->pieces, pieceIndex );
	if ( ( piece->flags & ( b3_pieceAnchored | b3_pieceFrozen ) ) == 0 )
	{
		b3Body_ApplyLinearImpulse( b3MakeBodyId( world, piece->bodyId ), impulse, point, true );
	}
}

void b3Destructible_SetStrengthScale( b3DestructibleId destructibleId, float scale )
{
	B3_ASSERT( b3IsValidFloat( scale ) && scale > 0.0f );

	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL )
	{
		return;
	}

	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	float ratio = scale / destructible->strengthScale;
	destructible->strengthScale = scale;

	for ( int i = 0; i < destructible->fragmentCount; ++i )
	{
		destructible->health[i] *= ratio;
	}

	// The limits changed, so every piece must be looked at again
	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		b3DestructionPiece* piece = destructible->pieces.data + i;
		if ( piece->bodyId != B3_NULL_INDEX && piece->system != NULL )
		{
			piece->system->evaluated = false;
		}
	}
}

float b3Destructible_GetStrengthScale( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->strengthScale;
}

void b3Destructible_SetLayerStrength( b3DestructibleId destructibleId, int layer, float scale )
{
	B3_ASSERT( b3IsValidFloat( scale ) && scale > 0.0f );

	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL )
	{
		return;
	}

	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= layer && layer < destructible->strengthLayerCount );
	if ( layer < 0 || layer >= destructible->strengthLayerCount )
	{
		return;
	}

	float ratio = scale / destructible->layerStrength[layer];
	destructible->layerStrength[layer] = scale;

	if ( layer == destructible->strengthLayerCount - 1 )
	{
		// The inside of the fragments. Their remaining health follows.
		for ( int i = 0; i < destructible->fragmentCount; ++i )
		{
			destructible->health[i] *= ratio;
		}

		return;
	}

	// The limits of the bonds on this layer changed, so every piece must be looked at again
	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		b3DestructionPiece* piece = destructible->pieces.data + i;
		if ( piece->bodyId != B3_NULL_INDEX && piece->system != NULL )
		{
			piece->system->evaluated = false;
		}
	}
}

float b3Destructible_GetLayerStrength( b3DestructibleId destructibleId, int layer )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	const b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	B3_ASSERT( 0 <= layer && layer < destructible->strengthLayerCount );
	return 0 <= layer && layer < destructible->strengthLayerCount ? destructible->layerStrength[layer] : 1.0f;
}

// Bring the pieces of a destructible in line with a change of its stress mode. Without stress the
// pieces drop their stress systems, which is most of the memory a large structure holds. With stress
// back on they rebuild them and solve the loads of the moment from rest.
static void b3OnStressModeChanged( b3World* world, b3Destructible* destructible, bool wasActive )
{
	bool isActive = b3IsStressActive( world->destruction, destructible );
	if ( isActive == wasActive )
	{
		return;
	}

	for ( int i = 0; i < destructible->pieces.count; ++i )
	{
		b3DestructionPiece* piece = destructible->pieces.data + i;
		if ( piece->bodyId == B3_NULL_INDEX || ( piece->flags & b3_pieceFrozen ) != 0 )
		{
			continue;
		}

		if ( isActive )
		{
			piece->flags |= b3_pieceSystemDirty | b3_pieceLoadDirty;
			continue;
		}

		b3DestroyStressSystem( piece->system );
		piece->system = NULL;
		piece->flags &= ~( b3_pieceCreeping | b3_pieceUnloaded | b3_pieceLoadDirty );

		// A slow failure kept this body awake. Nothing fails slowly without stress.
		if ( piece->flags & b3_pieceSleepDisabled )
		{
			b3Body_EnableSleep( b3MakeBodyId( world, piece->bodyId ), destructible->enableSleep );
			piece->flags &= ~b3_pieceSleepDisabled;
		}
	}

	// The stress the bonds showed is gone, and the old displacements are no start for a new solve
	for ( int i = 0; i < destructible->bondCount; ++i )
	{
		b3DestructionBond* bond = destructible->bonds + i;
		bond->elasticRatio = 0.0f;
		bond->fatalRatio = 0.0f;
		bond->solverIndex = B3_NULL_INDEX;
	}

	memset( destructible->displacements, 0, destructible->fragmentCount * sizeof( b3DVec6 ) );
}

void b3Destructible_EnableStress( b3DestructibleId destructibleId, bool flag )
{
	b3World* world = b3GetUnlockedWorld( destructibleId.world0 );
	if ( world == NULL )
	{
		return;
	}

	b3Destructible* destructible = b3GetDestructibleFromId( world, destructibleId );
	bool wasActive = b3IsStressActive( world->destruction, destructible );
	destructible->enableStress = flag;
	b3OnStressModeChanged( world, destructible, wasActive );
}

bool b3Destructible_IsStressEnabled( b3DestructibleId destructibleId )
{
	b3World* world = b3GetWorld( destructibleId.world0 );
	return b3GetDestructibleFromId( world, destructibleId )->enableStress;
}

static b3ShapeFragment b3GetShapeFragment( b3World* world, b3ShapeId shapeId )
{
	b3ShapeFragment none = { B3_NULL_INDEX, B3_NULL_INDEX, false };
	b3Destruction* destruction = world->destruction;
	int rawId = shapeId.index1 - 1;
	if ( destruction == NULL || rawId < 0 || rawId >= destruction->shapeFragments.count )
	{
		return none;
	}

	const b3Shape* shape = b3Array_Get( world->shapes, rawId );
	if ( shape->generation != shapeId.generation || ( shape->flags & b3_destructibleShape ) == 0 )
	{
		return none;
	}

	return destruction->shapeFragments.data[rawId];
}

b3DestructibleId b3Shape_GetDestructible( b3ShapeId shapeId )
{
	b3World* world = b3GetWorld( shapeId.world0 );
	b3ShapeFragment link = b3GetShapeFragment( world, shapeId );
	if ( link.destructibleId == B3_NULL_INDEX )
	{
		return b3_nullDestructibleId;
	}

	return b3MakeDestructibleId( world, b3Array_Get( world->destruction->destructibles, link.destructibleId ) );
}

int b3Shape_GetFragmentIndex( b3ShapeId shapeId )
{
	b3World* world = b3GetWorld( shapeId.world0 );
	return b3GetShapeFragment( world, shapeId ).fragmentIndex;
}

b3DestructibleId b3Body_GetDestructible( b3BodyId bodyId )
{
	b3World* world = b3GetWorld( bodyId.world0 );
	b3Destruction* destruction = world->destruction;
	int rawId = bodyId.index1 - 1;
	if ( destruction == NULL || rawId < 0 || rawId >= destruction->bodyOwners.count )
	{
		return b3_nullDestructibleId;
	}

	int owner = destruction->bodyOwners.data[rawId];
	if ( owner == B3_NULL_INDEX )
	{
		return b3_nullDestructibleId;
	}

	return b3MakeDestructibleId( world, b3Array_Get( destruction->destructibles, owner ) );
}

void b3World_SetDestructionSettings( b3WorldId worldId, const b3DestructionSettings* settings )
{
	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL )
	{
		return;
	}

	B3_ASSERT( settings->maxIterations > 0 && settings->workBudget > 0 );
	B3_ASSERT( b3IsValidFloat( settings->solverTolerance ) && settings->solverTolerance > 0.0f );
	B3_ASSERT( b3IsValidFloat( settings->damageTolerance ) && settings->damageTolerance > 0.0f );
	B3_ASSERT( b3IsValidFloat( settings->loadSmoothingTime ) && settings->loadSmoothingTime >= 0.0f );
	B3_ASSERT( b3IsValidFloat( settings->impactRadius ) && settings->impactRadius >= 0.0f );
	B3_ASSERT( b3IsValidFloat( settings->impactDamageScale ) && settings->impactDamageScale >= 0.0f );
	B3_ASSERT( b3IsValidFloat( settings->impactFractureScale ) && settings->impactFractureScale >= 0.0f );

	b3Destruction* destruction = b3EnsureDestruction( world );
	bool wasEnabled = destruction->settings.enableStress;

	destruction->settings = *settings;
	destruction->settings.maxIterations = b3MaxInt( 1, settings->maxIterations );
	destruction->settings.workBudget = b3MaxInt( 1, settings->workBudget );
	destruction->settings.maxNewBodiesPerStep = b3MaxInt( 1, settings->maxNewBodiesPerStep );

	if ( settings->enableStress != wasEnabled )
	{
		for ( int i = 0; i < destruction->destructibles.count; ++i )
		{
			b3Destructible* destructible = destruction->destructibles.data + i;
			if ( destructible->id != B3_NULL_INDEX && destructible->structure != NULL )
			{
				b3OnStressModeChanged( world, destructible, wasEnabled && destructible->enableStress );
			}
		}
	}
}

b3DestructionSettings b3World_GetDestructionSettings( b3WorldId worldId )
{
	b3World* world = b3GetWorldFromId( worldId );
	if ( world->destruction == NULL )
	{
		return b3DefaultDestructionSettings();
	}

	return world->destruction->settings;
}

b3DestructionEvents b3World_GetDestructionEvents( b3WorldId worldId )
{
	b3DestructionEvents events = { 0 };

	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL || world->destruction == NULL )
	{
		return events;
	}

	b3Destruction* destruction = world->destruction;
	events.fragmentEvents = destruction->fragmentEvents.data;
	events.fragmentCount = destruction->fragmentEvents.count;
	events.bondEvents = destruction->bondEvents.data;
	events.bondCount = destruction->bondEvents.count;
	events.bodyEvents = destruction->bodyEvents.data;
	events.bodyCount = destruction->bodyEvents.count;
	return events;
}

b3DestructionProfile b3World_GetDestructionProfile( b3WorldId worldId )
{
	b3World* world = b3GetWorldFromId( worldId );
	if ( world->destruction == NULL )
	{
		return (b3DestructionProfile){ 0 };
	}

	return world->destruction->profile;
}

int b3GetDestructionByteCount( b3World* world )
{
	b3Destruction* destruction = world->destruction;
	if ( destruction == NULL )
	{
		return 0;
	}

	int64_t byteCount = sizeof( b3Destruction );
	byteCount += b3Array_ByteCount( destruction->destructibles );
	byteCount += b3Array_ByteCount( destruction->shapeFragments );
	byteCount += b3Array_ByteCount( destruction->bodyOwners );
	byteCount += b3Array_ByteCount( destruction->fragmentEvents );
	byteCount += b3Array_ByteCount( destruction->bondEvents );
	byteCount += b3Array_ByteCount( destruction->bodyEvents );
	byteCount += b3Array_ByteCount( destruction->contactLoads );
	byteCount += b3Array_ByteCount( destruction->tasks );

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		const b3Destructible* destructible = destruction->destructibles.data + i;
		if ( destructible->id == B3_NULL_INDEX || destructible->structure == NULL )
		{
			continue;
		}

		int fragmentCount = destructible->fragmentCount;
		byteCount += fragmentCount * (int64_t)( 2 * sizeof( int ) + sizeof( float ) + sizeof( uint8_t ) + sizeof( b3NodeLoad ) +
												sizeof( b3DVec6 ) );
		byteCount += destructible->structure->hullCount * (int64_t)sizeof( int );
		byteCount += destructible->bondCount * (int64_t)sizeof( b3DestructionBond );
		byteCount += ( fragmentCount + 1 + destructible->adjacencyCount ) * (int64_t)sizeof( int );
		byteCount += b3Array_ByteCount( destructible->pieces );

		for ( int j = 0; j < destructible->pieces.count; ++j )
		{
			const b3DestructionPiece* piece = destructible->pieces.data + j;
			byteCount += piece->nodeCapacity * (int64_t)sizeof( int );
			byteCount += b3Array_ByteCount( piece->breaks );
			byteCount += piece->system != NULL ? b3GetStressSystemByteCount( piece->system ) : 0;
		}
	}

	return (int)b3MinInt( INT32_MAX, (int)byteCount );
}

b3DestructionCounters b3World_GetDestructionCounters( b3WorldId worldId )
{
	b3DestructionCounters counters = { 0 };

	b3World* world = b3GetWorldFromId( worldId );
	b3Destruction* destruction = world->destruction;
	if ( destruction == NULL )
	{
		return counters;
	}

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		const b3Destructible* destructible = destruction->destructibles.data + i;
		if ( destructible->id == B3_NULL_INDEX || destructible->structure == NULL )
		{
			continue;
		}

		counters.destructibleCount += 1;
		counters.pieceCount += destructible->pieces.count - destructible->freePieces.count;
		counters.bodyCount += destructible->bodyCount;
		counters.bondCount += destructible->intactBondCount;
	}

	counters.solvedPieceCount = destruction->solvedPieceCount;
	counters.iterationCount = destruction->iterationCount;
	counters.brokenBondCount = destruction->brokenBondCount;
	counters.newBodyCount = destruction->newBodyCount;
	counters.pendingSplitCount = destruction->pendingSplitCount;
	counters.byteCount = b3GetDestructionByteCount( world );
	return counters;
}

// Map a stress ratio to the heat map used by the bond debug draw: green to red up to the elastic
// limit, then blue to magenta up to the fatal limit.
static b3HexColor b3GetBondStressColor( float elasticRatio, float fatalRatio )
{
	if ( elasticRatio < 1.0f )
	{
		float t = b3ClampFloat( elasticRatio, 0.0f, 1.0f );
		uint32_t red = (uint32_t)( 255.0f * t );
		uint32_t green = (uint32_t)( 255.0f * ( 1.0f - t ) );
		return (b3HexColor)( ( red << 16 ) | ( green << 8 ) );
	}

	float t = b3ClampFloat( fatalRatio, 0.0f, 1.0f );
	uint32_t red = (uint32_t)( 255.0f * t );
	return (b3HexColor)( ( red << 16 ) | 0xFF );
}

void b3DrawDestruction( b3World* world, b3DebugDraw* draw )
{
	b3Destruction* destruction = world->destruction;
	if ( destruction == NULL || draw->DrawSegmentFcn == NULL || ( draw->drawBonds == false && draw->drawBondStress == false ) )
	{
		return;
	}

	b3AABB bounds = draw->drawingBounds;

	for ( int i = 0; i < destruction->destructibles.count; ++i )
	{
		const b3Destructible* destructible = destruction->destructibles.data + i;
		if ( destructible->id == B3_NULL_INDEX || destructible->structure == NULL )
		{
			continue;
		}

		const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );

		for ( int j = 0; j < destructible->bondCount; ++j )
		{
			const b3DestructionBond* bond = destructible->bonds + j;
			if ( bond->flags & b3_bondBroken )
			{
				continue;
			}

			int pieceIndex = destructible->fragmentPiece[bond->nodeA];
			if ( pieceIndex < 0 )
			{
				continue;
			}

			const b3DestructionPiece* piece = destructible->pieces.data + pieceIndex;
			b3WorldTransform transform = b3GetBodyTransform( world, piece->bodyId );

			b3Vec3 centerA = fragments[bond->nodeA].center;
			b3Vec3 centroid = b3Add( centerA, bond->rA );
			b3Pos p1 = b3TransformWorldPoint( transform, centerA );
			b3Pos p2 = b3TransformWorldPoint( transform, bond->nodeB != B3_NULL_INDEX ? fragments[bond->nodeB].center : centroid );

			b3Vec3 mid = b3Lerp( b3ToVec3( p1 ), b3ToVec3( p2 ), 0.5f );
			if ( mid.x < bounds.lowerBound.x || mid.y < bounds.lowerBound.y || mid.z < bounds.lowerBound.z ||
				 mid.x > bounds.upperBound.x || mid.y > bounds.upperBound.y || mid.z > bounds.upperBound.z )
			{
				continue;
			}

			b3HexColor color = b3_colorLightGray;
			if ( draw->drawBondStress )
			{
				color = b3GetBondStressColor( bond->elasticRatio, bond->fatalRatio );
			}
			else if ( bond->nodeB == B3_NULL_INDEX )
			{
				color = b3_colorOrange;
			}

			draw->DrawSegmentFcn( p1, p2, color, draw->context );

			if ( bond->nodeB == B3_NULL_INDEX && draw->DrawPointFcn != NULL )
			{
				// Mark the anchors
				draw->DrawPointFcn( p2, 6.0f, color, draw->context );
			}
		}
	}
}
