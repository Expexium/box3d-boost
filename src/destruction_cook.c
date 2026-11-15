// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

// Structure cooking
//
// Turns a list of fragments into a b3StructureData: a single block of memory with no pointers that
// holds the fragments, their hulls in structure space, the fragment hierarchy, and the bond graph.
//
// Bonds are generated from geometry. Two fragments are bonded where a face of one lies against a
// face of the other. Fracture tools cut fragments apart with planes, so neighbors share exactly
// coplanar faces and the overlap of the two face polygons is the bonded surface. Fragments that
// overlap without matching faces, such as kit pieces pushed into each other, fall back to the
// contact manifold between the hulls.

#include "core.h"
#include "destruction.h"

#include "box3d/collision.h"
#include "box3d/constants.h"
#include "box3d/destruction.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

// Faces match when their normals oppose within about 10 degrees.
#define B3_BOND_FACE_COSINE ( -0.985f )

// The most vertices of a clipped face polygon
#define B3_BOND_POLYGON_CAPACITY ( 2 * B3_MAX_HULL_VERTICES )

b3StructureMaterial b3DefaultStructureMaterial( void )
{
	// Plain concrete. Tensile strength is about a tenth of the compressive strength, which is why
	// bending brings structures down. The toughness is far above the fracture energy of a single
	// crack because crushing a fragment into rubble creates many surfaces and dissipates energy
	// in friction.
	b3StructureMaterial material = { 0 };
	material.density = 2400.0f;
	material.tensileElasticLimit = 2.0e6f;
	material.tensileFatalLimit = 3.0e6f;
	material.compressiveElasticLimit = 20.0e6f;
	material.compressiveFatalLimit = 30.0e6f;
	material.shearElasticLimit = 3.0e6f;
	material.shearFatalLimit = 4.5e6f;
	material.stiffness = 1.0f;
	material.toughness = 3.0e4f;
	material.creepTime = 0.75f;
	material.strengthVariation = 0.1f;
	material.userMaterialId = 0;
	return material;
}

b3FragmentDef b3DefaultFragmentDef( void )
{
	b3FragmentDef def = { 0 };
	def.transform = b3Transform_identity;
	def.parentIndex = B3_NULL_INDEX;
	return def;
}

b3StructureDef b3DefaultStructureDef( void )
{
	b3StructureDef def = { 0 };
	def.generateBonds = true;
	def.bondGap = 2.0f * B3_LINEAR_SLOP;
	def.minBondArea = 4.0f * B3_LINEAR_SLOP * B3_LINEAR_SLOP;
	def.internalValue = B3_SECRET_COOKIE;
	return def;
}

// Area weighted sum of the surfaces shared by two fragments
typedef struct b3SharedSurface
{
	float area;
	b3Vec3 centroid;
	b3Vec3 normal;
} b3SharedSurface;

typedef struct b3Point2
{
	float x, y;
} b3Point2;

static int b3GetFacePolygon( const b3HullData* hull, int faceIndex, b3Vec3* polygon, int capacity )
{
	const b3HullFace* faces = b3GetHullFaces( hull );
	const b3HullHalfEdge* edges = b3GetHullEdges( hull );
	const b3Vec3* points = b3GetHullPoints( hull );

	int startEdge = faces[faceIndex].edge;
	int edge = startEdge;
	int count = 0;
	do
	{
		polygon[count] = points[edges[edge].origin];
		count += 1;
		edge = edges[edge].next;
	}
	while ( edge != startEdge && count < capacity );

	return count;
}

static float b3PolygonArea2( const b3Point2* polygon, int count )
{
	// Twice the signed area
	float area = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		b3Point2 p1 = polygon[i];
		b3Point2 p2 = polygon[i + 1 < count ? i + 1 : 0];
		area += p1.x * p2.y - p2.x * p1.y;
	}
	return area;
}

static void b3ReversePolygon( b3Point2* polygon, int count )
{
	for ( int i = 0, j = count - 1; i < j; ++i, --j )
	{
		b3Point2 temp = polygon[i];
		polygon[i] = polygon[j];
		polygon[j] = temp;
	}
}

// Clip a convex polygon against a counter-clockwise convex polygon (Sutherland-Hodgman).
// Returns the vertex count of the result, which is written to the subject buffer.
static int b3ClipPolygon( b3Point2* subject, int subjectCount, const b3Point2* clip, int clipCount, b3Point2* scratch )
{
	b3Point2* input = subject;
	b3Point2* output = scratch;
	int count = subjectCount;

	for ( int i = 0; i < clipCount && count > 0; ++i )
	{
		b3Point2 c1 = clip[i];
		b3Point2 c2 = clip[i + 1 < clipCount ? i + 1 : 0];
		float ex = c2.x - c1.x;
		float ey = c2.y - c1.y;

		int outCount = 0;
		b3Point2 previous = input[count - 1];
		float previousSide = ex * ( previous.y - c1.y ) - ey * ( previous.x - c1.x );

		for ( int j = 0; j < count; ++j )
		{
			b3Point2 current = input[j];
			float currentSide = ex * ( current.y - c1.y ) - ey * ( current.x - c1.x );

			// The inside is to the left of the clip edge
			if ( ( previousSide < 0.0f ) != ( currentSide < 0.0f ) )
			{
				float t = previousSide / ( previousSide - currentSide );
				b3Point2 crossing = { previous.x + t * ( current.x - previous.x ), previous.y + t * ( current.y - previous.y ) };
				if ( outCount < B3_BOND_POLYGON_CAPACITY )
				{
					output[outCount++] = crossing;
				}
			}

			if ( currentSide >= 0.0f && outCount < B3_BOND_POLYGON_CAPACITY )
			{
				output[outCount++] = current;
			}

			previous = current;
			previousSide = currentSide;
		}

		b3Point2* temp = input;
		input = output;
		output = temp;
		count = outCount;
	}

	if ( input != subject )
	{
		memcpy( subject, input, count * sizeof( b3Point2 ) );
	}

	return count;
}

// Add the overlap of every pair of opposed coplanar faces of two hulls.
static void b3MatchHullFaces( const b3HullData* hullA, const b3HullData* hullB, float gap, b3SharedSurface* surface )
{
	const b3Plane* planesA = b3GetHullPlanes( hullA );
	const b3Plane* planesB = b3GetHullPlanes( hullB );
	int faceCountA = hullA->faceCount;
	int faceCountB = hullB->faceCount;

	b3Vec3 polygonA[B3_MAX_HULL_VERTICES];
	b3Vec3 polygonB[B3_MAX_HULL_VERTICES];
	b3Point2 flatA[B3_MAX_HULL_VERTICES];
	b3Point2 flatB[B3_BOND_POLYGON_CAPACITY];
	b3Point2 scratch[B3_BOND_POLYGON_CAPACITY];

	for ( int faceA = 0; faceA < faceCountA; ++faceA )
	{
		b3Plane planeA = planesA[faceA];
		int countA = 0;

		for ( int faceB = 0; faceB < faceCountB; ++faceB )
		{
			b3Plane planeB = planesB[faceB];
			if ( b3Dot( planeA.normal, planeB.normal ) > B3_BOND_FACE_COSINE )
			{
				continue;
			}

			int countB = b3GetFacePolygon( hullB, faceB, polygonB, B3_MAX_HULL_VERTICES );
			if ( countB < 3 )
			{
				continue;
			}

			// Distance of face B from the plane of face A. Negative when the hulls overlap.
			b3Vec3 centerB = b3Vec3_zero;
			for ( int i = 0; i < countB; ++i )
			{
				centerB = b3Add( centerB, polygonB[i] );
			}
			centerB = b3MulSV( 1.0f / countB, centerB );

			float separation = b3Dot( planeA.normal, centerB ) - planeA.offset;
			if ( separation > gap || separation < -2.0f * gap )
			{
				continue;
			}

			if ( countA == 0 )
			{
				countA = b3GetFacePolygon( hullA, faceA, polygonA, B3_MAX_HULL_VERTICES );
				if ( countA < 3 )
				{
					break;
				}
			}

			// Flatten both polygons onto the plane of face A, relative to its first vertex.
			b3Vec3 origin = polygonA[0];
			b3Vec3 axisX = b3Perp( planeA.normal );
			b3Vec3 axisY = b3Cross( planeA.normal, axisX );

			for ( int i = 0; i < countA; ++i )
			{
				b3Vec3 d = b3Sub( polygonA[i], origin );
				flatA[i] = (b3Point2){ b3Dot( d, axisX ), b3Dot( d, axisY ) };
			}

			for ( int i = 0; i < countB; ++i )
			{
				b3Vec3 d = b3Sub( polygonB[i], origin );
				flatB[i] = (b3Point2){ b3Dot( d, axisX ), b3Dot( d, axisY ) };
			}

			// The clipper needs a counter-clockwise clip polygon. Face B winds the other way because
			// its normal opposes face A.
			if ( b3PolygonArea2( flatA, countA ) < 0.0f )
			{
				b3ReversePolygon( flatA, countA );
			}

			int count = b3ClipPolygon( flatB, countB, flatA, countA, scratch );
			if ( count < 3 )
			{
				continue;
			}

			float area2 = 0.0f;
			float cx = 0.0f, cy = 0.0f;
			for ( int i = 0; i < count; ++i )
			{
				b3Point2 p1 = flatB[i];
				b3Point2 p2 = flatB[i + 1 < count ? i + 1 : 0];
				float cross = p1.x * p2.y - p2.x * p1.y;
				area2 += cross;
				cx += ( p1.x + p2.x ) * cross;
				cy += ( p1.y + p2.y ) * cross;
			}

			float area = 0.5f * b3AbsFloat( area2 );
			if ( area <= FLT_EPSILON )
			{
				continue;
			}

			cx /= 3.0f * area2;
			cy /= 3.0f * area2;

			// Put the centroid half way between the two faces.
			b3Vec3 centroid = b3Add( origin, b3Blend2( cx, axisX, cy, axisY ) );
			centroid = b3MulAdd( centroid, 0.5f * separation, planeA.normal );

			surface->area += area;
			surface->centroid = b3MulAdd( surface->centroid, area, centroid );
			surface->normal = b3MulAdd( surface->normal, area, planeA.normal );
		}
	}
}

// Fallback for hulls that touch or overlap without matching faces. Uses the contact patch.
static void b3MatchHullManifold( const b3HullData* hullA, const b3HullData* hullB, float gap, b3SharedSurface* surface )
{
	enum
	{
		pointCapacity = 16
	};

	b3LocalManifoldPoint points[pointCapacity];
	b3LocalManifold manifold = { 0 };
	manifold.points = points;

	b3SATCache cache = { 0 };
	b3CollideHulls( &manifold, pointCapacity, hullA, hullB, b3Transform_identity, &cache );

	// Keep the points that are actually touching.
	b3Vec3 patch[pointCapacity];
	int count = 0;
	for ( int i = 0; i < manifold.pointCount; ++i )
	{
		if ( points[i].separation <= gap )
		{
			patch[count++] = points[i].point;
		}
	}

	// A point or an edge contact has no area.
	if ( count < 3 )
	{
		return;
	}

	b3Vec3 normal = manifold.normal;
	b3Vec3 axisX = b3Perp( normal );
	b3Vec3 axisY = b3Cross( normal, axisX );

	b3Vec3 center = b3Vec3_zero;
	for ( int i = 0; i < count; ++i )
	{
		center = b3Add( center, patch[i] );
	}
	center = b3MulSV( 1.0f / count, center );

	// Order the points around the center so they form a polygon.
	float angles[pointCapacity];
	b3Point2 flat[pointCapacity];
	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 d = b3Sub( patch[i], center );
		flat[i] = (b3Point2){ b3Dot( d, axisX ), b3Dot( d, axisY ) };
		angles[i] = b3Atan2( flat[i].y, flat[i].x );
	}

	for ( int i = 1; i < count; ++i )
	{
		float angle = angles[i];
		b3Point2 point = flat[i];
		int j = i - 1;
		while ( j >= 0 && angles[j] > angle )
		{
			angles[j + 1] = angles[j];
			flat[j + 1] = flat[j];
			j -= 1;
		}
		angles[j + 1] = angle;
		flat[j + 1] = point;
	}

	float area = 0.5f * b3AbsFloat( b3PolygonArea2( flat, count ) );
	if ( area <= FLT_EPSILON )
	{
		return;
	}

	surface->area += area;
	surface->centroid = b3MulAdd( surface->centroid, area, center );
	surface->normal = b3MulAdd( surface->normal, area, normal );
}

// A pair of hulls from two different fragments whose bounds overlap
typedef struct b3HullPair
{
	int otherFragment;
	int hullA;
	int hullB;
} b3HullPair;

static int b3CompareHullPairs( const void* a, const void* b )
{
	const b3HullPair* pairA = a;
	const b3HullPair* pairB = b;
	if ( pairA->otherFragment != pairB->otherFragment )
	{
		return pairA->otherFragment < pairB->otherFragment ? -1 : 1;
	}
	if ( pairA->hullA != pairB->hullA )
	{
		return pairA->hullA < pairB->hullA ? -1 : 1;
	}
	if ( pairA->hullB != pairB->hullB )
	{
		return pairA->hullB < pairB->hullB ? -1 : 1;
	}
	return 0;
}

static int b3CompareBonds( const void* a, const void* b )
{
	const b3StructureBond* bondA = a;
	const b3StructureBond* bondB = b;
	if ( bondA->fragmentA != bondB->fragmentA )
	{
		return bondA->fragmentA < bondB->fragmentA ? -1 : 1;
	}

	// World bonds have fragmentB of -1 and sort first
	if ( bondA->fragmentB != bondB->fragmentB )
	{
		return bondA->fragmentB < bondB->fragmentB ? -1 : 1;
	}
	return 0;
}

static int b3CompareKeys( const void* a, const void* b )
{
	uint64_t keyA = *(const uint64_t*)a;
	uint64_t keyB = *(const uint64_t*)b;
	return keyA < keyB ? -1 : ( keyA > keyB ? 1 : 0 );
}

static inline uint64_t b3FragmentPairKey( int fragmentA, int fragmentB )
{
	uint32_t low = (uint32_t)b3MinInt( fragmentA, fragmentB );
	uint32_t high = (uint32_t)b3MaxInt( fragmentA, fragmentB );
	return ( (uint64_t)low << 32 ) | high;
}

typedef struct b3PairQueryContext
{
	const int* hullFragments;
	const b3StructureFragment* fragments;
	int fragmentA;
	int hullA;
	b3HullPair* pairs;
	int pairCount;
	int pairCapacity;
} b3PairQueryContext;

static bool b3PairQueryCallback( int proxyId, uint64_t userData, void* context )
{
	B3_UNUSED( proxyId );
	b3PairQueryContext* queryContext = context;

	int hullB = (int)userData;
	int fragmentB = queryContext->hullFragments[hullB];
	int fragmentA = queryContext->fragmentA;

	// Visit each pair of fragments once
	if ( fragmentB <= fragmentA )
	{
		return true;
	}

	// Support fragments bond with each other. Siblings share the surface their parent breaks along.
	if ( queryContext->fragments[fragmentA].parent != queryContext->fragments[fragmentB].parent )
	{
		return true;
	}

	if ( queryContext->pairCount == queryContext->pairCapacity )
	{
		int newCapacity = 2 * queryContext->pairCapacity;
		queryContext->pairs = B3_GROW( queryContext->pairs, queryContext->pairCapacity, newCapacity );
		queryContext->pairCapacity = newCapacity;
	}

	queryContext->pairs[queryContext->pairCount++] = (b3HullPair){ fragmentB, queryContext->hullA, hullB };
	return true;
}

b3StructureData* b3CreateStructure( const b3StructureDef* def )
{
	B3_CHECK_DEF( def );

	int fragmentCount = def->fragmentCount;
	if ( fragmentCount <= 0 || def->fragments == NULL )
	{
		return NULL;
	}

	// Validate the fragments and count the hulls
	int hullCount = 0;
	for ( int i = 0; i < fragmentCount; ++i )
	{
		const b3FragmentDef* fragmentDef = def->fragments + i;
		if ( fragmentDef->hullCount <= 0 || fragmentDef->hulls == NULL )
		{
			B3_ASSERT( false );
			return NULL;
		}

		if ( fragmentDef->parentIndex != B3_NULL_INDEX &&
			 ( fragmentDef->parentIndex < 0 || fragmentDef->parentIndex >= fragmentCount || fragmentDef->parentIndex == i ) )
		{
			B3_ASSERT( false );
			return NULL;
		}

		if ( def->materials != NULL && ( fragmentDef->materialIndex < 0 || fragmentDef->materialIndex >= def->materialCount ) )
		{
			B3_ASSERT( false );
			return NULL;
		}

		B3_ASSERT( b3IsValidTransform( fragmentDef->transform ) );
		hullCount += fragmentDef->hullCount;
	}

	b3StructureMaterial defaultMaterial = b3DefaultStructureMaterial();
	const b3StructureMaterial* materials = def->materials != NULL && def->materialCount > 0 ? def->materials : &defaultMaterial;
	int materialCount = def->materials != NULL && def->materialCount > 0 ? def->materialCount : 1;

	b3StructureFragment* fragments = b3AllocZero( fragmentCount * sizeof( b3StructureFragment ) );
	b3HullData** hulls = b3AllocZero( hullCount * sizeof( b3HullData* ) );
	int* hullFragments = b3Alloc( hullCount * sizeof( int ) );
	int* depths = b3AllocZero( fragmentCount * sizeof( int ) );
	float* sharedAreas = b3AllocZero( fragmentCount * sizeof( float ) );
	float* surfaceAreas = b3AllocZero( fragmentCount * sizeof( float ) );

	b3StructureData* structure = NULL;
	b3StructureBond* bonds = NULL;
	int bondCount = 0;
	int bondCapacity = 0;
	uint64_t* explicitKeys = NULL;
	int explicitKeyCount = 0;
	int adjacencyCount = 0;
	int* adjacencyCursors = NULL;
	int* childCursors = NULL;
	int* children = NULL;
	int childTotal = 0;
	b3DynamicTree tree = { 0 };
	bool haveTree = false;
	b3PairQueryContext queryContext = { 0 };
	bool failed = false;

	// Bring the hulls into structure space and compute the mass properties
	int hullIndex = 0;
	int maxDepth = 0;
	for ( int i = 0; i < fragmentCount && failed == false; ++i )
	{
		const b3FragmentDef* fragmentDef = def->fragments + i;
		b3StructureFragment* fragment = fragments + i;
		const b3StructureMaterial* material = materials + ( def->materials != NULL ? fragmentDef->materialIndex : 0 );

		b3Transform xf = fragmentDef->transform;
		bool isIdentity = xf.p.x == 0.0f && xf.p.y == 0.0f && xf.p.z == 0.0f && xf.q.v.x == 0.0f && xf.q.v.y == 0.0f &&
						  xf.q.v.z == 0.0f && xf.q.s == 1.0f;

		fragment->firstHull = hullIndex;
		fragment->hullCount = fragmentDef->hullCount;
		fragment->parent = fragmentDef->parentIndex;
		fragment->materialIndex = def->materials != NULL ? fragmentDef->materialIndex : 0;
		fragment->userData = fragmentDef->userData;
		fragment->flags = 0;

		// Walk up to find the depth. A cycle would never end, so bound the walk.
		int depth = 0;
		int parent = fragmentDef->parentIndex;
		while ( parent != B3_NULL_INDEX && depth <= fragmentCount )
		{
			depth += 1;
			parent = def->fragments[parent].parentIndex;
		}

		if ( depth > fragmentCount )
		{
			B3_ASSERT( false );
			failed = true;
			break;
		}

		depths[i] = depth;
		maxDepth = b3MaxInt( maxDepth, depth );

		if ( fragmentDef->isAnchored && depth == 0 )
		{
			fragment->flags |= b3_fragmentAnchored;
		}

		float mass = 0.0f;
		float volume = 0.0f;
		b3Vec3 center = b3Vec3_zero;
		b3AABB aabb = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };

		for ( int j = 0; j < fragmentDef->hullCount; ++j )
		{
			const b3HullData* source = fragmentDef->hulls[j];
			if ( source == NULL || source->version != B3_HULL_VERSION )
			{
				B3_ASSERT( false );
				failed = true;
				break;
			}

			b3HullData* hull = isIdentity ? b3CloneHull( source ) : b3CloneAndTransformHull( source, xf, b3Vec3_one );
			if ( hull == NULL )
			{
				failed = true;
				break;
			}

			hulls[hullIndex] = hull;
			hullFragments[hullIndex] = i;
			hullIndex += 1;

			b3MassData hullMass = b3ComputeHullMass( hull, material->density );
			mass += hullMass.mass;
			volume += hull->volume;
			center = b3MulAdd( center, hullMass.mass, hullMass.center );
			aabb = b3AABB_Union( aabb, hull->aabb );
			surfaceAreas[i] += hull->surfaceArea;
		}

		if ( failed )
		{
			break;
		}

		if ( mass > 0.0f )
		{
			center = b3MulSV( 1.0f / mass, center );
		}

		b3Matrix3 inertia = b3Mat3_zero;
		for ( int j = 0; j < fragmentDef->hullCount; ++j )
		{
			const b3HullData* hull = hulls[fragment->firstHull + j];
			b3MassData hullMass = b3ComputeHullMass( hull, material->density );
			b3Matrix3 shifted = b3AddMM( hullMass.inertia, b3Steiner( hullMass.mass, b3Sub( hullMass.center, center ) ) );
			inertia = b3AddMM( inertia, shifted );
		}

		if ( fragmentDef->massData != NULL )
		{
			B3_ASSERT( b3IsValidFloat( fragmentDef->massData->mass ) && fragmentDef->massData->mass > 0.0f );
			mass = fragmentDef->massData->mass;
			center = fragmentDef->massData->center;
			inertia = fragmentDef->massData->inertia;
		}

		float radiusSqr = 0.0f;
		for ( int j = 0; j < fragmentDef->hullCount; ++j )
		{
			const b3HullData* hull = hulls[fragment->firstHull + j];
			const b3Vec3* points = b3GetHullPoints( hull );
			for ( int k = 0; k < hull->vertexCount; ++k )
			{
				radiusSqr = b3MaxFloat( radiusSqr, b3DistanceSquared( points[k], center ) );
			}
		}

		fragment->mass = mass;
		fragment->center = center;
		fragment->inertia = inertia;
		fragment->volume = volume;
		fragment->aabb = aabb;
		fragment->radius = sqrtf( radiusSqr );
	}

	if ( failed )
	{
		goto cleanup;
	}

	// Child index array, grouped by parent
	for ( int i = 0; i < fragmentCount; ++i )
	{
		int parent = fragments[i].parent;
		if ( parent != B3_NULL_INDEX )
		{
			fragments[parent].childCount += 1;
			childTotal += 1;
		}
	}

	children = b3Alloc( b3MaxInt( 1, childTotal ) * sizeof( int ) );
	childCursors = b3Alloc( fragmentCount * sizeof( int ) );
	{
		int cursor = 0;
		for ( int i = 0; i < fragmentCount; ++i )
		{
			fragments[i].firstChild = cursor;
			childCursors[i] = cursor;
			cursor += fragments[i].childCount;
		}

		for ( int i = 0; i < fragmentCount; ++i )
		{
			int parent = fragments[i].parent;
			if ( parent != B3_NULL_INDEX )
			{
				children[childCursors[parent]++] = i;
			}
		}
	}

	// Children must add up to the mass of their parent so crushing a fragment conserves mass. Hulls of
	// children rarely add up exactly. Go level by level so a parent is final before its children.
	for ( int depth = 0; depth < maxDepth; ++depth )
	{
		for ( int i = 0; i < fragmentCount; ++i )
		{
			const b3StructureFragment* parent = fragments + i;
			if ( depths[i] != depth || parent->childCount == 0 )
			{
				continue;
			}

			float childMass = 0.0f;
			bool anyExplicit = false;
			for ( int j = 0; j < parent->childCount; ++j )
			{
				int child = children[parent->firstChild + j];
				childMass += fragments[child].mass;
				anyExplicit = anyExplicit || def->fragments[child].massData != NULL;
			}

			if ( anyExplicit || childMass <= 0.0f )
			{
				continue;
			}

			float scale = parent->mass / childMass;
			for ( int j = 0; j < parent->childCount; ++j )
			{
				b3StructureFragment* child = fragments + children[parent->firstChild + j];
				child->mass *= scale;
				child->inertia = b3MulSM( scale, child->inertia );
			}
		}
	}

	// Explicit bonds
	bondCapacity = b3MaxInt( 16, def->bondCount + 4 * fragmentCount );
	bonds = b3AllocZero( bondCapacity * sizeof( b3StructureBond ) );
	explicitKeys = b3Alloc( b3MaxInt( 1, def->bondCount ) * sizeof( uint64_t ) );

	for ( int i = 0; i < def->bondCount; ++i )
	{
		const b3BondDef* bondDef = def->bonds + i;
		bool validA = 0 <= bondDef->fragmentA && bondDef->fragmentA < fragmentCount && depths[bondDef->fragmentA] == 0;
		bool validB = bondDef->fragmentB == B3_NULL_INDEX ||
					  ( 0 <= bondDef->fragmentB && bondDef->fragmentB < fragmentCount && depths[bondDef->fragmentB] == 0 &&
						bondDef->fragmentB != bondDef->fragmentA );
		if ( validA == false || validB == false || bondDef->area <= 0.0f || b3LengthSquared( bondDef->normal ) == 0.0f )
		{
			// Bonds only join support fragments
			B3_ASSERT( false );
			continue;
		}

		b3StructureBond* bond = bonds + bondCount;
		bondCount += 1;

		// Keep the lower fragment first so the order is canonical
		if ( bondDef->fragmentB != B3_NULL_INDEX && bondDef->fragmentB < bondDef->fragmentA )
		{
			bond->fragmentA = bondDef->fragmentB;
			bond->fragmentB = bondDef->fragmentA;
			bond->normal = b3Neg( b3Normalize( bondDef->normal ) );
		}
		else
		{
			bond->fragmentA = bondDef->fragmentA;
			bond->fragmentB = bondDef->fragmentB;
			bond->normal = b3Normalize( bondDef->normal );
		}

		bond->centroid = bondDef->centroid;
		bond->area = bondDef->area;
		bond->flags = 0;

		if ( bondDef->fragmentB != B3_NULL_INDEX )
		{
			explicitKeys[explicitKeyCount++] = b3FragmentPairKey( bondDef->fragmentA, bondDef->fragmentB );
		}
	}

	qsort( explicitKeys, explicitKeyCount, sizeof( uint64_t ), b3CompareKeys );

	// Fragments fixed to the world get a rigid bond at their center
	for ( int i = 0; i < fragmentCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + i;
		if ( ( fragment->flags & b3_fragmentAnchored ) == 0 )
		{
			continue;
		}

		if ( bondCount == bondCapacity )
		{
			bonds = B3_GROW_ZERO( bonds, bondCapacity, 2 * bondCapacity );
			bondCapacity *= 2;
		}

		// A cross section through the fragment is a fair area for the stiffness of the anchor.
		float side = cbrtf( b3MaxFloat( fragment->volume, FLT_EPSILON ) );

		b3StructureBond* bond = bonds + bondCount;
		bondCount += 1;
		bond->fragmentA = i;
		bond->fragmentB = B3_NULL_INDEX;
		bond->centroid = fragment->center;
		bond->normal = b3Vec3_axisY;
		bond->area = side * side;
		bond->flags = b3_structureBondRigid;
	}

	// Find the surfaces shared by neighboring fragments
	{
		float gap = b3MaxFloat( 0.0f, def->bondGap );

		tree = b3DynamicTree_Create( hullCount );
		haveTree = true;
		for ( int i = 0; i < hullCount; ++i )
		{
			b3AABB aabb = b3AABB_Inflate( hulls[i]->aabb, gap );
			b3DynamicTree_CreateProxy( &tree, aabb, 1, (uint64_t)i );
		}

		queryContext.hullFragments = hullFragments;
		queryContext.fragments = fragments;
		queryContext.pairCapacity = 64;
		queryContext.pairs = b3Alloc( queryContext.pairCapacity * sizeof( b3HullPair ) );

		for ( int i = 0; i < fragmentCount; ++i )
		{
			const b3StructureFragment* fragment = fragments + i;
			bool isSupport = fragment->parent == B3_NULL_INDEX;

			if ( isSupport && def->generateBonds == false )
			{
				continue;
			}

			queryContext.fragmentA = i;
			queryContext.pairCount = 0;

			for ( int j = 0; j < fragment->hullCount; ++j )
			{
				int hullA = fragment->firstHull + j;
				queryContext.hullA = hullA;
				b3AABB aabb = b3AABB_Inflate( hulls[hullA]->aabb, gap );
				b3DynamicTree_Query( &tree, aabb, 1, false, b3PairQueryCallback, &queryContext );
			}

			// The tree reports pairs in an order that depends on its shape. Sort for a stable result.
			qsort( queryContext.pairs, queryContext.pairCount, sizeof( b3HullPair ), b3CompareHullPairs );

			int start = 0;
			while ( start < queryContext.pairCount )
			{
				int other = queryContext.pairs[start].otherFragment;
				int end = start;
				while ( end < queryContext.pairCount && queryContext.pairs[end].otherFragment == other )
				{
					end += 1;
				}

				b3SharedSurface surface = { 0 };
				for ( int k = start; k < end; ++k )
				{
					b3MatchHullFaces( hulls[queryContext.pairs[k].hullA], hulls[queryContext.pairs[k].hullB], gap, &surface );
				}

				if ( surface.area == 0.0f )
				{
					for ( int k = start; k < end; ++k )
					{
						b3MatchHullManifold( hulls[queryContext.pairs[k].hullA], hulls[queryContext.pairs[k].hullB], gap,
											 &surface );
					}
				}

				start = end;

				if ( surface.area < def->minBondArea || b3LengthSquared( surface.normal ) == 0.0f )
				{
					continue;
				}

				if ( isSupport == false )
				{
					// Siblings: this is surface the parent creates when it is crushed
					sharedAreas[fragment->parent] += surface.area;
					continue;
				}

				uint64_t key = b3FragmentPairKey( i, other );
				if ( explicitKeyCount > 0 &&
					 bsearch( &key, explicitKeys, explicitKeyCount, sizeof( uint64_t ), b3CompareKeys ) != NULL )
				{
					continue;
				}

				if ( bondCount == bondCapacity )
				{
					bonds = B3_GROW_ZERO( bonds, bondCapacity, 2 * bondCapacity );
					bondCapacity *= 2;
				}

				b3StructureBond* bond = bonds + bondCount;
				bondCount += 1;
				bond->fragmentA = i;
				bond->fragmentB = other;
				bond->centroid = b3MulSV( 1.0f / surface.area, surface.centroid );
				bond->normal = b3Normalize( surface.normal );
				bond->area = surface.area;
				bond->flags = 0;
			}
		}
	}

	qsort( bonds, bondCount, sizeof( b3StructureBond ), b3CompareBonds );

	// Health: the energy that crushes a fragment
	for ( int i = 0; i < fragmentCount; ++i )
	{
		b3StructureFragment* fragment = fragments + i;
		const b3StructureMaterial* material = materials + fragment->materialIndex;

		if ( def->fragments[i].health > 0.0f )
		{
			fragment->health = def->fragments[i].health;
		}
		else
		{
			// Crushing opens the surfaces between the children. A fragment without children, or with
			// children that don't touch, uses half of its own surface as a stand in.
			float area = sharedAreas[i] > 0.0f ? sharedAreas[i] : 0.5f * surfaceAreas[i];
			fragment->health = material->toughness * area;
		}
	}

	// Adjacency: the bonds of each fragment
	for ( int i = 0; i < bondCount; ++i )
	{
		fragments[bonds[i].fragmentA].bondCount += 1;
		adjacencyCount += 1;
		if ( bonds[i].fragmentB != B3_NULL_INDEX )
		{
			fragments[bonds[i].fragmentB].bondCount += 1;
			adjacencyCount += 1;
		}
	}

	adjacencyCursors = b3Alloc( fragmentCount * sizeof( int ) );
	{
		int cursor = 0;
		for ( int i = 0; i < fragmentCount; ++i )
		{
			fragments[i].firstBond = cursor;
			adjacencyCursors[i] = cursor;
			cursor += fragments[i].bondCount;
		}
	}

	// Lay out the block. Hulls are aligned because the collision code reads them with SIMD.
	{
		int offset = (int)sizeof( b3StructureData );
		offset = ( offset + 7 ) & ~7;

		int fragmentOffset = offset;
		offset += fragmentCount * (int)sizeof( b3StructureFragment );

		int bondOffset = offset;
		offset += bondCount * (int)sizeof( b3StructureBond );

		int adjacencyOffset = offset;
		offset += adjacencyCount * (int)sizeof( int32_t );

		int childOffset = offset;
		offset += childTotal * (int)sizeof( int32_t );

		int hullOffset = offset;
		offset += hullCount * (int)sizeof( int32_t );
		offset = ( offset + 7 ) & ~7;

		int materialOffset = offset;
		offset += materialCount * (int)sizeof( b3StructureMaterial );

		int64_t byteCount = offset;
		for ( int i = 0; i < hullCount; ++i )
		{
			byteCount = ( byteCount + ( B3_ALIGNMENT - 1 ) ) & ~(int64_t)( B3_ALIGNMENT - 1 );
			byteCount += hulls[i]->byteCount;
		}

		if ( byteCount > INT32_MAX )
		{
			// Offsets are 32-bit
			B3_ASSERT( false );
			goto cleanup;
		}

		// Zero everything so padding does not leak into the hash
		structure = b3AllocZero( (size_t)byteCount );
		structure->version = B3_STRUCTURE_VERSION;
		structure->hash = 0;
		structure->byteCount = (int32_t)byteCount;
		structure->fragmentCount = fragmentCount;
		structure->fragmentOffset = fragmentOffset;
		structure->bondCount = bondCount;
		structure->bondOffset = bondOffset;
		structure->adjacencyCount = adjacencyCount;
		structure->adjacencyOffset = adjacencyOffset;
		structure->childCount = childTotal;
		structure->childOffset = childOffset;
		structure->hullCount = hullCount;
		structure->hullOffset = hullOffset;
		structure->materialCount = materialCount;
		structure->materialOffset = materialOffset;

		char* base = (char*)structure;

		int32_t* adjacency = (int32_t*)( base + adjacencyOffset );
		for ( int i = 0; i < bondCount; ++i )
		{
			adjacency[adjacencyCursors[bonds[i].fragmentA]++] = i;
			if ( bonds[i].fragmentB != B3_NULL_INDEX )
			{
				adjacency[adjacencyCursors[bonds[i].fragmentB]++] = i;
			}
		}

		memcpy( base + fragmentOffset, fragments, fragmentCount * sizeof( b3StructureFragment ) );
		memcpy( base + bondOffset, bonds, bondCount * sizeof( b3StructureBond ) );
		memcpy( base + childOffset, children, childTotal * sizeof( int32_t ) );

		// Copy the materials field by field. The struct has padding that a block copy would carry over.
		b3StructureMaterial* blobMaterials = (b3StructureMaterial*)( base + materialOffset );
		for ( int i = 0; i < materialCount; ++i )
		{
			b3StructureMaterial* target = blobMaterials + i;
			const b3StructureMaterial* source = materials + i;
			B3_ASSERT( source->density > 0.0f && source->stiffness > 0.0f && source->creepTime > 0.0f );
			B3_ASSERT( source->tensileFatalLimit >= source->tensileElasticLimit && source->tensileElasticLimit > 0.0f );
			B3_ASSERT( source->compressiveFatalLimit >= source->compressiveElasticLimit && source->compressiveElasticLimit > 0.0f );
			B3_ASSERT( source->shearFatalLimit >= source->shearElasticLimit && source->shearElasticLimit > 0.0f );
			target->density = source->density;
			target->tensileElasticLimit = source->tensileElasticLimit;
			target->tensileFatalLimit = source->tensileFatalLimit;
			target->compressiveElasticLimit = source->compressiveElasticLimit;
			target->compressiveFatalLimit = source->compressiveFatalLimit;
			target->shearElasticLimit = source->shearElasticLimit;
			target->shearFatalLimit = source->shearFatalLimit;
			target->stiffness = source->stiffness;
			target->toughness = source->toughness;
			target->creepTime = source->creepTime;
			target->strengthVariation = b3ClampFloat( source->strengthVariation, 0.0f, 0.95f );
			target->userMaterialId = source->userMaterialId;
		}

		int32_t* hullOffsets = (int32_t*)( base + hullOffset );
		int64_t cursor = offset;
		for ( int i = 0; i < hullCount; ++i )
		{
			cursor = ( cursor + ( B3_ALIGNMENT - 1 ) ) & ~(int64_t)( B3_ALIGNMENT - 1 );
			hullOffsets[i] = (int32_t)cursor;
			memcpy( base + cursor, hulls[i], hulls[i]->byteCount );
			cursor += hulls[i]->byteCount;
		}

		b3AABB aabb = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
		float mass = 0.0f;
		int supportCount = 0;
		for ( int i = 0; i < fragmentCount; ++i )
		{
			if ( fragments[i].parent == B3_NULL_INDEX )
			{
				aabb = b3AABB_Union( aabb, fragments[i].aabb );
				mass += fragments[i].mass;
				supportCount += 1;
			}
		}

		structure->aabb = aabb;
		structure->mass = mass;
		structure->supportCount = supportCount;
		structure->hash = b3Hash64NonZero( (const uint8_t*)structure, structure->byteCount );
	}

cleanup:
	if ( haveTree )
	{
		b3DynamicTree_Destroy( &tree );
	}

	if ( queryContext.pairs != NULL )
	{
		b3Free( queryContext.pairs, queryContext.pairCapacity * sizeof( b3HullPair ) );
	}

	if ( adjacencyCursors != NULL )
	{
		b3Free( adjacencyCursors, fragmentCount * sizeof( int ) );
	}

	if ( childCursors != NULL )
	{
		b3Free( childCursors, fragmentCount * sizeof( int ) );
	}

	if ( children != NULL )
	{
		b3Free( children, b3MaxInt( 1, childTotal ) * sizeof( int ) );
	}

	if ( explicitKeys != NULL )
	{
		b3Free( explicitKeys, b3MaxInt( 1, def->bondCount ) * sizeof( uint64_t ) );
	}

	if ( bonds != NULL )
	{
		b3Free( bonds, bondCapacity * sizeof( b3StructureBond ) );
	}

	for ( int i = 0; i < hullCount; ++i )
	{
		if ( hulls[i] != NULL )
		{
			b3DestroyHull( hulls[i] );
		}
	}

	b3Free( surfaceAreas, fragmentCount * sizeof( float ) );
	b3Free( sharedAreas, fragmentCount * sizeof( float ) );
	b3Free( depths, fragmentCount * sizeof( int ) );
	b3Free( hullFragments, hullCount * sizeof( int ) );
	b3Free( hulls, hullCount * sizeof( b3HullData* ) );
	b3Free( fragments, fragmentCount * sizeof( b3StructureFragment ) );

	return structure;
}

void b3DestroyStructure( b3StructureData* structure )
{
	if ( structure == NULL )
	{
		return;
	}

	B3_ASSERT( structure->version == B3_STRUCTURE_VERSION );
	b3Free( structure, structure->byteCount );
}

const b3StructureData* b3ValidateStructure( const uint8_t* bytes, int byteCount )
{
	if ( bytes == NULL || byteCount < (int)sizeof( b3StructureData ) )
	{
		return NULL;
	}

	// Hulls inside the block need the same alignment the allocator gives
	if ( ( (uintptr_t)bytes & ( B3_ALIGNMENT - 1 ) ) != 0 )
	{
		return NULL;
	}

	const b3StructureData* structure = (const b3StructureData*)bytes;
	if ( structure->version != B3_STRUCTURE_VERSION || structure->byteCount != byteCount )
	{
		return NULL;
	}

	if ( structure->fragmentCount <= 0 || structure->bondCount < 0 || structure->hullCount <= 0 || structure->materialCount <= 0 )
	{
		return NULL;
	}

	// Every array must lie inside the block
	int64_t size = byteCount;
	bool inside = true;
	inside = inside && structure->fragmentOffset + (int64_t)structure->fragmentCount * (int64_t)sizeof( b3StructureFragment ) <= size;
	inside = inside && structure->bondOffset + (int64_t)structure->bondCount * (int64_t)sizeof( b3StructureBond ) <= size;
	inside = inside && structure->adjacencyOffset + (int64_t)structure->adjacencyCount * (int64_t)sizeof( int32_t ) <= size;
	inside = inside && structure->childOffset + (int64_t)structure->childCount * (int64_t)sizeof( int32_t ) <= size;
	inside = inside && structure->hullOffset + (int64_t)structure->hullCount * (int64_t)sizeof( int32_t ) <= size;
	inside = inside && structure->materialOffset + (int64_t)structure->materialCount * (int64_t)sizeof( b3StructureMaterial ) <= size;
	if ( inside == false )
	{
		return NULL;
	}

	const int32_t* hullOffsets = (const int32_t*)( bytes + structure->hullOffset );
	for ( int i = 0; i < structure->hullCount; ++i )
	{
		int64_t offset = hullOffsets[i];
		if ( offset < 0 || offset + (int64_t)sizeof( b3HullData ) > size )
		{
			return NULL;
		}

		const b3HullData* hull = (const b3HullData*)( bytes + offset );
		if ( hull->version != B3_HULL_VERSION || offset + hull->byteCount > size )
		{
			return NULL;
		}
	}

	const b3StructureFragment* fragments = b3GetStructureFragments( structure );
	for ( int i = 0; i < structure->fragmentCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + i;
		bool valid = fragment->firstHull >= 0 && fragment->hullCount > 0 &&
					 fragment->firstHull + fragment->hullCount <= structure->hullCount && fragment->firstBond >= 0 &&
					 fragment->bondCount >= 0 && fragment->firstBond + fragment->bondCount <= structure->adjacencyCount &&
					 fragment->firstChild >= 0 && fragment->childCount >= 0 &&
					 fragment->firstChild + fragment->childCount <= structure->childCount &&
					 fragment->parent >= B3_NULL_INDEX && fragment->parent < structure->fragmentCount &&
					 fragment->materialIndex >= 0 && fragment->materialIndex < structure->materialCount;
		if ( valid == false )
		{
			return NULL;
		}
	}

	const b3StructureBond* bonds = b3GetStructureBonds( structure );
	for ( int i = 0; i < structure->bondCount; ++i )
	{
		const b3StructureBond* bond = bonds + i;
		bool valid = bond->fragmentA >= 0 && bond->fragmentA < structure->fragmentCount && bond->fragmentB >= B3_NULL_INDEX &&
					 bond->fragmentB < structure->fragmentCount && bond->area > 0.0f;
		if ( valid == false )
		{
			return NULL;
		}
	}

	return structure;
}
