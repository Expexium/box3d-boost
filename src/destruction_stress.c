// SPDX-FileCopyrightText: 2026 Box3D contributors
// SPDX-License-Identifier: MIT

// Stress solver
//
// A piece is modeled as rigid fragments joined by stiff springs at the bonds. This is the rigid body
// spring model. Each fragment has a small translation u and rotation t. A bond between fragments a
// and b at point p stretches by
//
//   d = (u_b + t_b x r_b) - (u_a + t_a x r_a)      r = p - fragment center
//   w = t_b - t_a
//
// and answers with the force F = kl * d and the moment M = ka * w. Summing over the bonds gives the
// linear system K x = f, where x holds the fragment displacements and f the loads. K is symmetric
// positive semi-definite and is never assembled: one loop over the bonds applies it.
//
// Why displacements instead of bond impulses (as in NVIDIA Blast)? Both give the same bond forces
// when the bond stiffness is the same. But a structure with several load paths has many sets of
// bond forces that balance the loads, so an impulse solve must be cold started whenever a bond
// breaks or it keeps stale self-balancing forces forever. The displacement of an anchored piece is
// unique, so the previous solution stays a valid starting point while the piece falls apart, which
// is exactly when the solve is needed most. It also has three times fewer unknowns.
//
// The system is solved with conjugate gradients, preconditioned by the exact inverse of each
// fragment's own 6x6 stiffness block. A piece gets a budget of iterations per step and continues in
// the next step, so a large structure converges over a few frames instead of stalling one.
//
// A piece that is not anchored has no unique displacement: it can move as a whole. Its loads are
// first balanced with the inertia forces of the rigid motion they cause (inertia relief). A piece in
// free fall then has no load at all, which is correct: gravity alone does not stress a falling body.
//
// The bond force and moment are converted into stress with beam formulas for a square patch of the
// bond area. Stress between the elastic and the fatal limit shrinks the effective area of the bond.
// That raises the stress of the next step, so an overloaded bond runs away and breaks.
//
// Impacts are solved apart from the sustained loads. Only the fragments near the hit are free to
// move and the rest of the piece is held fixed. A short blow cannot move the far side of a building
// in the time it lasts, so the damage stays near the hit instead of snapping a distant base.

#include "arena_allocator.h"
#include "body.h"
#include "core.h"
#include "destruction.h"
#include "physics_world.h"

#include "box3d/destruction.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// A bond with less than this share of its area left is treated as broken.
#define B3_MIN_BOND_AREA_FRACTION 0.05f

// The most iterations for the local impact solve
#define B3_IMPACT_ITERATIONS 48

// Bonds this close to the worst overload of their piece break in the same round
#define B3_BREAK_BAND 0.85f

// The loads of a free piece are solved again once they differ this much from the ones in the solver
#define B3_LOAD_DRIFT 0.05

// The most steps the solver works on one set of loads before it takes fresh ones regardless
#define B3_MAX_LOAD_STEPS 120

// A solve with a larger relative error than this breaks nothing, however overloaded a bond looks
#define B3_MAX_ROUGH_ERROR 0.5

static inline b3DVec3 b3DMakeVec3( b3Vec3 v )
{
	return (b3DVec3){ v.x, v.y, v.z };
}

static inline b3DVec3 b3DAdd( b3DVec3 a, b3DVec3 b )
{
	return (b3DVec3){ a.x + b.x, a.y + b.y, a.z + b.z };
}

static inline b3DVec3 b3DSub( b3DVec3 a, b3DVec3 b )
{
	return (b3DVec3){ a.x - b.x, a.y - b.y, a.z - b.z };
}

static inline b3DVec3 b3DScale( double s, b3DVec3 a )
{
	return (b3DVec3){ s * a.x, s * a.y, s * a.z };
}

static inline b3DVec3 b3DCross( b3DVec3 a, b3DVec3 b )
{
	return (b3DVec3){ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

static inline double b3DDot( b3DVec3 a, b3DVec3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline double b3DDot6( const b3DVec6* a, const b3DVec6* b, int count )
{
	double sum = 0.0;
	for ( int i = 0; i < count; ++i )
	{
		sum += b3DDot( a[i].lin, b[i].lin ) + b3DDot( a[i].ang, b[i].ang );
	}
	return sum;
}

b3StressSystem* b3CreateStressSystem( int nodeCount, int bondCapacity )
{
	// One extra node at the end for the ground
	int vectorCount = nodeCount + 1;

	size_t size = sizeof( b3StressSystem );
	size_t bondOffset = size;
	size += (size_t)bondCapacity * sizeof( b3SolverBond );
	size_t bondIdOffset = size;
	size += (size_t)bondCapacity * sizeof( int );
	size = ( size + 15 ) & ~(size_t)15;
	size_t vectorOffset = size;
	size += 6 * (size_t)vectorCount * sizeof( b3DVec6 );
	size_t precondOffset = size;
	size += (size_t)vectorCount * sizeof( b3NodePrecond );

	char* memory = b3AllocZero( size );
	b3StressSystem* system = (b3StressSystem*)memory;
	system->nodeCount = nodeCount;
	system->bondCount = 0;
	system->bondCapacity = bondCapacity;
	system->bonds = (b3SolverBond*)( memory + bondOffset );
	system->bondIds = (int*)( memory + bondIdOffset );

	b3DVec6* vectors = (b3DVec6*)( memory + vectorOffset );
	system->x = vectors + 0 * vectorCount;
	system->r = vectors + 1 * vectorCount;
	system->z = vectors + 2 * vectorCount;
	system->p = vectors + 3 * vectorCount;
	system->q = vectors + 4 * vectorCount;
	system->f = vectors + 5 * vectorCount;
	system->precond = (b3NodePrecond*)( memory + precondOffset );
	system->byteCount = (int)size;
	return system;
}

static void b3DestroyStressLevels( b3StressSystem* system )
{
	for ( int i = 0; i < system->levelCount; ++i )
	{
		b3Free( system->levels[i], system->levels[i]->byteCount );
		system->levels[i] = NULL;
	}

	system->levelCount = 0;
	system->levelsValid = false;
}

void b3DestroyStressSystem( b3StressSystem* system )
{
	if ( system != NULL )
	{
		b3DestroyStressLevels( system );
		b3Free( system, system->byteCount );
	}
}

int b3GetStressSystemByteCount( const b3StressSystem* system )
{
	int byteCount = system->byteCount;
	for ( int i = 0; i < system->levelCount; ++i )
	{
		byteCount += system->levels[i]->byteCount;
	}

	return byteCount;
}

b3BondLimits b3GetBondLimits( const b3Destructible* destructible, const b3DestructionBond* bond )
{
	const b3StructureMaterial* material = b3GetStructureMaterials( destructible->structure ) + bond->materialIndex;
	float scale = destructible->strengthScale * bond->strength;

	b3BondLimits limits;
	limits.tensileElastic = scale * material->tensileElasticLimit;
	limits.tensileFatal = scale * material->tensileFatalLimit;
	limits.compressiveElastic = scale * material->compressiveElasticLimit;
	limits.compressiveFatal = scale * material->compressiveFatalLimit;
	limits.shearElastic = scale * material->shearElasticLimit;
	limits.shearFatal = scale * material->shearFatalLimit;
	return limits;
}

void b3ComputePieceMass( const b3Destructible* destructible, b3DestructionPiece* piece )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );

	double mass = 0.0;
	double cx = 0.0, cy = 0.0, cz = 0.0;
	for ( int i = 0; i < piece->nodeCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + piece->nodes[i];
		mass += fragment->mass;
		cx += (double)fragment->mass * fragment->center.x;
		cy += (double)fragment->mass * fragment->center.y;
		cz += (double)fragment->mass * fragment->center.z;
	}

	b3Vec3 center = b3Vec3_zero;
	if ( mass > 0.0 )
	{
		center = (b3Vec3){ (float)( cx / mass ), (float)( cy / mass ), (float)( cz / mass ) };
	}

	b3Matrix3 inertia = b3Mat3_zero;
	float radius = 0.0f;
	for ( int i = 0; i < piece->nodeCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + piece->nodes[i];
		b3Vec3 offset = b3Sub( fragment->center, center );
		inertia = b3AddMM( inertia, b3AddMM( fragment->inertia, b3Steiner( fragment->mass, offset ) ) );
		radius = b3MaxFloat( radius, b3Length( offset ) + fragment->radius );
	}

	piece->mass = (float)mass;
	piece->center = center;
	piece->inertia = inertia;
	piece->invInertia = b3Det( inertia ) > 0.0f ? b3InvertMatrix( inertia ) : b3Mat3_zero;
	piece->radius = radius;
}

// Build the solver arrays of a piece from its fragment list.
static void b3BuildStressSystem( b3Destructible* destructible, b3DestructionPiece* piece )
{
	const b3StructureData* structure = destructible->structure;
	const b3StructureMaterial* materials = b3GetStructureMaterials( structure );
	const int* adjacencyStarts = destructible->adjacencyStarts;
	const int* adjacency = destructible->adjacency;
	b3DestructionBond* bonds = destructible->bonds;

	int nodeCount = piece->nodeCount;

	// Every intact bond joins two fragments of the same piece, so count each bond at its first fragment.
	int bondCount = 0;
	for ( int i = 0; i < nodeCount; ++i )
	{
		int node = piece->nodes[i];
		for ( int j = adjacencyStarts[node]; j < adjacencyStarts[node + 1]; ++j )
		{
			const b3DestructionBond* bond = bonds + adjacency[j];
			if ( ( bond->flags & b3_bondBroken ) == 0 && bond->nodeA == node )
			{
				bondCount += 1;
			}
		}
	}

	if ( piece->system != NULL && ( piece->system->nodeCount != nodeCount || piece->system->bondCapacity < bondCount ) )
	{
		b3DestroyStressSystem( piece->system );
		piece->system = NULL;
	}

	piece->bondCount = bondCount;
	piece->flags &= ~b3_pieceSystemDirty;

	if ( bondCount == 0 )
	{
		b3DestroyStressSystem( piece->system );
		piece->system = NULL;
		return;
	}

	if ( piece->system == NULL )
	{
		piece->system = b3CreateStressSystem( nodeCount, bondCount );
	}

	b3StressSystem* system = piece->system;
	int groundNode = nodeCount;
	int count = 0;

	for ( int i = 0; i < nodeCount; ++i )
	{
		int node = piece->nodes[i];
		B3_ASSERT( destructible->fragmentLocal[node] == i );

		for ( int j = adjacencyStarts[node]; j < adjacencyStarts[node + 1]; ++j )
		{
			int bondIndex = adjacency[j];
			b3DestructionBond* bond = bonds + bondIndex;
			if ( ( bond->flags & b3_bondBroken ) != 0 || bond->nodeA != node )
			{
				continue;
			}

			b3SolverBond* solverBond = system->bonds + count;
			solverBond->a = i;
			solverBond->b = bond->nodeB == B3_NULL_INDEX ? groundNode : destructible->fragmentLocal[bond->nodeB];
			solverBond->rA = bond->rA;
			solverBond->rB = bond->rB;
			b3GetBondStiffness( materials + bond->materialIndex, bond, &solverBond->kl, &solverBond->ka );

			system->bondIds[count] = bondIndex;
			bond->solverIndex = count;
			count += 1;
		}
	}

	B3_ASSERT( count == bondCount );
	system->bondCount = count;

	// Start from the latest solution of these fragments. Losing a few bonds changes it only nearby.
	for ( int i = 0; i < nodeCount; ++i )
	{
		system->x[i] = destructible->displacements[piece->nodes[i]];
	}

	system->x[groundNode] = (b3DVec6){ 0 };
	system->f[groundNode] = (b3DVec6){ 0 };
	system->cgValid = false;
	system->precondValid = false;
	system->levelsValid = false;
	system->loadsValid = false;
	system->evaluated = false;

	piece->flags |= b3_pieceLoadDirty;
}

// y = K x
static void b3StressMultiply( const b3SolverBond* bonds, int bondCount, int nodeCount, const b3DVec6* x, b3DVec6* y )
{
	memset( y, 0, ( nodeCount + 1 ) * sizeof( b3DVec6 ) );

	for ( int i = 0; i < bondCount; ++i )
	{
		const b3SolverBond* bond = bonds + i;
		const b3DVec6* xa = x + bond->a;
		const b3DVec6* xb = x + bond->b;

		double rax = bond->rA.x, ray = bond->rA.y, raz = bond->rA.z;
		double rbx = bond->rB.x, rby = bond->rB.y, rbz = bond->rB.z;

		// Stretch of the bond: displacement of the bond point on b minus on a
		double dx = ( xb->lin.x + xb->ang.y * rbz - xb->ang.z * rby ) - ( xa->lin.x + xa->ang.y * raz - xa->ang.z * ray );
		double dy = ( xb->lin.y + xb->ang.z * rbx - xb->ang.x * rbz ) - ( xa->lin.y + xa->ang.z * rax - xa->ang.x * raz );
		double dz = ( xb->lin.z + xb->ang.x * rby - xb->ang.y * rbx ) - ( xa->lin.z + xa->ang.x * ray - xa->ang.y * rax );

		double kl = bond->kl, ka = bond->ka;
		double fx = kl * dx, fy = kl * dy, fz = kl * dz;
		double mx = ka * ( xb->ang.x - xa->ang.x );
		double my = ka * ( xb->ang.y - xa->ang.y );
		double mz = ka * ( xb->ang.z - xa->ang.z );

		b3DVec6* ya = y + bond->a;
		b3DVec6* yb = y + bond->b;

		ya->lin.x -= fx;
		ya->lin.y -= fy;
		ya->lin.z -= fz;
		ya->ang.x -= ( ray * fz - raz * fy ) + mx;
		ya->ang.y -= ( raz * fx - rax * fz ) + my;
		ya->ang.z -= ( rax * fy - ray * fx ) + mz;

		yb->lin.x += fx;
		yb->lin.y += fy;
		yb->lin.z += fz;
		yb->ang.x += ( rby * fz - rbz * fy ) + mx;
		yb->ang.y += ( rbz * fx - rbx * fz ) + my;
		yb->ang.z += ( rbx * fy - rby * fx ) + mz;
	}

	// The ground does not move
	y[nodeCount] = (b3DVec6){ 0 };
}

// Force and moment a bond carries for the given displacements, in the bond convention: positive along
// the normal pulls the fragments apart.
static inline void b3GetBondWrench( const b3SolverBond* bond, const b3DVec6* x, b3DVec3* force, b3DVec3* moment )
{
	const b3DVec6* xa = x + bond->a;
	const b3DVec6* xb = x + bond->b;
	b3DVec3 ra = b3DMakeVec3( bond->rA );
	b3DVec3 rb = b3DMakeVec3( bond->rB );
	b3DVec3 da = b3DAdd( xa->lin, b3DCross( xa->ang, ra ) );
	b3DVec3 db = b3DAdd( xb->lin, b3DCross( xb->ang, rb ) );
	*force = b3DScale( bond->kl, b3DSub( db, da ) );
	*moment = b3DScale( bond->ka, b3DSub( xb->ang, xa->ang ) );
}

// Build the inverse of each fragment's own 6x6 stiffness block.
//
// The block of a fragment is [k 1, -[w]x; [w]x, T] with k the summed linear stiffness of its bonds,
// w the summed stiffness times lever arm, and T the summed rotational stiffness. It is inverted with
// the Schur complement S = T - (|w|^2 1 - w w^T) / k of the rotation block.
static void b3BuildPrecond( const b3SolverBond* bonds, int bondCount, int nodeCount, b3NodePrecond* precond, b3Arena arena )
{
	// Rotational block T per node, symmetric: xx, xy, xz, yy, yz, zz
	double* rotation = b3Bump( &arena, ( nodeCount + 1 ) * 6 * (int)sizeof( double ) );
	memset( rotation, 0, ( nodeCount + 1 ) * 6 * sizeof( double ) );
	memset( precond, 0, ( nodeCount + 1 ) * sizeof( b3NodePrecond ) );

	for ( int i = 0; i < bondCount; ++i )
	{
		const b3SolverBond* bond = bonds + i;
		double kl = bond->kl, ka = bond->ka;

		for ( int side = 0; side < 2; ++side )
		{
			int node = side == 0 ? bond->a : bond->b;
			b3Vec3 r = side == 0 ? bond->rA : bond->rB;
			double rx = r.x, ry = r.y, rz = r.z;
			double rr = rx * rx + ry * ry + rz * rz;

			b3NodePrecond* np = precond + node;
			np->invK += kl;
			np->w.x += kl * rx;
			np->w.y += kl * ry;
			np->w.z += kl * rz;

			double* t = rotation + 6 * node;
			t[0] += kl * ( rr - rx * rx ) + ka;
			t[1] += -kl * rx * ry;
			t[2] += -kl * rx * rz;
			t[3] += kl * ( rr - ry * ry ) + ka;
			t[4] += -kl * ry * rz;
			t[5] += kl * ( rr - rz * rz ) + ka;
		}
	}

	for ( int i = 0; i < nodeCount; ++i )
	{
		b3NodePrecond* np = precond + i;
		double k = np->invK;
		if ( k <= 0.0 )
		{
			// A fragment without bonds. Cannot happen in a connected piece but stay safe.
			*np = (b3NodePrecond){ 0 };
			continue;
		}

		double invK = 1.0 / k;
		double wx = np->w.x, wy = np->w.y, wz = np->w.z;
		double ww = wx * wx + wy * wy + wz * wz;

		const double* t = rotation + 6 * i;
		double sxx = t[0] - invK * ( ww - wx * wx );
		double sxy = t[1] + invK * wx * wy;
		double sxz = t[2] + invK * wx * wz;
		double syy = t[3] - invK * ( ww - wy * wy );
		double syz = t[4] + invK * wy * wz;
		double szz = t[5] - invK * ( ww - wz * wz );

		// Inverse of the symmetric 3x3 by cofactors
		double cxx = syy * szz - syz * syz;
		double cxy = sxz * syz - sxy * szz;
		double cxz = sxy * syz - sxz * syy;
		double det = sxx * cxx + sxy * cxy + sxz * cxz;

		np->invK = invK;
		if ( det > 0.0 )
		{
			double invDet = 1.0 / det;
			np->invS[0] = invDet * cxx;
			np->invS[1] = invDet * cxy;
			np->invS[2] = invDet * cxz;
			np->invS[3] = invDet * ( sxx * szz - sxz * sxz );
			np->invS[4] = invDet * ( sxy * sxz - sxx * syz );
			np->invS[5] = invDet * ( sxx * syy - sxy * sxy );
		}
		else
		{
			// Degenerate rotation block: fall back to the diagonal
			np->invS[0] = t[0] > 0.0 ? 1.0 / t[0] : 0.0;
			np->invS[3] = t[3] > 0.0 ? 1.0 / t[3] : 0.0;
			np->invS[5] = t[5] > 0.0 ? 1.0 / t[5] : 0.0;
			np->invS[1] = np->invS[2] = np->invS[4] = 0.0;
			np->w = (b3DVec3){ 0.0, 0.0, 0.0 };
		}
	}

	// The ground node stays at zero
	precond[nodeCount] = (b3NodePrecond){ 0 };
}

// z = inverse(block) * r for one node
static inline b3DVec6 b3ApplyPrecond( const b3NodePrecond* np, const b3DVec6* r )
{
	// Rotation first: t = inverse(S) * (r_ang - w x r_lin / k)
	b3DVec3 c = b3DCross( np->w, r->lin );
	double bx = r->ang.x - np->invK * c.x;
	double by = r->ang.y - np->invK * c.y;
	double bz = r->ang.z - np->invK * c.z;

	b3DVec6 z;
	z.ang.x = np->invS[0] * bx + np->invS[1] * by + np->invS[2] * bz;
	z.ang.y = np->invS[1] * bx + np->invS[3] * by + np->invS[4] * bz;
	z.ang.z = np->invS[2] * bx + np->invS[4] * by + np->invS[5] * bz;

	// Then translation: u = (r_lin + w x t) / k
	b3DVec3 d = b3DCross( np->w, z.ang );
	z.lin.x = np->invK * ( r->lin.x + d.x );
	z.lin.y = np->invK * ( r->lin.y + d.y );
	z.lin.z = np->invK * ( r->lin.z + d.z );
	return z;
}

// Pieces with fewer fragments than this converge quickly without coarse levels
#define B3_MIN_LEVEL_NODES 48

// A level must shrink the system at least this much to be worth its cost
#define B3_MAX_LEVEL_RATIO 0.75f

// A neighbor counts as strongly tied when its bonds are at least this share of the strongest tie of
// the node. Clusters only grow along strong ties.
#define B3_STRONG_TIE 0.25f

// Lump the nodes of a bond system into clusters of neighbors. Returns the number of clusters and
// writes the cluster of each node.
static int b3ClusterNodes( const b3SolverBond* bonds, int bondCount, int nodeCount, int* parents, b3Arena arena )
{
	// Neighbor lists. Bonds to the ground have no neighbor.
	int* starts = b3Bump( &arena, ( nodeCount + 1 ) * (int)sizeof( int ) );
	memset( starts, 0, ( nodeCount + 1 ) * sizeof( int ) );

	for ( int i = 0; i < bondCount; ++i )
	{
		if ( bonds[i].b < nodeCount )
		{
			starts[bonds[i].a + 1] += 1;
			starts[bonds[i].b + 1] += 1;
		}
	}

	for ( int i = 0; i < nodeCount; ++i )
	{
		starts[i + 1] += starts[i];
	}

	int neighborCount = starts[nodeCount];
	int* ends = b3Bump( &arena, nodeCount * (int)sizeof( int ) );
	int* neighbors = b3Bump( &arena, neighborCount * (int)sizeof( int ) );
	float* weights = b3Bump( &arena, neighborCount * (int)sizeof( float ) );
	memcpy( ends, starts, nodeCount * sizeof( int ) );

	for ( int i = 0; i < bondCount; ++i )
	{
		const b3SolverBond* bond = bonds + i;
		if ( bond->b < nodeCount )
		{
			neighbors[ends[bond->a]] = bond->b;
			weights[ends[bond->a]++] = bond->kl;
			neighbors[ends[bond->b]] = bond->a;
			weights[ends[bond->b]++] = bond->kl;
		}
	}

	// Two nodes can share several bonds, on the coarse levels most do. Merge them into one tie with
	// the summed stiffness, and find the strongest tie of each node.
	int* slots = b3Bump( &arena, nodeCount * (int)sizeof( int ) );
	float* strongest = b3Bump( &arena, nodeCount * (int)sizeof( float ) );
	memset( slots, 0xFF, nodeCount * sizeof( int ) );

	for ( int i = 0; i < nodeCount; ++i )
	{
		int begin = starts[i];
		int end = begin;
		for ( int j = begin; j < ends[i]; ++j )
		{
			int neighbor = neighbors[j];
			int slot = slots[neighbor];
			if ( slot >= begin && slot < end && neighbors[slot] == neighbor )
			{
				weights[slot] += weights[j];
			}
			else
			{
				slots[neighbor] = end;
				neighbors[end] = neighbor;
				weights[end] = weights[j];
				end += 1;
			}
		}

		ends[i] = end;

		float maxWeight = 0.0f;
		for ( int j = begin; j < end; ++j )
		{
			maxWeight = b3MaxFloat( maxWeight, weights[j] );
		}

		strongest[i] = maxWeight;
	}

	memset( parents, 0xFF, nodeCount * sizeof( int ) );
	int clusterCount = 0;

	// A node whose strong neighbors are all still free starts a cluster and takes them in. A part
	// that hangs on the rest by weak bonds then gets clusters of its own, so the coarse levels can
	// move it against the rest. That motion is the slowest one for the solver to find by itself.
	for ( int i = 0; i < nodeCount; ++i )
	{
		if ( parents[i] != B3_NULL_INDEX )
		{
			continue;
		}

		float threshold = B3_STRONG_TIE * strongest[i];
		bool isFree = true;
		for ( int j = starts[i]; j < ends[i]; ++j )
		{
			if ( weights[j] >= threshold && parents[neighbors[j]] != B3_NULL_INDEX )
			{
				isFree = false;
				break;
			}
		}

		if ( isFree == false )
		{
			continue;
		}

		parents[i] = clusterCount;
		for ( int j = starts[i]; j < ends[i]; ++j )
		{
			if ( weights[j] >= threshold )
			{
				parents[neighbors[j]] = clusterCount;
			}
		}

		clusterCount += 1;
	}

	// The nodes left over join the cluster they are tied to the hardest. These choices are stored
	// as negative values until the end so that they only look at the clusters made above.
	for ( int i = 0; i < nodeCount; ++i )
	{
		if ( parents[i] != B3_NULL_INDEX )
		{
			continue;
		}

		int best = B3_NULL_INDEX;
		float bestWeight = -1.0f;
		for ( int j = starts[i]; j < ends[i]; ++j )
		{
			int cluster = parents[neighbors[j]];
			if ( cluster >= 0 && weights[j] > bestWeight )
			{
				best = cluster;
				bestWeight = weights[j];
			}
		}

		if ( best == B3_NULL_INDEX )
		{
			best = clusterCount++;
		}

		parents[i] = -2 - best;
	}

	for ( int i = 0; i < nodeCount; ++i )
	{
		if ( parents[i] < 0 )
		{
			parents[i] = -2 - parents[i];
		}
	}

	return clusterCount;
}

// Build the coarse levels of a stress system from its bonds.
static void b3BuildStressLevels( const b3Destructible* destructible, const b3DestructionPiece* piece, b3StressSystem* system, b3Arena arena )
{
	b3DestroyStressLevels( system );
	system->levelsValid = true;

	if ( system->nodeCount < B3_MIN_LEVEL_NODES )
	{
		return;
	}

	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );

	// The finest level is the system itself
	const b3SolverBond* bonds = system->bonds;
	int bondCount = system->bondCount;
	int nodeCount = system->nodeCount;
	b3Vec3* finePositions = b3Bump( &arena, nodeCount * (int)sizeof( b3Vec3 ) );
	for ( int i = 0; i < nodeCount; ++i )
	{
		finePositions[i] = fragments[piece->nodes[i]].center;
	}

	const b3Vec3* positions = finePositions;
	int* parents = b3Bump( &arena, nodeCount * (int)sizeof( int ) );

	while ( system->levelCount < B3_MAX_STRESS_LEVELS && nodeCount > 1 )
	{
		int clusterCount = b3ClusterNodes( bonds, bondCount, nodeCount, parents, arena );
		if ( (float)clusterCount > B3_MAX_LEVEL_RATIO * (float)nodeCount )
		{
			break;
		}

		int groundNode = nodeCount;
		int coarseBondCount = 0;
		for ( int i = 0; i < bondCount; ++i )
		{
			const b3SolverBond* bond = bonds + i;
			if ( bond->b == groundNode || parents[bond->a] != parents[bond->b] )
			{
				coarseBondCount += 1;
			}
		}

		if ( coarseBondCount == 0 )
		{
			// A free piece lumped into one cluster. Its rigid motion carries no stress.
			break;
		}

		// One block for the level and its arrays
		int vectorCount = clusterCount + 1;
		size_t size = sizeof( b3StressLevel );
		size_t bondOffset = size;
		size += (size_t)coarseBondCount * sizeof( b3SolverBond );
		size_t sourceOffset = size;
		size += (size_t)coarseBondCount * sizeof( int );
		size_t parentOffset = size;
		size += (size_t)nodeCount * sizeof( int );
		size_t offsetOffset = size;
		size += (size_t)nodeCount * sizeof( b3Vec3 );
		size_t positionOffset = size;
		size += (size_t)clusterCount * sizeof( b3Vec3 );
		size = ( size + 15 ) & ~(size_t)15;
		size_t vectorOffset = size;
		size += 3 * (size_t)vectorCount * sizeof( b3DVec6 );
		size_t precondOffset = size;
		size += (size_t)vectorCount * sizeof( b3NodePrecond );

		char* memory = b3AllocZero( size );
		b3StressLevel* level = (b3StressLevel*)memory;
		level->nodeCount = clusterCount;
		level->bondCount = coarseBondCount;
		level->bonds = (b3SolverBond*)( memory + bondOffset );
		level->sources = (int*)( memory + sourceOffset );
		level->parents = (int*)( memory + parentOffset );
		level->offsets = (b3Vec3*)( memory + offsetOffset );
		level->positions = (b3Vec3*)( memory + positionOffset );
		level->r = (b3DVec6*)( memory + vectorOffset );
		level->z = level->r + vectorCount;
		level->t = level->z + vectorCount;
		level->precond = (b3NodePrecond*)( memory + precondOffset );
		level->byteCount = (int)size;

		// A cluster sits at the average position of its nodes. Any point would do, this one keeps
		// the lever arms short.
		double* sums = b3Bump( &arena, 4 * clusterCount * (int)sizeof( double ) );
		memset( sums, 0, 4 * clusterCount * sizeof( double ) );
		for ( int i = 0; i < nodeCount; ++i )
		{
			double* sum = sums + 4 * parents[i];
			sum[0] += positions[i].x;
			sum[1] += positions[i].y;
			sum[2] += positions[i].z;
			sum[3] += 1.0;
		}

		for ( int i = 0; i < clusterCount; ++i )
		{
			const double* sum = sums + 4 * i;
			double inverse = 1.0 / sum[3];
			level->positions[i] = (b3Vec3){ (float)( inverse * sum[0] ), (float)( inverse * sum[1] ), (float)( inverse * sum[2] ) };
		}

		for ( int i = 0; i < nodeCount; ++i )
		{
			level->parents[i] = parents[i];
			level->offsets[i] = b3Sub( positions[i], level->positions[parents[i]] );
		}

		// A bond acts at the same point as before, now measured from the cluster it hangs on
		int count = 0;
		for ( int i = 0; i < bondCount; ++i )
		{
			const b3SolverBond* bond = bonds + i;
			bool isGround = bond->b == groundNode;
			if ( isGround == false && parents[bond->a] == parents[bond->b] )
			{
				continue;
			}

			b3SolverBond* coarse = level->bonds + count;
			coarse->a = parents[bond->a];
			coarse->rA = b3Add( bond->rA, level->offsets[bond->a] );
			coarse->b = isGround ? clusterCount : parents[bond->b];
			coarse->rB = isGround ? bond->rB : b3Add( bond->rB, level->offsets[bond->b] );
			coarse->kl = bond->kl;
			coarse->ka = bond->ka;
			level->sources[count] = i;
			count += 1;
		}

		B3_ASSERT( count == coarseBondCount );

		system->levels[system->levelCount++] = level;

		bonds = level->bonds;
		bondCount = coarseBondCount;
		nodeCount = clusterCount;
		positions = level->positions;
	}
}

// Bring the stiffness of the coarse levels up to date with the bonds and rebuild their blocks.
static void b3UpdateStressLevels( b3StressSystem* system, b3Arena arena )
{
	const b3SolverBond* fineBonds = system->bonds;
	int work = system->bondCount;

	for ( int i = 0; i < system->levelCount; ++i )
	{
		b3StressLevel* level = system->levels[i];
		for ( int j = 0; j < level->bondCount; ++j )
		{
			const b3SolverBond* source = fineBonds + level->sources[j];
			level->bonds[j].kl = source->kl;
			level->bonds[j].ka = source->ka;
		}

		b3BuildPrecond( level->bonds, level->bondCount, level->nodeCount, level->precond, arena );
		fineBonds = level->bonds;

		// A cycle multiplies twice on every level it passes through
		work += 2 * level->bondCount;
	}

	if ( system->levelCount > 0 )
	{
		work += 2 * system->bondCount;
	}

	system->iterationWork = work;
}

typedef struct b3ConjugateGradient
{
	const b3SolverBond* bonds;
	int bondCount;
	int nodeCount;
	const b3NodePrecond* precond;
	b3StressLevel* const* levels;
	int levelCount;
	b3DVec6* x;
	b3DVec6* r;
	b3DVec6* z;
	b3DVec6* p;
	b3DVec6* q;
	const b3DVec6* f;
	// Work of the residual and of the loads: each dotted with its preconditioned self
	double rz;
	double loadWork;
	bool valid;
} b3ConjugateGradient;

// Damping of the block smoother in a cycle. Below one so that the smoother never overshoots.
#define B3_SMOOTHER_DAMPING 0.8

// correction += damping * inverse(blocks) * ( r - K correction ), or the same from zero
static void b3SmoothLevel( const b3SolverBond* bonds, int bondCount, int nodeCount, const b3NodePrecond* precond, const b3DVec6* r,
						   b3DVec6* z, b3DVec6* scratch, bool fromZero )
{
	if ( fromZero )
	{
		for ( int i = 0; i < nodeCount; ++i )
		{
			b3DVec6 d = b3ApplyPrecond( precond + i, r + i );
			z[i].lin = b3DScale( B3_SMOOTHER_DAMPING, d.lin );
			z[i].ang = b3DScale( B3_SMOOTHER_DAMPING, d.ang );
		}

		z[nodeCount] = (b3DVec6){ 0 };
		return;
	}

	b3StressMultiply( bonds, bondCount, nodeCount, z, scratch );
	for ( int i = 0; i < nodeCount; ++i )
	{
		b3DVec6 residual;
		residual.lin = b3DSub( r[i].lin, scratch[i].lin );
		residual.ang = b3DSub( r[i].ang, scratch[i].ang );
		b3DVec6 d = b3ApplyPrecond( precond + i, &residual );
		z[i].lin = b3DAdd( z[i].lin, b3DScale( B3_SMOOTHER_DAMPING, d.lin ) );
		z[i].ang = b3DAdd( z[i].ang, b3DScale( B3_SMOOTHER_DAMPING, d.ang ) );
	}
}

// z = M r, with M one symmetric cycle through the levels: smooth, correct on the coarser level,
// smooth again. The smoother fixes the error between neighbors, the coarse levels fix the error
// that spans the piece, which is the part plain iterations take forever to move.
static void b3ApplyStressCycle( const b3ConjugateGradient* cg, const b3DVec6* r, b3DVec6* z, b3DVec6* scratch )
{
	int levelCount = cg->levelCount;

	// Down: smooth and hand the remaining residual to the next level
	const b3SolverBond* bonds = cg->bonds;
	int bondCount = cg->bondCount;
	int nodeCount = cg->nodeCount;
	const b3NodePrecond* precond = cg->precond;
	const b3DVec6* levelR = r;
	b3DVec6* levelZ = z;
	b3DVec6* levelT = scratch;

	for ( int i = 0; i < levelCount; ++i )
	{
		b3StressLevel* coarse = cg->levels[i];

		b3SmoothLevel( bonds, bondCount, nodeCount, precond, levelR, levelZ, levelT, true );

		b3StressMultiply( bonds, bondCount, nodeCount, levelZ, levelT );
		memset( coarse->r, 0, ( coarse->nodeCount + 1 ) * sizeof( b3DVec6 ) );

		for ( int j = 0; j < nodeCount; ++j )
		{
			b3DVec3 force = b3DSub( levelR[j].lin, levelT[j].lin );
			b3DVec3 torque = b3DSub( levelR[j].ang, levelT[j].ang );
			b3DVec3 offset = b3DMakeVec3( coarse->offsets[j] );

			// Force and torque about the cluster position
			b3DVec6* target = coarse->r + coarse->parents[j];
			target->lin = b3DAdd( target->lin, force );
			target->ang = b3DAdd( target->ang, b3DAdd( torque, b3DCross( offset, force ) ) );
		}

		bonds = coarse->bonds;
		bondCount = coarse->bondCount;
		nodeCount = coarse->nodeCount;
		precond = coarse->precond;
		levelR = coarse->r;
		levelZ = coarse->z;
		levelT = coarse->t;
	}

	// The coarsest level is small. Its blocks alone are a good enough answer.
	for ( int i = 0; i < nodeCount; ++i )
	{
		levelZ[i] = b3ApplyPrecond( precond + i, levelR + i );
	}

	levelZ[nodeCount] = (b3DVec6){ 0 };

	// Up: move the nodes with their clusters and smooth again
	for ( int i = levelCount - 1; i >= 0; --i )
	{
		const b3StressLevel* coarse = cg->levels[i];
		const b3StressLevel* fine = i > 0 ? cg->levels[i - 1] : NULL;

		bonds = fine != NULL ? fine->bonds : cg->bonds;
		bondCount = fine != NULL ? fine->bondCount : cg->bondCount;
		nodeCount = fine != NULL ? fine->nodeCount : cg->nodeCount;
		precond = fine != NULL ? fine->precond : cg->precond;
		levelR = fine != NULL ? fine->r : r;
		levelZ = fine != NULL ? fine->z : z;
		levelT = fine != NULL ? fine->t : scratch;

		for ( int j = 0; j < nodeCount; ++j )
		{
			const b3DVec6* motion = coarse->z + coarse->parents[j];
			b3DVec3 offset = b3DMakeVec3( coarse->offsets[j] );
			levelZ[j].lin = b3DAdd( levelZ[j].lin, b3DAdd( motion->lin, b3DCross( motion->ang, offset ) ) );
			levelZ[j].ang = b3DAdd( levelZ[j].ang, motion->ang );
		}

		b3SmoothLevel( bonds, bondCount, nodeCount, precond, levelR, levelZ, levelT, false );
	}
}

// z = M r for the whole system. Returns the dot product of r with z.
static double b3Precondition( b3ConjugateGradient* cg, const b3DVec6* r, b3DVec6* z )
{
	int n = cg->nodeCount;
	double rz = 0.0;

	if ( cg->levelCount == 0 )
	{
		for ( int i = 0; i < n; ++i )
		{
			z[i] = b3ApplyPrecond( cg->precond + i, r + i );
			rz += b3DDot( r[i].lin, z[i].lin ) + b3DDot( r[i].ang, z[i].ang );
		}
	}
	else
	{
		// The scratch vector q is free between two multiplies
		b3ApplyStressCycle( cg, r, z, cg->q );

		for ( int i = 0; i < n; ++i )
		{
			rz += b3DDot( r[i].lin, z[i].lin ) + b3DDot( r[i].ang, z[i].ang );
		}
	}

	z[n] = (b3DVec6){ 0 };
	return rz;
}

// Run preconditioned conjugate gradient iterations until the error drops below the tolerance or the
// iterations run out. Continues a previous run when the state is still valid.
//
// The error is measured in energy: the work the remaining residual would do, against the work the
// loads do. That is the measure conjugate gradient brings down in every iteration, and it is the
// one that matters for stress. The plain size of the residual is not: it jumps up and down on the
// way, and a few stiff fragments that are slightly off can dominate it while the bond forces are
// already right.
static int b3RunConjugateGradient( b3ConjugateGradient* cg, int maxIterations, double tolerance )
{
	int n = cg->nodeCount;

	if ( cg->valid == false )
	{
		// Work of the loads, the reference for the error. Uses p as scratch.
		cg->loadWork = b3Precondition( cg, cg->f, cg->p );

		// Restart: residual of the current displacement
		b3StressMultiply( cg->bonds, cg->bondCount, n, cg->x, cg->q );

		for ( int i = 0; i < n; ++i )
		{
			cg->r[i].lin = b3DSub( cg->f[i].lin, cg->q[i].lin );
			cg->r[i].ang = b3DSub( cg->f[i].ang, cg->q[i].ang );
		}

		cg->r[n] = (b3DVec6){ 0 };
		cg->rz = b3Precondition( cg, cg->r, cg->z );
		memcpy( cg->p, cg->z, ( n + 1 ) * sizeof( b3DVec6 ) );
		cg->valid = true;
	}

	double target = tolerance * tolerance * cg->loadWork;
	int iteration = 0;
	while ( iteration < maxIterations && cg->rz > target )
	{
		b3StressMultiply( cg->bonds, cg->bondCount, n, cg->p, cg->q );

		double pq = b3DDot6( cg->p, cg->q, n );
		if ( !( pq > 0.0 ) || !( cg->rz > 0.0 ) )
		{
			// Nothing left to gain along this direction. Happens once the solve reaches the limit of
			// the arithmetic. The next call starts over from the true residual.
			cg->valid = false;
			break;
		}

		double alpha = cg->rz / pq;
		for ( int i = 0; i < n; ++i )
		{
			b3DVec6* x = cg->x + i;
			const b3DVec6* p = cg->p + i;
			const b3DVec6* q = cg->q + i;
			b3DVec6* r = cg->r + i;

			x->lin.x += alpha * p->lin.x;
			x->lin.y += alpha * p->lin.y;
			x->lin.z += alpha * p->lin.z;
			x->ang.x += alpha * p->ang.x;
			x->ang.y += alpha * p->ang.y;
			x->ang.z += alpha * p->ang.z;

			r->lin.x -= alpha * q->lin.x;
			r->lin.y -= alpha * q->lin.y;
			r->lin.z -= alpha * q->lin.z;
			r->ang.x -= alpha * q->ang.x;
			r->ang.y -= alpha * q->ang.y;
			r->ang.z -= alpha * q->ang.z;
		}

		double rz = b3Precondition( cg, cg->r, cg->z );

		double beta = rz / cg->rz;
		for ( int i = 0; i < n; ++i )
		{
			b3DVec6* p = cg->p + i;
			const b3DVec6* z = cg->z + i;
			p->lin.x = z->lin.x + beta * p->lin.x;
			p->lin.y = z->lin.y + beta * p->lin.y;
			p->lin.z = z->lin.z + beta * p->lin.z;
			p->ang.x = z->ang.x + beta * p->ang.x;
			p->ang.y = z->ang.y + beta * p->ang.y;
			p->ang.z = z->ang.z + beta * p->ang.z;
		}

		cg->rz = rz;
		iteration += 1;
	}

	return iteration;
}

// The relative error of a solve, in the energy measure the solver uses
static inline double b3GetSolveError( double rz, double loadWork )
{
	if ( loadWork <= 0.0 )
	{
		return rz > 0.0 ? 1.0 : 0.0;
	}

	return sqrt( b3MaxFloat( 0.0f, (float)( rz / loadWork ) ) );
}

// Balance the loads on a free set of fragments with the inertia forces of the rigid motion they
// cause, so the remaining loads only deform it. The loads enter as force and torque about each
// fragment center and leave balanced. The angular velocity adds the centripetal pull of a spin.
// The mass properties must be those of exactly these fragments, which is passed as a piece that
// holds them. It need not be the piece the fragments currently belong to.
static void b3RelieveInertia( const b3Destructible* destructible, const b3DestructionPiece* piece, const int* nodes, int nodeCount,
							  b3Vec3 angularVelocity, b3DVec6* loads )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );

	b3DVec3 totalForce = { 0 };
	b3DVec3 totalTorque = { 0 };
	b3DVec3 center = b3DMakeVec3( piece->center );

	for ( int i = 0; i < nodeCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + nodes[i];
		b3DVec3 r = b3DSub( b3DMakeVec3( fragment->center ), center );
		totalForce = b3DAdd( totalForce, loads[i].lin );
		totalTorque = b3DAdd( totalTorque, b3DAdd( loads[i].ang, b3DCross( r, loads[i].lin ) ) );
	}

	if ( piece->mass <= 0.0f )
	{
		return;
	}

	// Rigid body acceleration: m a = F and I alpha + w x I w = T
	b3Vec3 w = angularVelocity;
	b3Vec3 gyro = b3Cross( w, b3MulMV( piece->inertia, w ) );
	b3Vec3 torque = { (float)totalTorque.x - gyro.x, (float)totalTorque.y - gyro.y, (float)totalTorque.z - gyro.z };
	b3Vec3 alpha = b3MulMV( piece->invInertia, torque );
	b3DVec3 a = b3DScale( 1.0 / piece->mass, totalForce );

	b3DVec3 dw = b3DMakeVec3( w );
	b3DVec3 dalpha = b3DMakeVec3( alpha );

	for ( int i = 0; i < nodeCount; ++i )
	{
		const b3StructureFragment* fragment = fragments + nodes[i];
		b3DVec3 r = b3DSub( b3DMakeVec3( fragment->center ), center );

		// Acceleration of this fragment in the rigid motion
		b3DVec3 acceleration = b3DAdd( a, b3DAdd( b3DCross( dalpha, r ), b3DCross( dw, b3DCross( dw, r ) ) ) );
		loads[i].lin = b3DSub( loads[i].lin, b3DScale( fragment->mass, acceleration ) );

		b3Vec3 spin = b3Add( b3MulMV( fragment->inertia, alpha ), b3Cross( w, b3MulMV( fragment->inertia, w ) ) );
		loads[i].ang = b3DSub( loads[i].ang, b3DMakeVec3( spin ) );
	}
}

// Gather the sustained loads of a piece: gravity, smoothed contact forces, and inertia.
// Returns false if the piece carries no load.
static bool b3BuildLoads( b3World* world, b3Destructible* destructible, b3DestructionPiece* piece, b3Arena arena )
{
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );
	b3StressSystem* system = piece->system;
	int nodeCount = piece->nodeCount;

	b3Body* body = b3Array_Get( world->bodies, piece->bodyId );
	b3BodySim* bodySim = b3GetBodySim( world, body );
	b3Quat rotation = bodySim->transform.q;

	// Gravity in structure space
	b3Vec3 gravity = b3InvRotateVector( rotation, b3MulSV( bodySim->gravityScale, world->gravity ) );

	// Compare with the weight of the piece to tell a real load from round-off left by inertia relief.
	double weight = (double)piece->mass * b3Length( world->gravity );
	double floorSqr = 1.0e-10 * weight * weight;

	if ( piece->flags & b3_pieceAnchored )
	{
		// The anchor body may ride on a moving host. Its acceleration acts like extra gravity.
		b3Vec3 a0 = destructible->anchorLinearAcceleration;
		b3Vec3 alpha = destructible->anchorAngularAcceleration;
		b3Vec3 w = destructible->anchorAngularVelocity;
		bool isMoving = b3LengthSquared( a0 ) + b3LengthSquared( alpha ) + b3LengthSquared( w ) > 0.0f;

		for ( int i = 0; i < nodeCount; ++i )
		{
			int node = piece->nodes[i];
			const b3StructureFragment* fragment = fragments + node;
			b3NodeLoad* load = destructible->loads + node;

			b3Vec3 force = b3MulAdd( load->force, fragment->mass, gravity );
			b3Vec3 torque = load->torque;

			// Remember the contact force this solve uses. The loads are only rebuilt, and the solve
			// only restarted, once the smoothed force has drifted away from it. Without this the
			// contact jitter of resting debris restarts the solve every step and it never converges.
			load->solvedForce = load->force;

			if ( isMoving )
			{
				b3Vec3 c = fragment->center;
				b3Vec3 acceleration = b3Add( a0, b3Add( b3Cross( alpha, c ), b3Cross( w, b3Cross( w, c ) ) ) );
				force = b3MulSub( force, fragment->mass, acceleration );
				torque = b3Sub( torque, b3Add( b3MulMV( fragment->inertia, alpha ), b3Cross( w, b3MulMV( fragment->inertia, w ) ) ) );
			}

			system->f[i].lin = b3DMakeVec3( force );
			system->f[i].ang = b3DMakeVec3( torque );
		}
	}
	else
	{
		b3Vec3 angularVelocity = b3Vec3_zero;
		b3BodyState* state = b3GetBodyState( world, body );
		if ( state != NULL )
		{
			angularVelocity = b3InvRotateVector( rotation, state->angularVelocity );
		}

		b3DVec6* loads = b3Bump( &arena, nodeCount * (int)sizeof( b3DVec6 ) );
		for ( int i = 0; i < nodeCount; ++i )
		{
			int node = piece->nodes[i];
			const b3StructureFragment* fragment = fragments + node;
			const b3NodeLoad* load = destructible->loads + node;
			loads[i].lin = b3DMakeVec3( b3MulAdd( load->force, fragment->mass, gravity ) );
			loads[i].ang = b3DMakeVec3( load->torque );
		}

		b3RelieveInertia( destructible, piece, piece->nodes, nodeCount, angularVelocity, loads );

		// The loads of a free piece move a little every step with its contacts and its spin. Keep
		// solving for the loads the solver already works on until the real ones have drifted away
		// from them. Starting over every step throws away what the solver has learned, and a large
		// piece would never get to an answer.
		if ( system->loadsValid )
		{
			double driftSqr = 0.0, loadSqr = 0.0;
			for ( int i = 0; i < nodeCount; ++i )
			{
				b3DVec3 dl = b3DSub( loads[i].lin, system->f[i].lin );
				b3DVec3 da = b3DSub( loads[i].ang, system->f[i].ang );
				driftSqr += b3DDot( dl, dl ) + b3DDot( da, da );
				loadSqr += b3DDot( loads[i].lin, loads[i].lin ) + b3DDot( loads[i].ang, loads[i].ang );
			}

			double reference = loadSqr > system->loadSqr ? loadSqr : system->loadSqr;
			if ( driftSqr <= B3_LOAD_DRIFT * B3_LOAD_DRIFT * reference )
			{
				piece->flags &= ~b3_pieceLoadDirty;
				return system->loadSqr > floorSqr;
			}
		}

		memcpy( system->f, loads, nodeCount * sizeof( b3DVec6 ) );
	}

	system->f[nodeCount] = (b3DVec6){ 0 };
	system->loadSqr = b3DDot6( system->f, system->f, nodeCount );
	system->loadsValid = true;
	system->loadsEvaluated = false;
	system->loadSteps = 0;
	system->cgValid = false;
	system->evaluated = false;
	piece->flags &= ~b3_pieceLoadDirty;

	return system->loadSqr > floorSqr;
}

// Stress measures of a bond relative to its limits
typedef struct b3BondStress
{
	float elasticRatio;
	float fatalRatio;

	// Damage measure: zero below the elastic limits, one at a fatal limit
	float damage;

	// Magnitude of the force
	float force;
} b3BondStress;

static inline float b3DamageTerm( float stress, float elastic, float fatal )
{
	if ( stress <= elastic )
	{
		return 0.0f;
	}

	if ( fatal <= elastic )
	{
		return 1.0f;
	}

	return ( stress - elastic ) / ( fatal - elastic );
}

// Convert the force and moment on a bond into stress and compare with the limits.
static b3BondStress b3ComputeBondStress( const b3DestructionBond* bond, const b3BondLimits* limits, b3DVec3 force, b3DVec3 moment )
{
	b3DVec3 n = b3DMakeVec3( bond->normal );

	// Split into the parts along and across the normal
	double axialForce = b3DDot( force, n );
	b3DVec3 shearVector = b3DSub( force, b3DScale( axialForce, n ) );
	double shearForce = sqrt( b3DDot( shearVector, shearVector ) );
	double twistMoment = b3DDot( moment, n );
	b3DVec3 bendVector = b3DSub( moment, b3DScale( twistMoment, n ) );
	double bendMoment = sqrt( b3DDot( bendVector, bendVector ) );

	// Beam formulas for a square patch of side sqrt(A): section modulus A^1.5 / 6 in bending and
	// about A^1.5 / 4.8 in torsion.
	double area = bond->area;
	double invArea = 1.0 / area;
	double invArea15 = invArea / sqrt( area );

	double axial = axialForce * invArea;
	double bend = 6.0 * bendMoment * invArea15;
	double shear = shearForce * invArea + 4.8 * fabs( twistMoment ) * invArea15;

	// Bending pulls one edge of the patch and pushes the other
	float tension = (float)( axial + bend );
	float compression = (float)( -axial + bend );
	float shearStress = (float)shear;

	b3BondStress result;
	result.elasticRatio = b3MaxFloat( tension / limits->tensileElastic,
									  b3MaxFloat( compression / limits->compressiveElastic, shearStress / limits->shearElastic ) );
	result.fatalRatio = b3MaxFloat( tension / limits->tensileFatal,
									b3MaxFloat( compression / limits->compressiveFatal, shearStress / limits->shearFatal ) );
	result.elasticRatio = b3MaxFloat( 0.0f, result.elasticRatio );
	result.fatalRatio = b3MaxFloat( 0.0f, result.fatalRatio );
	result.damage = b3DamageTerm( tension, limits->tensileElastic, limits->tensileFatal ) +
					b3DamageTerm( compression, limits->compressiveElastic, limits->compressiveFatal ) +
					b3DamageTerm( shearStress, limits->shearElastic, limits->shearFatal );
	result.force = (float)sqrt( b3DDot( force, force ) );
	return result;
}

// Queue a bond to break at the commit.
static void b3QueueBondBreak( b3DestructionPiece* piece, b3DestructionBond* bond, int bondIndex, float force, float excess,
							  b3BondBreakCause cause )
{
	if ( bond->flags & ( b3_bondBroken | b3_bondBreaking ) )
	{
		return;
	}

	bond->flags |= b3_bondBreaking;
	b3BondBreak bondBreak = { bondIndex, force, excess, cause };
	b3Array_Push( piece->breaks, bondBreak );
}

// Shrink the effective area of a bond. Returns false if the bond has nothing left to give.
static bool b3WeakenBond( const b3Destructible* destructible, b3DestructionBond* bond, b3SolverBond* solverBond, float keep )
{
	bond->area *= b3ClampFloat( keep, 0.0f, 1.0f );
	if ( bond->area < B3_MIN_BOND_AREA_FRACTION * bond->area0 )
	{
		return false;
	}

	if ( solverBond != NULL )
	{
		const b3StructureMaterial* material = b3GetStructureMaterials( destructible->structure ) + bond->materialIndex;
		b3GetBondStiffness( material, bond, &solverBond->kl, &solverBond->ka );
	}

	return true;
}

// Evaluate every bond of a piece against the sustained loads. Creep only applies when the solve is
// close enough to trust. A bond breaks once its overload reaches minOverload: one for a solve that is
// close, more for a rough one, so that only bonds that are over their limit beyond doubt break early.
static void b3EvaluateSustained( b3Destructible* destructible, b3DestructionPiece* piece, float timeStep, bool allowCreep, float minOverload,
								 b3Arena arena )
{
	b3StressSystem* system = piece->system;
	const b3StructureMaterial* materials = b3GetStructureMaterials( destructible->structure );
	int bondCount = system->bondCount;
	bool anyCreep = false;
	bool anyChange = false;

	// How far each bond is past the point of breaking. Zero for a bond that holds.
	float* overloads = b3Bump( &arena, bondCount * (int)sizeof( float ) );
	float* forces = b3Bump( &arena, bondCount * (int)sizeof( float ) );
	float maxOverload = 0.0f;

	for ( int i = 0; i < bondCount; ++i )
	{
		int bondIndex = system->bondIds[i];
		b3DestructionBond* bond = destructible->bonds + bondIndex;
		b3SolverBond* solverBond = system->bonds + i;

		b3DVec3 force, moment;
		b3GetBondWrench( solverBond, system->x, &force, &moment );

		b3BondLimits limits = b3GetBondLimits( destructible, bond );
		b3BondStress stress = b3ComputeBondStress( bond, &limits, force, moment );
		bond->elasticRatio = stress.elasticRatio;
		bond->fatalRatio = stress.fatalRatio;
		overloads[i] = 0.0f;
		forces[i] = stress.force;

		if ( bond->flags & ( b3_bondRigid | b3_bondBreaking ) )
		{
			continue;
		}

		if ( stress.fatalRatio >= 1.0f || stress.damage >= 1.0f )
		{
			float overload = b3MaxFloat( 1.0f, stress.fatalRatio );
			if ( overload >= minOverload )
			{
				overloads[i] = overload;
				maxOverload = b3MaxFloat( maxOverload, overload );
			}

			continue;
		}

		if ( allowCreep && stress.damage > 0.0f )
		{
			// Held between the elastic and the fatal limit: fail over the creep time
			float creepTime = materials[bond->materialIndex].creepTime;
			float keep = 1.0f - stress.damage * timeStep / creepTime;
			if ( b3WeakenBond( destructible, bond, solverBond, keep ) == false )
			{
				b3QueueBondBreak( piece, bond, bondIndex, stress.force, 0.0f, b3_bondBreakCreep );
			}

			anyCreep = true;
			anyChange = true;
		}
	}

	if ( maxOverload > 0.0f )
	{
		// Failure runs in rounds. This solve assumed every bond holds, so it shows many bonds over
		// their limit that would never get there: once the worst ones go, the load moves elsewhere or
		// the piece comes free and the stress is gone. Break only the bonds near the worst overload
		// and let the next solve decide about the rest.
		float threshold = B3_BREAK_BAND * maxOverload;

		for ( int i = 0; i < bondCount; ++i )
		{
			if ( overloads[i] > 0.0f && overloads[i] >= threshold )
			{
				// No excess here: a slow overload has no impactor to give momentum back to
				int bondIndex = system->bondIds[i];
				b3QueueBondBreak( piece, destructible->bonds + bondIndex, bondIndex, forces[i], 0.0f, b3_bondBreakOverload );
			}
		}

		anyChange = true;
	}

	if ( anyCreep )
	{
		piece->flags |= b3_pieceCreeping;
	}
	else
	{
		piece->flags &= ~b3_pieceCreeping;
	}

	if ( anyChange )
	{
		// Stiffness changed, so the solver state no longer matches
		system->cgValid = false;
		system->precondValid = false;
		system->evaluated = false;
	}
	else
	{
		system->evaluated = true;
	}
}

// A bond of the local impact system
typedef struct b3ImpactBond
{
	int bondIndex;

	// Minus one when the local bond runs from fragment B to the ground, so its force has the
	// opposite sign of the bond convention.
	float sign;
} b3ImpactBond;

// What an impact wants to do to a bond
typedef struct b3ImpactDamage
{
	int bondIndex;

	// How far over its limit the bond is. The impact energy goes to the worst bonds first.
	float fatalRatio;

	// Share of the bond area to remove. One breaks the bond.
	float damage;

	// Force on the bond and the share of it above what the bond could carry
	float force;
	float excess;
} b3ImpactDamage;

// Worst bond first. The bond index makes this a total order, so the result does not depend on the
// sort implementation.
static int b3CompareImpactDamage( const void* a, const void* b )
{
	const b3ImpactDamage* damageA = a;
	const b3ImpactDamage* damageB = b;

	if ( damageA->fatalRatio != damageB->fatalRatio )
	{
		return damageA->fatalRatio > damageB->fatalRatio ? -1 : 1;
	}

	return damageA->bondIndex < damageB->bondIndex ? -1 : ( damageA->bondIndex > damageB->bondIndex ? 1 : 0 );
}

// Solve the stress an impact causes near the hit and damage the bonds there at once.
static int b3SolveImpact( b3World* world, b3Destructible* destructible, b3DestructionPiece* piece, b3Arena arena )
{
	const b3DestructionSettings* settings = &world->destruction->settings;
	const b3StructureFragment* fragments = b3GetStructureFragments( destructible->structure );
	const b3StructureMaterial* materials = b3GetStructureMaterials( destructible->structure );
	const int* adjacencyStarts = destructible->adjacencyStarts;
	const int* adjacency = destructible->adjacency;

	int nodeCount = piece->nodeCount;
	float radius = settings->impactRadius;
	float timeStep = world->destruction->timeStep;
	float duration = settings->impactDuration > 0.0f ? settings->impactDuration : timeStep;
	if ( duration <= 0.0f )
	{
		return 0;
	}

	// Breaking a surface costs energy. An impact that has none to give cannot damage a bond.
	float energy = piece->impactEnergy;
	if ( energy <= 0.0f && destructible->minToughness > 0.0f )
	{
		return 0;
	}

	// Grow the free region outward from the hit fragments along intact bonds.
	int* localIndex = b3Bump( &arena, nodeCount * (int)sizeof( int ) );
	int* queue = b3Bump( &arena, nodeCount * (int)sizeof( int ) );
	b3Vec3* origins = b3Bump( &arena, nodeCount * (int)sizeof( b3Vec3 ) );
	memset( localIndex, 0xFF, nodeCount * sizeof( int ) );

	int freeCount = 0;
	for ( int i = 0; i < nodeCount; ++i )
	{
		int node = piece->nodes[i];
		if ( destructible->nodeFlags[node] & b3_nodeImpact )
		{
			localIndex[i] = freeCount;
			queue[freeCount] = i;
			origins[freeCount] = fragments[node].center;
			freeCount += 1;
		}
	}

	if ( freeCount == 0 )
	{
		return 0;
	}

	int bondCapacity = 0;
	for ( int head = 0; head < freeCount; ++head )
	{
		int pieceLocal = queue[head];
		int node = piece->nodes[pieceLocal];
		b3Vec3 origin = origins[head];

		for ( int j = adjacencyStarts[node]; j < adjacencyStarts[node + 1]; ++j )
		{
			const b3DestructionBond* bond = destructible->bonds + adjacency[j];
			if ( bond->flags & ( b3_bondBroken | b3_bondBreaking ) )
			{
				continue;
			}

			bondCapacity += 1;

			if ( bond->nodeB == B3_NULL_INDEX )
			{
				continue;
			}

			int other = bond->nodeA == node ? bond->nodeB : bond->nodeA;
			int otherLocal = destructible->fragmentLocal[other];
			if ( localIndex[otherLocal] != B3_NULL_INDEX )
			{
				continue;
			}

			if ( b3DistanceSquared( fragments[other].center, origin ) > radius * radius )
			{
				continue;
			}

			localIndex[otherLocal] = freeCount;
			queue[freeCount] = otherLocal;
			origins[freeCount] = origin;
			freeCount += 1;
		}
	}

	// Local system. Fragments outside the free region are clamped: their side of a bond is the ground.
	int groundNode = freeCount;
	b3SolverBond* solverBonds = b3Bump( &arena, bondCapacity * (int)sizeof( b3SolverBond ) );
	b3ImpactBond* impactBonds = b3Bump( &arena, bondCapacity * (int)sizeof( b3ImpactBond ) );
	int* freeNodes = b3Bump( &arena, freeCount * (int)sizeof( int ) );
	int bondCount = 0;
	bool isHeld = false;

	for ( int i = 0; i < freeCount; ++i )
	{
		int node = piece->nodes[queue[i]];
		freeNodes[i] = node;

		for ( int j = adjacencyStarts[node]; j < adjacencyStarts[node + 1]; ++j )
		{
			int bondIndex = adjacency[j];
			const b3DestructionBond* bond = destructible->bonds + bondIndex;
			if ( bond->flags & ( b3_bondBroken | b3_bondBreaking ) )
			{
				continue;
			}

			b3SolverBond* solverBond = solverBonds + bondCount;
			b3ImpactBond* impactBond = impactBonds + bondCount;
			impactBond->bondIndex = bondIndex;
			b3GetBondStiffness( materials + bond->materialIndex, bond, &solverBond->kl, &solverBond->ka );

			if ( bond->nodeA == node )
			{
				int otherLocal = bond->nodeB == B3_NULL_INDEX ? B3_NULL_INDEX : localIndex[destructible->fragmentLocal[bond->nodeB]];
				solverBond->a = i;
				solverBond->rA = bond->rA;
				solverBond->b = otherLocal == B3_NULL_INDEX ? groundNode : otherLocal;
				solverBond->rB = otherLocal == B3_NULL_INDEX ? b3Vec3_zero : bond->rB;
				impactBond->sign = 1.0f;
				isHeld = isHeld || otherLocal == B3_NULL_INDEX;
				bondCount += 1;
			}
			else if ( localIndex[destructible->fragmentLocal[bond->nodeA]] == B3_NULL_INDEX )
			{
				// Fragment A is clamped. Run the bond from this fragment to the ground.
				solverBond->a = i;
				solverBond->rA = bond->rB;
				solverBond->b = groundNode;
				solverBond->rB = b3Vec3_zero;
				impactBond->sign = -1.0f;
				isHeld = true;
				bondCount += 1;
			}

			// Otherwise the bond was added from the side of fragment A.
		}
	}

	if ( bondCount == 0 )
	{
		return 0;
	}

	int vectorCount = freeCount + 1;
	b3DVec6* vectors = b3Bump( &arena, 6 * vectorCount * (int)sizeof( b3DVec6 ) );
	b3NodePrecond* precond = b3Bump( &arena, vectorCount * (int)sizeof( b3NodePrecond ) );
	memset( vectors, 0, 6 * vectorCount * sizeof( b3DVec6 ) );

	b3DVec6* x = vectors + 0 * vectorCount;
	b3DVec6* f = vectors + 5 * vectorCount;

	// The impact force is the impulse spread over the impact duration.
	double invDuration = 1.0 / duration;
	for ( int i = 0; i < freeCount; ++i )
	{
		const b3NodeLoad* load = destructible->loads + freeNodes[i];
		f[i].lin = b3DScale( invDuration, b3DMakeVec3( load->impulse ) );
		f[i].ang = b3DScale( invDuration, b3DMakeVec3( load->angularImpulse ) );
	}

	if ( isHeld == false )
	{
		// Nothing holds the free fragments, so the blow accelerates them as a rigid body. This is
		// usually the whole piece. It is a part of it when the bonds to the rest are already set to
		// break in this step, so use the mass properties of just these fragments.
		b3DestructionPiece freePart = { 0 };
		freePart.nodes = freeNodes;
		freePart.nodeCount = freeCount;
		b3ComputePieceMass( destructible, &freePart );
		b3RelieveInertia( destructible, &freePart, freeNodes, freeCount, b3Vec3_zero, f );
	}

	double loadSqr = b3DDot6( f, f, freeCount );
	if ( loadSqr <= 0.0 )
	{
		return 0;
	}

	b3BuildPrecond( solverBonds, bondCount, freeCount, precond, arena );

	b3ConjugateGradient cg = { 0 };
	cg.bonds = solverBonds;
	cg.bondCount = bondCount;
	cg.nodeCount = freeCount;
	cg.precond = precond;
	cg.x = x;
	cg.r = vectors + 1 * vectorCount;
	cg.z = vectors + 2 * vectorCount;
	cg.p = vectors + 3 * vectorCount;
	cg.q = vectors + 4 * vectorCount;
	cg.f = f;
	cg.valid = false;

	double tolerance = settings->solverTolerance;
	int iterations = b3RunConjugateGradient( &cg, B3_IMPACT_ITERATIONS, tolerance );

	// Find the bonds the impact pushes past their limits, on top of the sustained stress.
	b3StressSystem* system = piece->system;
	b3ImpactDamage* damages = b3Bump( &arena, bondCount * (int)sizeof( b3ImpactDamage ) );
	int damageCount = 0;

	for ( int i = 0; i < bondCount; ++i )
	{
		int bondIndex = impactBonds[i].bondIndex;
		const b3DestructionBond* bond = destructible->bonds + bondIndex;
		if ( bond->flags & ( b3_bondRigid | b3_bondBreaking ) )
		{
			continue;
		}

		b3DVec3 impactForce, impactMoment;
		b3GetBondWrench( solverBonds + i, x, &impactForce, &impactMoment );
		impactForce = b3DScale( impactBonds[i].sign, impactForce );
		impactMoment = b3DScale( impactBonds[i].sign, impactMoment );

		b3DVec3 force = { 0 }, moment = { 0 };
		if ( system != NULL && bond->solverIndex != B3_NULL_INDEX )
		{
			b3GetBondWrench( system->bonds + bond->solverIndex, system->x, &force, &moment );
		}

		b3BondLimits limits = b3GetBondLimits( destructible, bond );
		b3BondStress sustained = b3ComputeBondStress( bond, &limits, force, moment );
		b3BondStress total = b3ComputeBondStress( bond, &limits, b3DAdd( force, impactForce ), b3DAdd( moment, impactMoment ) );

		b3ImpactDamage damage = { 0 };
		damage.bondIndex = bondIndex;
		damage.fatalRatio = total.fatalRatio;
		damage.force = total.force;

		if ( total.fatalRatio >= 1.0f || total.damage >= 1.0f )
		{
			// The part of the force above what the bond could carry never reached the far side.
			damage.damage = 1.0f;
			damage.excess = total.fatalRatio > 1.0f ? 1.0f - 1.0f / total.fatalRatio : 0.0f;
		}
		else
		{
			damage.damage = total.damage - sustained.damage;
		}

		if ( damage.damage > 0.0f )
		{
			damages[damageCount++] = damage;
		}
	}

	// The stress picks which bonds fail, the energy of the impact decides how many. A blow breaks the
	// worst bonds until it has spent the energy it lost, so a light tap cannot shatter a wall however
	// stiff the contact is, and the outcome does not depend on the time step.
	qsort( damages, damageCount, sizeof( b3ImpactDamage ), b3CompareImpactDamage );

	bool anyChange = false;

	for ( int i = 0; i < damageCount; ++i )
	{
		const b3ImpactDamage* damage = damages + i;
		b3DestructionBond* bond = destructible->bonds + damage->bondIndex;
		b3SolverBond* solverBond = NULL;
		if ( system != NULL && bond->solverIndex != B3_NULL_INDEX )
		{
			solverBond = system->bonds + bond->solverIndex;
		}

		// Energy to break what is left of the bond
		float breakEnergy = destructible->strengthScale * bond->strength * materials[bond->materialIndex].toughness * bond->area;
		float cost = damage->damage * breakEnergy;
		float share = damage->damage;
		bool isFull = cost <= energy;

		if ( isFull == false )
		{
			// Not enough left for this bond. Spend the rest on weakening it.
			share = energy / breakEnergy;
			cost = energy;
		}

		energy -= cost;

		if ( share <= 0.0f )
		{
			continue;
		}

		// An impact is one event, so its damage applies once and not per unit of time.
		anyChange = true;
		if ( isFull && damage->damage >= 1.0f )
		{
			b3QueueBondBreak( piece, bond, damage->bondIndex, damage->force, damage->excess, b3_bondBreakImpact );
		}
		else if ( b3WeakenBond( destructible, bond, solverBond, 1.0f - share ) == false )
		{
			b3QueueBondBreak( piece, bond, damage->bondIndex, damage->force, 0.0f, b3_bondBreakImpact );
		}
	}

	if ( anyChange && system != NULL )
	{
		system->cgValid = false;
		system->precondValid = false;
		system->evaluated = false;
	}

	return iterations;
}

// Set the stress of every bond of a piece to zero. Used when a piece carries no load.
static void b3ClearPieceStress( b3Destructible* destructible, b3DestructionPiece* piece )
{
	b3StressSystem* system = piece->system;
	if ( system == NULL )
	{
		return;
	}

	for ( int i = 0; i < system->bondCount; ++i )
	{
		b3DestructionBond* bond = destructible->bonds + system->bondIds[i];
		bond->elasticRatio = 0.0f;
		bond->fatalRatio = 0.0f;
	}

	memset( system->x, 0, ( system->nodeCount + 1 ) * sizeof( b3DVec6 ) );
	system->cgValid = false;
	system->evaluated = true;
	system->rz = 0.0;
	system->loadWork = 0.0;
	system->loadSqr = 0.0;
}

int b3SolvePieceStress( b3World* world, b3Destructible* destructible, int pieceIndex, b3Arena arena )
{
	const b3Destruction* destruction = world->destruction;
	const b3DestructionSettings* settings = &destruction->settings;
	b3DestructionPiece* piece = b3Array_Get( destructible->pieces, pieceIndex );
	int iterations = 0;

	if ( piece->flags & b3_pieceSystemDirty )
	{
		b3BuildStressSystem( destructible, piece );
	}

	b3StressSystem* system = piece->system;
	if ( system == NULL )
	{
		// No bonds, nothing to solve
		piece->flags &= ~( b3_pieceCreeping | b3_pieceLoadDirty );
		return 0;
	}

	if ( piece->breaks.count > 0 )
	{
		// Bonds the application asked to break go at the end of this step and may leave the piece in
		// parts. Solving it as a whole now would be wasted, so only the blows of this step count.
		if ( piece->flags & b3_pieceHasImpact )
		{
			iterations += b3SolveImpact( world, destructible, piece, arena );
		}

		return iterations;
	}

	bool isAnchored = ( piece->flags & b3_pieceAnchored ) != 0;

	// A large piece needs several steps to solve, and its loads move a little in every one of them.
	// Chasing them would mean starting over each step and never arriving. So the solver holds on to
	// the loads it started with until they are solved and checked against the bond limits, and only
	// then takes the loads of the moment. The piece reacts a few steps late instead of never.
	bool isBusy = system->loadsValid && system->loadsEvaluated == false && system->loadSteps < B3_MAX_LOAD_STEPS;
	system->loadSteps += 1;

	// The loads on a free piece change whenever it turns, so look at them every step it is not busy.
	if ( isBusy == false && ( isAnchored == false || ( piece->flags & b3_pieceLoadDirty ) != 0 ) )
	{
		bool isLoaded = b3BuildLoads( world, destructible, piece, arena );
		if ( isLoaded == false )
		{
			if ( ( piece->flags & b3_pieceUnloaded ) == 0 )
			{
				b3ClearPieceStress( destructible, piece );
				piece->flags |= b3_pieceUnloaded;
			}

			piece->flags &= ~b3_pieceCreeping;

			if ( piece->flags & b3_pieceHasImpact )
			{
				iterations += b3SolveImpact( world, destructible, piece, arena );
			}

			return iterations;
		}

		piece->flags &= ~b3_pieceUnloaded;
	}

	if ( system->precondValid == false )
	{
		b3BuildPrecond( system->bonds, system->bondCount, system->nodeCount, system->precond, arena );

		if ( system->levelsValid == false )
		{
			b3BuildStressLevels( destructible, piece, system, arena );
		}

		b3UpdateStressLevels( system, arena );
		system->precondValid = true;
		system->cgValid = false;
	}

	// Spend the work budget of this step
	int budget = settings->workBudget / b3MaxInt( 1, system->iterationWork );
	int maxIterations = b3ClampInt( budget, b3MinInt( 4, settings->maxIterations ), settings->maxIterations );

	double solverTolerance = settings->solverTolerance;

	b3ConjugateGradient cg = { 0 };
	cg.bonds = system->bonds;
	cg.bondCount = system->bondCount;
	cg.nodeCount = system->nodeCount;
	cg.precond = system->precond;
	cg.levels = system->levels;
	cg.levelCount = system->levelCount;
	cg.x = system->x;
	cg.r = system->r;
	cg.z = system->z;
	cg.p = system->p;
	cg.q = system->q;
	cg.f = system->f;
	cg.rz = system->rz;
	cg.loadWork = system->loadWork;
	cg.valid = system->cgValid;

	int used = b3RunConjugateGradient( &cg, maxIterations, solverTolerance );
	iterations += used;

	system->rz = cg.rz;
	system->loadWork = cg.loadWork;
	system->cgValid = cg.valid;
	if ( used > 0 )
	{
		system->evaluated = false;

		// Keep the solution with the fragments. The system is rebuilt when the piece loses bonds or
		// fragments, and the next solve continues from here.
		for ( int i = 0; i < system->nodeCount; ++i )
		{
			destructible->displacements[piece->nodes[i]] = system->x[i];
		}
	}

	// Sustained damage waits until the solve is close. A half converged solution can show stress that
	// is not there.
	double damageTolerance = b3MaxFloat( settings->damageTolerance, settings->solverTolerance );
	double error = b3GetSolveError( system->rz, system->loadWork );
	bool isClose = error <= damageTolerance;
	bool isConverged = error <= solverTolerance;

	// A bond that is far over its limit fails whatever is left to solve, so it need not wait. The
	// rougher the solve, the further over a bond must be. Without this a large piece that is beyond
	// saving would stand around until its solve is exact, and how long it stands would depend on
	// the work budget.
	float minOverload = 1.0f;
	if ( isClose == false )
	{
		minOverload = error < B3_MAX_ROUGH_ERROR ? (float)( 1.0 / ( 1.0 - error / B3_MAX_ROUGH_ERROR ) ) : FLT_MAX;
	}

	if ( settings->enableStress == false )
	{
		minOverload = FLT_MAX;
	}

	if ( system->evaluated == false || ( piece->flags & b3_pieceCreeping ) != 0 )
	{
		b3EvaluateSustained( destructible, piece, destruction->timeStep, isClose && settings->enableStress, minOverload, arena );

		if ( isClose )
		{
			system->loadsEvaluated = true;
		}

		// Keep refining until the solve reaches the solver tolerance
		if ( isConverged == false )
		{
			system->evaluated = false;
		}
	}

	if ( piece->flags & b3_pieceHasImpact )
	{
		iterations += b3SolveImpact( world, destructible, piece, arena );
	}

	return iterations;
}
