// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

// Wide front end of the narrow phase. Runs the two tests that decide the fate of most contacts in a
// pile on several contacts at once:
// - the contact recycling test of b3CollideTask
// - the distance test at the top of b3CollideCapsules
//
// Both are transcriptions of the scalar code, not approximations. Every value is computed with the same
// operations in the same order, and every branch on a float is a compare and blend with the same
// meaning for NaN. A lane therefore gets bit for bit the result the scalar code would compute, and the
// simulation does not depend on which path a contact took. Keep these in sync with the scalar code.

#include "body.h"
#include "contact.h"
#include "shape.h"
#include "simd_wide.h"

#include <float.h>

// Eight consecutive floats are loaded from each lane, so these must stay in this order.
_Static_assert( offsetof( b3Contact, cachedRotationB ) == offsetof( b3Contact, cachedRotationA ) + 4 * sizeof( float ),
				"contact recycling cache layout" );
_Static_assert( offsetof( b3Transform, q ) == 3 * sizeof( float ) && sizeof( b3Transform ) == 7 * sizeof( float ),
				"transform layout" );
_Static_assert( offsetof( b3Contact, cachedRelativePose ) + 8 * sizeof( float ) <= sizeof( b3Contact ), "contact layout" );
_Static_assert( offsetof( b3Capsule, center2 ) == 3 * sizeof( float ) && offsetof( b3Capsule, radius ) == 6 * sizeof( float ),
				"capsule layout" );
_Static_assert( offsetof( b3Shape, capsule ) + 8 * sizeof( float ) <= sizeof( b3Shape ), "shape layout" );

// Loads eight consecutive floats from each lane pointer and returns them as eight wide values
#if B3_SIMD_WIDTH == 8
#define B3_GATHER_ROWS( P, C0, C1, C2, C3, C4, C5, C6, C7 )                                                                      \
	b3TransposeW( b3LoadW( P[0] ), b3LoadW( P[1] ), b3LoadW( P[2] ), b3LoadW( P[3] ), b3LoadW( P[4] ), b3LoadW( P[5] ),          \
				  b3LoadW( P[6] ), b3LoadW( P[7] ), &C0, &C1, &C2, &C3, &C4, &C5, &C6, &C7 )
#else
#define B3_GATHER_ROWS( P, C0, C1, C2, C3, C4, C5, C6, C7 )                                                                      \
	b3TransposeW( b3LoadW( P[0] ), b3LoadW( P[1] ), b3LoadW( P[2] ), b3LoadW( P[3] ), &C0, &C1, &C2, &C3 );                      \
	b3TransposeW( b3LoadW( P[0] + 4 ), b3LoadW( P[1] + 4 ), b3LoadW( P[2] + 4 ), b3LoadW( P[3] + 4 ), &C4, &C5, &C6, &C7 )
#endif

// b3LengthSquared
static inline b3FloatW b3LengthSquaredCW( b3Vec3W a )
{
	return b3AddW( b3AddW( b3MulW( a.X, a.X ), b3MulW( a.Y, a.Y ) ), b3MulW( a.Z, a.Z ) );
}

// The contact recycling test of b3CollideTask for one contact per lane.
// Returns a bit per lane that is set if the contact can be recycled.
int B3_WIDE( b3TestContactRecycling )( const b3RecycleLanes* lanes, b3RecycleResults* results )
{
	b3QuatW qA, qB, rotationA, rotationB;

	// Origin of body B relative to the origin of body A in world space
	b3Vec3W d;

	b3FloatW unused;
	B3_UNUSED( unused );

#if defined( BOX3D_DOUBLE_PRECISION )
	{
		// The body origins are doubles, so the subtraction is done per lane like b3InvMulWorldTransforms
		_Alignas( B3_WIDE_ALIGNMENT ) float lane[11][B3_SIMD_WIDTH];
		for ( int i = 0; i < B3_SIMD_WIDTH; ++i )
		{
			b3WorldTransform xfA = lanes->simsA[i]->transform;
			b3WorldTransform xfB = lanes->simsB[i]->transform;
			lane[0][i] = (float)( xfB.p.x - xfA.p.x );
			lane[1][i] = (float)( xfB.p.y - xfA.p.y );
			lane[2][i] = (float)( xfB.p.z - xfA.p.z );
			lane[3][i] = xfA.q.v.x;
			lane[4][i] = xfA.q.v.y;
			lane[5][i] = xfA.q.v.z;
			lane[6][i] = xfA.q.s;
			lane[7][i] = xfB.q.v.x;
			lane[8][i] = xfB.q.v.y;
			lane[9][i] = xfB.q.v.z;
			lane[10][i] = xfB.q.s;
		}

		d = (b3Vec3W){ b3LoadW( lane[0] ), b3LoadW( lane[1] ), b3LoadW( lane[2] ) };
		qA = (b3QuatW){ { b3LoadW( lane[3] ), b3LoadW( lane[4] ), b3LoadW( lane[5] ) }, b3LoadW( lane[6] ) };
		qB = (b3QuatW){ { b3LoadW( lane[7] ), b3LoadW( lane[8] ), b3LoadW( lane[9] ) }, b3LoadW( lane[10] ) };
	}
#else
	{
		_Static_assert( offsetof( b3BodySim, transform ) == 0, "body sim layout" );

		const float* rowsA[B3_SIMD_WIDTH];
		const float* rowsB[B3_SIMD_WIDTH];
		for ( int i = 0; i < B3_SIMD_WIDTH; ++i )
		{
			rowsA[i] = (const float*)lanes->simsA[i];
			rowsB[i] = (const float*)lanes->simsB[i];
		}

		b3Vec3W pA, pB;
		B3_GATHER_ROWS( rowsA, pA.X, pA.Y, pA.Z, qA.V.X, qA.V.Y, qA.V.Z, qA.S, unused );
		B3_GATHER_ROWS( rowsB, pB.X, pB.Y, pB.Z, qB.V.X, qB.V.Y, qB.V.Z, qB.S, unused );
		d = b3SubVW( pB, pA );
	}
#endif

	// The cached rotations of both bodies and the cached relative pose
	b3Vec3W poseP;
	b3QuatW poseQ;
	{
		const float* rotationRows[B3_SIMD_WIDTH];
		const float* poseRows[B3_SIMD_WIDTH];
		for ( int i = 0; i < B3_SIMD_WIDTH; ++i )
		{
			rotationRows[i] = (const float*)&lanes->contacts[i]->cachedRotationA;
			poseRows[i] = (const float*)&lanes->contacts[i]->cachedRelativePose;
		}

		B3_GATHER_ROWS( rotationRows, rotationA.V.X, rotationA.V.Y, rotationA.V.Z, rotationA.S, rotationB.V.X, rotationB.V.Y,
						rotationB.V.Z, rotationB.S );
		B3_GATHER_ROWS( poseRows, poseP.X, poseP.Y, poseP.Z, poseQ.V.X, poseQ.V.Y, poseQ.V.Z, poseQ.S, unused );
	}

	b3FloatW tolerance = b3LoadW( lanes->tolerances );
	b3Vec3W maxExtent = { b3LoadW( lanes->extentX ), b3LoadW( lanes->extentY ), b3LoadW( lanes->extentZ ) };

	b3FloatW angleA = b3DotQuatJW( qA, rotationA );
	b3FloatW angleB = b3DotQuatJW( qB, rotationB );
	b3FloatW angularDistance = b3MinJW( b3MulW( angleA, angleA ), b3MulW( angleB, angleB ) );

	// b3InvMulWorldTransforms
	b3QuatW xfQ = b3InvMulQuatJW( qA, qB );
	b3Vec3W xfP = b3InvRotateVectorJW( qA, d );

	// b3DistanceSquared( xf.p, xfc.p )
	b3FloatW distSquared = b3LengthSquaredCW( b3SubVW( poseP, xfP ) );

	b3FloatW closeEnough = b3AndW( b3GreaterThanW( angularDistance, b3SplatW( B3_CONTACT_RECYCLE_ANGULAR_DISTANCE ) ),
								   b3LessThanW( distSquared, b3MulW( tolerance, tolerance ) ) );

	// The scalar code only gets here if the test above passes. A lane that failed computes values
	// that are never used.
	b3FloatW distance = b3SqrtW( distSquared );
	b3FloatW slack = b3SubW( tolerance, distance );

	b3QuatW qr = b3InvMulQuatJW( poseQ, xfQ );
	b3Vec3W absV = { b3AbsJW( qr.V.X ), b3AbsJW( qr.V.Y ), b3AbsJW( qr.V.Z ) };
	b3Vec3W arc = b3ModifiedCrossJW( absV, maxExtent );

	b3FloatW arcSquared = b3MulW( b3SplatW( 4.0f ), b3LengthSquaredCW( arc ) );
	b3FloatW recycle = b3AndW( closeEnough, b3LessThanW( arcSquared, b3MulW( slack, slack ) ) );

	b3StoreW( results->px, xfP.X );
	b3StoreW( results->py, xfP.Y );
	b3StoreW( results->pz, xfP.Z );
	b3StoreW( results->qx, xfQ.V.X );
	b3StoreW( results->qy, xfQ.V.Y );
	b3StoreW( results->qz, xfQ.V.Z );
	b3StoreW( results->qs, xfQ.S );
	b3StoreW( results->slack, slack );
	b3StoreW( results->arcSquared, arcSquared );
	b3StoreW( results->angularDistance, angularDistance );

	return b3MoveMaskW( recycle );
}

// The distance test of b3CollideCapsules for one pair of capsules per lane.
// Returns a bit per lane that is set if b3CollideCapsules returns no points because of that test.
int B3_WIDE( b3TestCapsuleSeparation )( const b3CapsuleLanes* lanes )
{
	b3FloatW unused;
	B3_UNUSED( unused );

	b3Vec3W a1, a2, b1Local, b2Local;
	b3FloatW radiusA, radiusB;
	{
		const float* rowsA[B3_SIMD_WIDTH];
		const float* rowsB[B3_SIMD_WIDTH];
		for ( int i = 0; i < B3_SIMD_WIDTH; ++i )
		{
			rowsA[i] = (const float*)&lanes->shapesA[i]->capsule;
			rowsB[i] = (const float*)&lanes->shapesB[i]->capsule;
		}

		B3_GATHER_ROWS( rowsA, a1.X, a1.Y, a1.Z, a2.X, a2.Y, a2.Z, radiusA, unused );
		B3_GATHER_ROWS( rowsB, b1Local.X, b1Local.Y, b1Local.Z, b2Local.X, b2Local.Y, b2Local.Z, radiusB, unused );
	}

	// Pose of shape B in the frame of shape A
	b3Vec3W p = { b3LoadW( lanes->px ), b3LoadW( lanes->py ), b3LoadW( lanes->pz ) };
	b3QuatW q = { { b3LoadW( lanes->qx ), b3LoadW( lanes->qy ), b3LoadW( lanes->qz ) }, b3LoadW( lanes->qs ) };

	// b3TransformPoint
	b3Vec3W b1 = b3AddVW( b3RotateVectorJW( q, b1Local ), p );
	b3Vec3W b2 = b3AddVW( b3RotateVectorJW( q, b2Local ), p );

	b3FloatW radius = b3AddW( radiusA, radiusB );
	b3FloatW maxDistance = b3AddW( radius, b3SplatW( B3_SPECULATIVE_DISTANCE ) );

	// b3SegmentDistance( a1, a2, b1, b2 ). The scalar code returns early for degenerate segments. Here
	// all cases are computed and the right one is selected, which gives the same points.
	b3FloatW zero = b3ZeroW();
	b3FloatW one = b3SplatW( 1.0f );

	b3Vec3W d1 = b3SubVW( a2, a1 );
	b3Vec3W d2 = b3SubVW( b2, b1 );
	b3Vec3W r = b3SubVW( a1, b1 );

	b3FloatW a = b3DotW( d1, d1 );
	b3FloatW b = b3DotW( d1, d2 );
	b3FloatW c = b3DotW( d1, r );
	b3FloatW e = b3DotW( d2, d2 );
	b3FloatW f = b3DotW( d2, r );

	b3FloatW epsilon = b3SplatW( 100.0f * FLT_EPSILON );
	b3FloatW pointA = b3LessThanW( a, epsilon );
	b3FloatW pointB = b3LessThanW( e, epsilon );

	// Both segments have length
	b3FloatW denom = b3SubW( b3MulW( a, e ), b3MulW( b, b ) );
	b3FloatW s1 = b3ClampJW( b3DivW( b3SubW( b3MulW( b, f ), b3MulW( c, e ) ), denom ), zero, one );
	s1 = b3BlendW( zero, s1, b3GreaterThanW( denom, b3SplatW( 1000.0f * FLT_MIN ) ) );
	b3FloatW s2 = b3DivW( b3AddW( b3MulW( b, s1 ), f ), e );

	// Clamp the fraction on segment B and recompute the fraction on segment A if necessary
	b3FloatW s1Start = b3ClampJW( b3DivW( b3NegW( c ), a ), zero, one );
	b3FloatW s1End = b3ClampJW( b3DivW( b3SubW( b, c ), a ), zero, one );
	b3FloatW before = b3LessThanW( s2, zero );
	b3FloatW after = b3GreaterThanW( s2, one );
	s1 = b3BlendW( b3BlendW( s1, s1End, after ), s1Start, before );
	s2 = b3BlendW( b3BlendW( s2, one, after ), zero, before );

	// Segment B is a point: the closest point on segment A to that point
	s1 = b3BlendW( s1, s1Start, pointB );

	// Segment A is a point: the closest point on segment B to that point
	s2 = b3BlendW( s2, b3ClampJW( b3DivW( f, e ), zero, one ), pointA );

	// A segment that is a point returns its start, which also covers both being points
	b3Vec3W point1 = b3BlendVW( b3MulAddSVW( a1, s1, d1 ), a1, pointA );
	b3Vec3W point2 = b3BlendVW( b3MulAddSVW( b1, s2, d2 ), b1, pointB );

	b3FloatW distanceSquared = b3LengthSquaredCW( b3SubVW( point2, point1 ) );

	float linearSlop = B3_LINEAR_SLOP;
	float minDistance = 0.01f * linearSlop;

	b3FloatW separated = b3OrW( b3GreaterThanW( distanceSquared, b3MulW( maxDistance, maxDistance ) ),
								b3LessThanW( distanceSquared, b3SplatW( minDistance * minDistance ) ) );

	return b3MoveMaskW( separated );
}
