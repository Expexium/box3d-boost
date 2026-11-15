// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

// Wide solver for spherical and revolute joints. One joint per SIMD lane.
//
// The joints of a graph color don't share dynamic bodies, so they can be solved in any order, including
// all at once. This solves B3_SIMD_WIDTH joints at a time. Spherical and revolute joints are the joints
// of a ragdoll, so a pile of ragdolls spends much of its solver time here.
//
// This must produce results that are bit for bit identical to the scalar joint solver in
// spherical_joint.c and revolute_joint.c. That is what keeps the simulation deterministic across SIMD
// widths and platforms. To achieve this every function here is a direct transcription of the scalar code:
// - the same operations in the same order with the same grouping. Floating point addition is not
//   associative, so (a + b) + c must not become a + (b + c).
// - a scalar branch becomes a blend. Both sides are computed and one is selected per lane.
// - min, max, and clamp are spelled out with compares because the SIMD min/max of each platform treats
//   signed zero differently.
// - no fused multiply-add
// When changing the scalar solver the matching code here must change with it, and the reverse.
//
// The life of a wide joint during a time step:
// 1. b3PrepareWideJoints runs the scalar prepare on each joint of a group, which leaves the result in
//    the b3JointSim as before, and then transposes those joints into a wide joint
// 2. warm start and solve run on the wide joints each sub-step
// 3. b3StoreWideJoints writes the impulses back to each b3JointSim
//
// This file is included after contact_solver_wide.inl and shares its wide body state, gather, and scatter.

#include "body.h"
#include "constraint_graph.h"
#include "joint.h"
#include "physics_world.h"
#include "simd.h"
#include "simd_wide.h"
#include "solver.h"

#include <float.h>

// A lane without a joint reads from this. All its masks are off and it has no bodies.
static const b3JointSim b3_zeroJointSim = { 0 };

// Collect one float per lane from the joints of a group. FIELD is a member path of b3JointSim.
// This builds the wide value in registers. Going through a stack array would stall on store forwarding.
#if B3_SIMD_WIDTH == 8
#define B3_GATHER_JOINT( DST, FIELD )                                                                                            \
	( DST ) = b3SetW( sims[0]->FIELD, sims[1]->FIELD, sims[2]->FIELD, sims[3]->FIELD, sims[4]->FIELD, sims[5]->FIELD,            \
					  sims[6]->FIELD, sims[7]->FIELD )
#else
#define B3_GATHER_JOINT( DST, FIELD ) ( DST ) = b3SetW( sims[0]->FIELD, sims[1]->FIELD, sims[2]->FIELD, sims[3]->FIELD )
#endif

#define B3_GATHER_JOINT_V3( DST, FIELD )                                                                                         \
	do                                                                                                                           \
	{                                                                                                                            \
		B3_GATHER_JOINT( ( DST ).X, FIELD.x );                                                                                   \
		B3_GATHER_JOINT( ( DST ).Y, FIELD.y );                                                                                   \
		B3_GATHER_JOINT( ( DST ).Z, FIELD.z );                                                                                   \
	}                                                                                                                            \
	while ( 0 )

#define B3_GATHER_JOINT_QUAT( DST, FIELD )                                                                                       \
	do                                                                                                                           \
	{                                                                                                                            \
		B3_GATHER_JOINT( ( DST ).V.X, FIELD.v.x );                                                                               \
		B3_GATHER_JOINT( ( DST ).V.Y, FIELD.v.y );                                                                               \
		B3_GATHER_JOINT( ( DST ).V.Z, FIELD.v.z );                                                                               \
		B3_GATHER_JOINT( ( DST ).S, FIELD.s );                                                                                   \
	}                                                                                                                            \
	while ( 0 )

#define B3_GATHER_JOINT_M3( DST, FIELD )                                                                                         \
	do                                                                                                                           \
	{                                                                                                                            \
		B3_GATHER_JOINT_V3( ( DST ).cx, FIELD.cx );                                                                              \
		B3_GATHER_JOINT_V3( ( DST ).cy, FIELD.cy );                                                                              \
		B3_GATHER_JOINT_V3( ( DST ).cz, FIELD.cz );                                                                              \
	}                                                                                                                            \
	while ( 0 )

// A mask that is on in the lanes where the bool FIELD is true
#if B3_SIMD_WIDTH == 8
#define B3_GATHER_JOINT_MASK( DST, FIELD )                                                                                       \
	( DST ) = b3GreaterThanW( b3SetW( (float)sims[0]->FIELD, (float)sims[1]->FIELD, (float)sims[2]->FIELD,                       \
									  (float)sims[3]->FIELD, (float)sims[4]->FIELD, (float)sims[5]->FIELD,                       \
									  (float)sims[6]->FIELD, (float)sims[7]->FIELD ),                                            \
							  b3ZeroW() )
#else
#define B3_GATHER_JOINT_MASK( DST, FIELD )                                                                                       \
	( DST ) = b3GreaterThanW(                                                                                                    \
		b3SetW( (float)sims[0]->FIELD, (float)sims[1]->FIELD, (float)sims[2]->FIELD, (float)sims[3]->FIELD ), b3ZeroW() )
#endif

// Write a wide value to one float per joint. Unused lanes are skipped.
#define B3_SCATTER_JOINT( FIELD, SRC )                                                                                           \
	do                                                                                                                           \
	{                                                                                                                            \
		float lanes_[B3_SIMD_WIDTH];                                                                                             \
		b3StoreW( lanes_, ( SRC ) );                                                                                             \
		for ( int lane_ = 0; lane_ < B3_SIMD_WIDTH; ++lane_ )                                                                    \
		{                                                                                                                        \
			if ( joints[lane_] != NULL )                                                                                         \
			{                                                                                                                    \
				joints[lane_]->FIELD = lanes_[lane_];                                                                            \
			}                                                                                                                    \
		}                                                                                                                        \
	}                                                                                                                            \
	while ( 0 )

#define B3_SCATTER_JOINT_V3( FIELD, SRC )                                                                                        \
	do                                                                                                                           \
	{                                                                                                                            \
		B3_SCATTER_JOINT( FIELD.x, ( SRC ).X );                                                                                  \
		B3_SCATTER_JOINT( FIELD.y, ( SRC ).Y );                                                                                  \
		B3_SCATTER_JOINT( FIELD.z, ( SRC ).Z );                                                                                  \
	}                                                                                                                            \
	while ( 0 )

// Write one lane of a wide value to a scalar
#define B3_STORE_LANE( DST, SRC, LANE )                                                                                          \
	do                                                                                                                           \
	{                                                                                                                            \
		float lanes_[B3_SIMD_WIDTH];                                                                                             \
		b3StoreW( lanes_, ( SRC ) );                                                                                             \
		( DST ) = lanes_[( LANE )];                                                                                              \
	}                                                                                                                            \
	while ( 0 )

#define B3_STORE_LANE_V3( DST, SRC, LANE )                                                                                       \
	do                                                                                                                           \
	{                                                                                                                            \
		B3_STORE_LANE( ( DST ).x, ( SRC ).X, LANE );                                                                             \
		B3_STORE_LANE( ( DST ).y, ( SRC ).Y, LANE );                                                                             \
		B3_STORE_LANE( ( DST ).z, ( SRC ).Z, LANE );                                                                             \
	}                                                                                                                            \
	while ( 0 )

// The scalar counterparts are named in the comment above each function.

// b3MaxFloat: a > b ? a : b
static inline b3FloatW b3MaxJW( b3FloatW a, b3FloatW b )
{
	return b3BlendW( b, a, b3GreaterThanW( a, b ) );
}

// b3MinFloat: a < b ? a : b
static inline b3FloatW b3MinJW( b3FloatW a, b3FloatW b )
{
	return b3BlendW( b, a, b3LessThanW( a, b ) );
}

// b3ClampFloat: a < lower ? lower : ( upper < a ? upper : a )
static inline b3FloatW b3ClampJW( b3FloatW a, b3FloatW lower, b3FloatW upper )
{
	b3FloatW inner = b3BlendW( a, upper, b3LessThanW( upper, a ) );
	return b3BlendW( inner, lower, b3LessThanW( a, lower ) );
}

// mask ? b : a
static inline b3Vec3W b3BlendVW( b3Vec3W a, b3Vec3W b, b3FloatW mask )
{
	return (b3Vec3W){ b3BlendW( a.X, b.X, mask ), b3BlendW( a.Y, b.Y, mask ), b3BlendW( a.Z, b.Z, mask ) };
}

// b3AddMM
static inline b3Matrix3W b3AddMMJW( b3Matrix3W a, b3Matrix3W b )
{
	return (b3Matrix3W){ b3AddVW( a.cx, b.cx ), b3AddVW( a.cy, b.cy ), b3AddVW( a.cz, b.cz ) };
}

// b3RotateVector
static inline b3Vec3W b3RotateVectorJW( b3QuatW q, b3Vec3W v )
{
	b3Vec3W t1 = b3CrossW( q.V, v );
	b3Vec3W t2 = b3MulAddSVW( t1, q.S, v );
	b3Vec3W t3 = b3CrossW( q.V, t2 );
	return b3MulAddSVW( v, b3SplatW( 2.0f ), t3 );
}

// b3InvMulQuat
static inline b3QuatW b3InvMulQuatJW( b3QuatW q1, b3QuatW q2 )
{
	b3Vec3W t1 = b3CrossW( q2.V, q1.V );
	b3Vec3W t2 = b3MulAddSVW( t1, q1.S, q2.V );
	b3Vec3W t3 = b3MulSubSVW( t2, q2.S, q1.V );
	return (b3QuatW){ t3, b3AddW( b3MulW( q1.S, q2.S ), b3DotW( q1.V, q2.V ) ) };
}

// b3DotQuat
static inline b3FloatW b3DotQuatJW( b3QuatW a, b3QuatW b )
{
	return b3AddW( b3AddW( b3AddW( b3MulW( a.V.X, b.V.X ), b3MulW( a.V.Y, b.V.Y ) ), b3MulW( a.V.Z, b.V.Z ) ),
				   b3MulW( a.S, b.S ) );
}

// b3NegateQuat
static inline b3QuatW b3NegateQuatJW( b3QuatW q )
{
	return (b3QuatW){ b3NegVW( q.V ), b3NegW( q.S ) };
}

// b3Atan2
static inline b3FloatW b3Atan2JW( b3FloatW y, b3FloatW x )
{
	b3FloatW zero = b3ZeroW();

	// b3AbsFloat( -0 ) is -0 while this gives +0, which only matters below where both compare the same
	b3FloatW ax = b3AbsW( x );
	b3FloatW ay = b3AbsW( y );
	b3FloatW mx = b3MaxJW( ay, ax );
	b3FloatW mn = b3MinJW( ay, ax );
	b3FloatW a = b3DivW( mn, mx );

	// Minimax polynomial approximation to atan(a) on [0,1]
	b3FloatW s = b3MulW( a, a );
	b3FloatW c = b3MulW( s, a );
	b3FloatW q = b3MulW( s, s );
	b3FloatW r = b3AddW( b3MulW( b3SplatW( 0.024840285f ), q ), b3SplatW( 0.18681418f ) );
	b3FloatW t = b3SubW( b3MulW( b3SplatW( -0.094097948f ), q ), b3SplatW( 0.33213072f ) );
	r = b3AddW( b3MulW( r, s ), t );
	r = b3AddW( b3MulW( r, c ), a );

	// Map to full circle
	r = b3BlendW( r, b3SubW( b3SplatW( 1.57079637f ), r ), b3GreaterThanW( ay, ax ) );
	r = b3BlendW( r, b3SubW( b3SplatW( 3.14159274f ), r ), b3LessThanW( x, zero ) );
	r = b3BlendW( r, b3NegW( r ), b3LessThanW( y, zero ) );

	// The scalar version returns zero up front for (0,0)
	b3FloatW origin = b3AndW( b3EqualsW( x, zero ), b3EqualsW( y, zero ) );
	return b3BlendW( r, zero, origin );
}

// b3GetTwistAngle
static inline b3FloatW b3GetTwistAngleJW( b3QuatW q )
{
	b3FloatW negative = b3LessThanW( q.S, b3ZeroW() );
	b3FloatW y = b3BlendW( q.V.Z, b3NegW( q.V.Z ), negative );
	b3FloatW x = b3BlendW( q.S, b3NegW( q.S ), negative );
	return b3MulW( b3Atan2JW( y, x ), b3SplatW( 2.0f ) );
}

// b3GetSwingAngle
static inline b3FloatW b3GetSwingAngleJW( b3QuatW q )
{
	b3FloatW x = b3SqrtW( b3AddW( b3MulW( q.V.Z, q.V.Z ), b3MulW( q.S, q.S ) ) );
	b3FloatW y = b3SqrtW( b3AddW( b3MulW( q.V.X, q.V.X ), b3MulW( q.V.Y, q.V.Y ) ) );
	return b3MulW( b3SplatW( 2.0f ), b3Atan2JW( y, x ) );
}

// b3DeltaQuatToRotation
static inline b3Vec3W b3DeltaQuatToRotationJW( b3QuatW q, b3QuatW target )
{
	// Correct polarity
	b3FloatW flip = b3LessThanW( b3DotQuatJW( q, target ), b3ZeroW() );
	b3QuatW s;
	s.V = b3BlendVW( q.V, b3NegVW( q.V ), flip );
	s.S = b3BlendW( q.S, b3NegW( q.S ), flip );

	b3QuatW diff = { b3SubVW( target.V, s.V ), b3SubW( target.S, s.S ) };
	b3QuatW conjugate = { b3NegVW( s.V ), s.S };
	b3QuatW product = b3MulQuatJW( diff, conjugate );
	return b3MulSVW( b3SplatW( 2.0f ), product.V );
}

// b3SolvePointConstraint
static inline b3Vec3W b3SolvePointConstraintJW( b3FloatW mA, b3FloatW mB, b3Matrix3W iA, b3Matrix3W iB, b3Vec3W rA, b3Vec3W rB,
												b3Vec3W rhs )
{
	b3FloatW axy = b3SubW( b3MulW( rA.Z, iA.cy.Y ), b3MulW( rA.Y, iA.cz.Y ) );
	b3FloatW axz = b3SubW( b3MulW( rA.Z, iA.cy.Z ), b3MulW( rA.Y, iA.cz.Z ) );
	b3Vec3W ay = b3SubVW( b3MulSVW( rA.X, iA.cz ), b3MulSVW( rA.Z, iA.cx ) );
	b3Vec3W az = b3SubVW( b3MulSVW( rA.Y, iA.cx ), b3MulSVW( rA.X, iA.cy ) );

	b3FloatW bxy = b3SubW( b3MulW( rB.Z, iB.cy.Y ), b3MulW( rB.Y, iB.cz.Y ) );
	b3FloatW bxz = b3SubW( b3MulW( rB.Z, iB.cy.Z ), b3MulW( rB.Y, iB.cz.Z ) );
	b3Vec3W by = b3SubVW( b3MulSVW( rB.X, iB.cz ), b3MulSVW( rB.Z, iB.cx ) );
	b3Vec3W bz = b3SubVW( b3MulSVW( rB.Y, iB.cx ), b3MulSVW( rB.X, iB.cy ) );

	b3FloatW m = b3AddW( mA, mB );
	b3FloatW kxx = b3AddW( b3AddW( b3SubW( b3MulW( axy, rA.Z ), b3MulW( axz, rA.Y ) ),
								   b3SubW( b3MulW( bxy, rB.Z ), b3MulW( bxz, rB.Y ) ) ),
						   m );
	b3FloatW kyy = b3AddW( b3AddW( b3SubW( b3MulW( ay.Z, rA.X ), b3MulW( ay.X, rA.Z ) ),
								   b3SubW( b3MulW( by.Z, rB.X ), b3MulW( by.X, rB.Z ) ) ),
						   m );
	b3FloatW kzz = b3AddW( b3AddW( b3SubW( b3MulW( az.X, rA.Y ), b3MulW( az.Y, rA.X ) ),
								   b3SubW( b3MulW( bz.X, rB.Y ), b3MulW( bz.Y, rB.X ) ) ),
						   m );
	b3FloatW kxy = b3AddW( b3SubW( b3MulW( ay.Y, rA.Z ), b3MulW( ay.Z, rA.Y ) ),
						   b3SubW( b3MulW( by.Y, rB.Z ), b3MulW( by.Z, rB.Y ) ) );
	b3FloatW kxz = b3AddW( b3SubW( b3MulW( az.Y, rA.Z ), b3MulW( az.Z, rA.Y ) ),
						   b3SubW( b3MulW( bz.Y, rB.Z ), b3MulW( bz.Z, rB.Y ) ) );
	b3FloatW kyz = b3AddW( b3SubW( b3MulW( az.Z, rA.X ), b3MulW( az.X, rA.Z ) ),
						   b3SubW( b3MulW( bz.Z, rB.X ), b3MulW( bz.X, rB.Z ) ) );

	b3FloatW cxx = b3SubW( b3MulW( kyy, kzz ), b3MulW( kyz, kyz ) );
	b3FloatW cyy = b3SubW( b3MulW( kxx, kzz ), b3MulW( kxz, kxz ) );
	b3FloatW czz = b3SubW( b3MulW( kxx, kyy ), b3MulW( kxy, kxy ) );
	b3FloatW cxy = b3SubW( b3MulW( kxz, kyz ), b3MulW( kxy, kzz ) );
	b3FloatW cxz = b3SubW( b3MulW( kxy, kyz ), b3MulW( kyy, kxz ) );
	b3FloatW cyz = b3SubW( b3MulW( kxy, kxz ), b3MulW( kxx, kyz ) );

	b3FloatW det = b3AddW( b3AddW( b3MulW( kxx, cxx ), b3MulW( kxy, cxy ) ), b3MulW( kxz, cxz ) );
	b3FloatW valid = b3GreaterThanW( b3AbsW( det ), b3SplatW( 1000.0f * FLT_MIN ) );
	b3FloatW invDet = b3DivW( b3SplatW( 1.0f ), det );

	b3Vec3W b = {
		b3MulW( invDet, b3AddW( b3AddW( b3MulW( cxx, rhs.X ), b3MulW( cxy, rhs.Y ) ), b3MulW( cxz, rhs.Z ) ) ),
		b3MulW( invDet, b3AddW( b3AddW( b3MulW( cxy, rhs.X ), b3MulW( cyy, rhs.Y ) ), b3MulW( cyz, rhs.Z ) ) ),
		b3MulW( invDet, b3AddW( b3AddW( b3MulW( cxz, rhs.X ), b3MulW( cyz, rhs.Y ) ), b3MulW( czz, rhs.Z ) ) ),
	};

	b3FloatW zero = b3ZeroW();
	return (b3Vec3W){ b3BlendW( zero, b.X, valid ), b3BlendW( zero, b.Y, valid ), b3BlendW( zero, b.Z, valid ) };
}

// Data shared by all wide joint types
typedef struct B3_WIDE( b3JointBase )
{
	// Body state indices. These are base 1, zero is a static body or an unused lane.
	_Alignas( B3_WIDE_ALIGNMENT ) int indexA[B3_SIMD_WIDTH];
	int indexB[B3_SIMD_WIDTH];

	// Null for an unused lane
	b3JointSim* joints[B3_SIMD_WIDTH];

	b3FloatW invMassA, invMassB;
	b3Matrix3W invIA, invIB;

	// Constraint softness
	b3FloatW biasRate;
	b3FloatW massScale;
	b3FloatW impulseScale;

	// Mask
	b3FloatW fixedRotation;

	// Bit per lane for joints that report joint events
	int eventLanes;
} b3JointBaseWide;

typedef struct B3_WIDE( b3SphericalJoint )
{
	b3JointBaseWide base;

	b3QuatW frameAq, frameBq;
	b3Vec3W frameAp, frameBp;
	b3Vec3W deltaCenter;
	b3Vec3W swingAxis;
	b3Vec3W twistJacobian;
	b3Matrix3W rotationMass;
	b3FloatW swingMass;
	b3FloatW twistMass;

	// Spring softness
	b3FloatW springBiasRate;
	b3FloatW springMassScale;
	b3FloatW springImpulseScale;

	b3QuatW targetRotation;
	b3Vec3W motorVelocity;
	b3FloatW maxMotorTorque;
	b3FloatW lowerTwistAngle;
	b3FloatW upperTwistAngle;
	b3FloatW coneAngle;

	b3Vec3W linearImpulse;
	b3Vec3W springImpulse;
	b3Vec3W motorImpulse;
	b3FloatW lowerTwistImpulse;
	b3FloatW upperTwistImpulse;
	b3FloatW swingImpulse;

	// Masks
	b3FloatW enableSpring;
	b3FloatW enableMotor;
	b3FloatW enableConeLimit;
	b3FloatW enableTwistLimit;
} b3SphericalJointWide;

typedef struct B3_WIDE( b3RevoluteJoint )
{
	b3JointBaseWide base;

	b3QuatW frameAq, frameBq;
	b3Vec3W frameAp, frameBp;
	b3Vec3W deltaCenter;
	b3Vec3W rotationAxisZ;
	b3Vec3W perpAxisX;
	b3Vec3W perpAxisY;
	b3FloatW axialMass;

	// Spring softness
	b3FloatW springBiasRate;
	b3FloatW springMassScale;
	b3FloatW springImpulseScale;

	b3FloatW targetAngle;
	b3FloatW lowerAngle;
	b3FloatW upperAngle;
	b3FloatW maxMotorTorque;
	b3FloatW motorSpeed;

	b3Vec3W linearImpulse;
	b3FloatW perpImpulseX;
	b3FloatW perpImpulseY;
	b3FloatW springImpulse;
	b3FloatW motorImpulse;
	b3FloatW lowerImpulse;
	b3FloatW upperImpulse;

	// Masks
	b3FloatW enableSpring;
	b3FloatW enableMotor;
	b3FloatW enableLimit;
} b3RevoluteJointWide;

int B3_WIDE( b3GetWideSphericalJointByteCount )( void )
{
	return sizeof( b3SphericalJointWide );
}

int B3_WIDE( b3GetWideRevoluteJointByteCount )( void )
{
	return sizeof( b3RevoluteJointWide );
}

// Transpose the data shared by all joint types. The index of each lane comes from the typed joint data.
static void b3PrepareJointBaseW( b3JointBaseWide* w, b3JointSim* const* joints, const b3JointSim* const* sims )
{
	int eventLanes = 0;
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		b3JointSim* joint = joints[lane];
		w->joints[lane] = joint;

		if ( joint != NULL && ( joint->forceThreshold < FLT_MAX || joint->torqueThreshold < FLT_MAX ) )
		{
			eventLanes |= 1 << lane;
		}
	}

	w->eventLanes = eventLanes;

	B3_GATHER_JOINT( w->invMassA, invMassA );
	B3_GATHER_JOINT( w->invMassB, invMassB );
	B3_GATHER_JOINT_M3( w->invIA, invIA );
	B3_GATHER_JOINT_M3( w->invIB, invIB );
	B3_GATHER_JOINT( w->biasRate, constraintSoftness.biasRate );
	B3_GATHER_JOINT( w->massScale, constraintSoftness.massScale );
	B3_GATHER_JOINT( w->impulseScale, constraintSoftness.impulseScale );
	B3_GATHER_JOINT_MASK( w->fixedRotation, fixedRotation );
}

static void b3PrepareSphericalJointW( b3SphericalJointWide* w, b3JointSim* const* joints )
{
	const b3JointSim* sims[B3_SIMD_WIDTH];
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		const b3JointSim* joint = joints[lane];
		B3_ASSERT( joint == NULL || joint->type == b3_sphericalJoint );
		sims[lane] = joint != NULL ? joint : &b3_zeroJointSim;

		// The zero joint has index zero, which would be a real body
		w->base.indexA[lane] = joint != NULL ? joint->sphericalJoint.indexA + 1 : 0;
		w->base.indexB[lane] = joint != NULL ? joint->sphericalJoint.indexB + 1 : 0;
	}

	b3PrepareJointBaseW( &w->base, joints, sims );

	B3_GATHER_JOINT_QUAT( w->frameAq, sphericalJoint.frameA.q );
	B3_GATHER_JOINT_QUAT( w->frameBq, sphericalJoint.frameB.q );
	B3_GATHER_JOINT_V3( w->frameAp, sphericalJoint.frameA.p );
	B3_GATHER_JOINT_V3( w->frameBp, sphericalJoint.frameB.p );
	B3_GATHER_JOINT_V3( w->deltaCenter, sphericalJoint.deltaCenter );
	B3_GATHER_JOINT_V3( w->swingAxis, sphericalJoint.swingAxis );
	B3_GATHER_JOINT_V3( w->twistJacobian, sphericalJoint.twistJacobian );
	B3_GATHER_JOINT_M3( w->rotationMass, sphericalJoint.rotationMass );
	B3_GATHER_JOINT( w->swingMass, sphericalJoint.swingMass );
	B3_GATHER_JOINT( w->twistMass, sphericalJoint.twistMass );
	B3_GATHER_JOINT( w->springBiasRate, sphericalJoint.springSoftness.biasRate );
	B3_GATHER_JOINT( w->springMassScale, sphericalJoint.springSoftness.massScale );
	B3_GATHER_JOINT( w->springImpulseScale, sphericalJoint.springSoftness.impulseScale );
	B3_GATHER_JOINT_QUAT( w->targetRotation, sphericalJoint.targetRotation );
	B3_GATHER_JOINT_V3( w->motorVelocity, sphericalJoint.motorVelocity );
	B3_GATHER_JOINT( w->maxMotorTorque, sphericalJoint.maxMotorTorque );
	B3_GATHER_JOINT( w->lowerTwistAngle, sphericalJoint.lowerTwistAngle );
	B3_GATHER_JOINT( w->upperTwistAngle, sphericalJoint.upperTwistAngle );
	B3_GATHER_JOINT( w->coneAngle, sphericalJoint.coneAngle );
	B3_GATHER_JOINT_V3( w->linearImpulse, sphericalJoint.linearImpulse );
	B3_GATHER_JOINT_V3( w->springImpulse, sphericalJoint.springImpulse );
	B3_GATHER_JOINT_V3( w->motorImpulse, sphericalJoint.motorImpulse );
	B3_GATHER_JOINT( w->lowerTwistImpulse, sphericalJoint.lowerTwistImpulse );
	B3_GATHER_JOINT( w->upperTwistImpulse, sphericalJoint.upperTwistImpulse );
	B3_GATHER_JOINT( w->swingImpulse, sphericalJoint.swingImpulse );
	B3_GATHER_JOINT_MASK( w->enableSpring, sphericalJoint.enableSpring );
	B3_GATHER_JOINT_MASK( w->enableMotor, sphericalJoint.enableMotor );
	B3_GATHER_JOINT_MASK( w->enableConeLimit, sphericalJoint.enableConeLimit );
	B3_GATHER_JOINT_MASK( w->enableTwistLimit, sphericalJoint.enableTwistLimit );
}

static void b3PrepareRevoluteJointW( b3RevoluteJointWide* w, b3JointSim* const* joints )
{
	const b3JointSim* sims[B3_SIMD_WIDTH];
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		const b3JointSim* joint = joints[lane];
		B3_ASSERT( joint == NULL || joint->type == b3_revoluteJoint );
		sims[lane] = joint != NULL ? joint : &b3_zeroJointSim;

		// The zero joint has index zero, which would be a real body
		w->base.indexA[lane] = joint != NULL ? joint->revoluteJoint.indexA + 1 : 0;
		w->base.indexB[lane] = joint != NULL ? joint->revoluteJoint.indexB + 1 : 0;
	}

	b3PrepareJointBaseW( &w->base, joints, sims );

	B3_GATHER_JOINT_QUAT( w->frameAq, revoluteJoint.frameA.q );
	B3_GATHER_JOINT_QUAT( w->frameBq, revoluteJoint.frameB.q );
	B3_GATHER_JOINT_V3( w->frameAp, revoluteJoint.frameA.p );
	B3_GATHER_JOINT_V3( w->frameBp, revoluteJoint.frameB.p );
	B3_GATHER_JOINT_V3( w->deltaCenter, revoluteJoint.deltaCenter );
	B3_GATHER_JOINT_V3( w->rotationAxisZ, revoluteJoint.rotationAxisZ );
	B3_GATHER_JOINT_V3( w->perpAxisX, revoluteJoint.perpAxisX );
	B3_GATHER_JOINT_V3( w->perpAxisY, revoluteJoint.perpAxisY );
	B3_GATHER_JOINT( w->axialMass, revoluteJoint.axialMass );
	B3_GATHER_JOINT( w->springBiasRate, revoluteJoint.springSoftness.biasRate );
	B3_GATHER_JOINT( w->springMassScale, revoluteJoint.springSoftness.massScale );
	B3_GATHER_JOINT( w->springImpulseScale, revoluteJoint.springSoftness.impulseScale );
	B3_GATHER_JOINT( w->targetAngle, revoluteJoint.targetAngle );
	B3_GATHER_JOINT( w->lowerAngle, revoluteJoint.lowerAngle );
	B3_GATHER_JOINT( w->upperAngle, revoluteJoint.upperAngle );
	B3_GATHER_JOINT( w->maxMotorTorque, revoluteJoint.maxMotorTorque );
	B3_GATHER_JOINT( w->motorSpeed, revoluteJoint.motorSpeed );
	B3_GATHER_JOINT_V3( w->linearImpulse, revoluteJoint.linearImpulse );
	B3_GATHER_JOINT( w->perpImpulseX, revoluteJoint.perpImpulse.x );
	B3_GATHER_JOINT( w->perpImpulseY, revoluteJoint.perpImpulse.y );
	B3_GATHER_JOINT( w->springImpulse, revoluteJoint.springImpulse );
	B3_GATHER_JOINT( w->motorImpulse, revoluteJoint.motorImpulse );
	B3_GATHER_JOINT( w->lowerImpulse, revoluteJoint.lowerImpulse );
	B3_GATHER_JOINT( w->upperImpulse, revoluteJoint.upperImpulse );
	B3_GATHER_JOINT_MASK( w->enableSpring, revoluteJoint.enableSpring );
	B3_GATHER_JOINT_MASK( w->enableMotor, revoluteJoint.enableMotor );
	B3_GATHER_JOINT_MASK( w->enableLimit, revoluteJoint.enableLimit );
}

// Write the solver results of one lane back to its joint
static void b3StoreSphericalJointLaneW( const b3SphericalJointWide* w, int lane )
{
	b3SphericalJoint* joint = &w->base.joints[lane]->sphericalJoint;
	B3_STORE_LANE_V3( joint->linearImpulse, w->linearImpulse, lane );
	B3_STORE_LANE_V3( joint->springImpulse, w->springImpulse, lane );
	B3_STORE_LANE_V3( joint->motorImpulse, w->motorImpulse, lane );
	B3_STORE_LANE( joint->lowerTwistImpulse, w->lowerTwistImpulse, lane );
	B3_STORE_LANE( joint->upperTwistImpulse, w->upperTwistImpulse, lane );
	B3_STORE_LANE( joint->swingImpulse, w->swingImpulse, lane );
}

// Write the solver results of one lane back to its joint. The scalar solver also updates the axes.
static void b3StoreRevoluteJointLaneW( const b3RevoluteJointWide* w, int lane )
{
	b3RevoluteJoint* joint = &w->base.joints[lane]->revoluteJoint;
	B3_STORE_LANE_V3( joint->linearImpulse, w->linearImpulse, lane );
	B3_STORE_LANE( joint->perpImpulse.x, w->perpImpulseX, lane );
	B3_STORE_LANE( joint->perpImpulse.y, w->perpImpulseY, lane );
	B3_STORE_LANE( joint->springImpulse, w->springImpulse, lane );
	B3_STORE_LANE( joint->motorImpulse, w->motorImpulse, lane );
	B3_STORE_LANE( joint->lowerImpulse, w->lowerImpulse, lane );
	B3_STORE_LANE( joint->upperImpulse, w->upperImpulse, lane );
	B3_STORE_LANE_V3( joint->perpAxisX, w->perpAxisX, lane );
	B3_STORE_LANE_V3( joint->perpAxisY, w->perpAxisY, lane );
}

// Flag a joint for a joint event if its reaction exceeds a threshold. Same test as the scalar joint task.
// The impulses of the joint must already be stored.
static void b3CheckJointEventW( b3JointSim* joint, b3StepContext* context, b3BitSet* jointStateBitSet )
{
	if ( b3GetBit( jointStateBitSet, joint->jointId ) )
	{
		return;
	}

	float force, torque;
	b3GetJointReaction( context->world, joint, context->inv_h, &force, &torque );

	// Check thresholds. A zero threshold means all awake joints get reported.
	if ( force >= joint->forceThreshold || torque >= joint->torqueThreshold )
	{
		// Flag this joint for processing.
		b3SetBit( jointStateBitSet, joint->jointId );
	}
}

// Preparing a joint reads both bodies and then both body sims. These are scattered in memory, so the
// prepare loop pulls them in ahead of time, the bodies of a group first and their sims one group later.
static inline void b3PrefetchJointBodiesW( b3JointSim* const* joints, const b3Body* bodies )
{
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		const b3JointSim* joint = joints[lane];
		if ( joint != NULL )
		{
			b3Prefetch( bodies + joint->bodyIdA );
			b3Prefetch( bodies + joint->bodyIdB );
		}
	}
}

static inline void b3PrefetchBodySimW( const char* sim )
{
	for ( int offset = 0; offset < (int)sizeof( b3BodySim ); offset += B3_CACHE_LINE_SIZE )
	{
		b3Prefetch( sim + offset );
	}
}

static inline void b3PrefetchJointBodySimsW( b3JointSim* const* joints, b3World* world )
{
	const b3Body* bodies = world->bodies.data;
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		const b3JointSim* joint = joints[lane];
		if ( joint != NULL )
		{
			// A joint in the constraint graph only has awake and static bodies
			const b3Body* bodyA = bodies + joint->bodyIdA;
			const b3Body* bodyB = bodies + joint->bodyIdB;
			if ( bodyA->setIndex < b3_firstSleepingSet && bodyB->setIndex < b3_firstSleepingSet )
			{
				b3PrefetchBodySimW( (const char*)( world->fixedSets[bodyA->setIndex].bodySims.data + bodyA->localIndex ) );
				b3PrefetchBodySimW( (const char*)( world->fixedSets[bodyB->setIndex].bodySims.data + bodyB->localIndex ) );
			}
		}
	}
}

// The wide joints of all colors are in two flat arrays, spherical then revolute. The block is a range
// of groups across both arrays.
void B3_WIDE( b3PrepareWideJoints )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( prepare_wide_joints, "PrepWideJoints", b3_colorOldLace, true );

	b3World* world = context->world;
	const b3Body* bodies = world->bodies.data;
	b3JointSim* const* groupJoints = context->wideJointSims;
	b3SphericalJointWide* sphericalJoints = context->wideSphericalJoints;
	b3RevoluteJointWide* revoluteJoints = context->wideRevoluteJoints;
	int sphericalCount = context->wideSphericalCount;
	int endIndex = block.startIndex + block.count;

	for ( int i = block.startIndex; i < endIndex; ++i )
	{
		b3JointSim* const* joints = groupJoints + B3_SIMD_WIDTH * i;

		if ( i + 2 < endIndex )
		{
			b3PrefetchJointBodiesW( joints + 2 * B3_SIMD_WIDTH, bodies );
		}

		if ( i + 1 < endIndex )
		{
			b3PrefetchJointBodySimsW( joints + B3_SIMD_WIDTH, world );
		}

		// Scalar prepare, then transpose while the joints are still in the cache
		for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
		{
			if ( joints[lane] != NULL )
			{
				b3PrepareJoint( joints[lane], context );
			}
		}

		if ( i < sphericalCount )
		{
			b3PrepareSphericalJointW( sphericalJoints + i, joints );
		}
		else
		{
			b3PrepareRevoluteJointW( revoluteJoints + ( i - sphericalCount ), joints );
		}
	}

	b3TracyCZoneEnd( prepare_wide_joints );
}

// Write the solver results of a whole group back to its joints
static void b3StoreSphericalJointW( const b3SphericalJointWide* w )
{
	b3JointSim* const* joints = w->base.joints;
	B3_SCATTER_JOINT_V3( sphericalJoint.linearImpulse, w->linearImpulse );
	B3_SCATTER_JOINT_V3( sphericalJoint.springImpulse, w->springImpulse );
	B3_SCATTER_JOINT_V3( sphericalJoint.motorImpulse, w->motorImpulse );
	B3_SCATTER_JOINT( sphericalJoint.lowerTwistImpulse, w->lowerTwistImpulse );
	B3_SCATTER_JOINT( sphericalJoint.upperTwistImpulse, w->upperTwistImpulse );
	B3_SCATTER_JOINT( sphericalJoint.swingImpulse, w->swingImpulse );
}

// Write the solver results of a whole group back to its joints. The scalar solver also updates the axes.
static void b3StoreRevoluteJointW( const b3RevoluteJointWide* w )
{
	b3JointSim* const* joints = w->base.joints;
	B3_SCATTER_JOINT_V3( revoluteJoint.linearImpulse, w->linearImpulse );
	B3_SCATTER_JOINT( revoluteJoint.perpImpulse.x, w->perpImpulseX );
	B3_SCATTER_JOINT( revoluteJoint.perpImpulse.y, w->perpImpulseY );
	B3_SCATTER_JOINT( revoluteJoint.springImpulse, w->springImpulse );
	B3_SCATTER_JOINT( revoluteJoint.motorImpulse, w->motorImpulse );
	B3_SCATTER_JOINT( revoluteJoint.lowerImpulse, w->lowerImpulse );
	B3_SCATTER_JOINT( revoluteJoint.upperImpulse, w->upperImpulse );
	B3_SCATTER_JOINT_V3( revoluteJoint.perpAxisX, w->perpAxisX );
	B3_SCATTER_JOINT_V3( revoluteJoint.perpAxisY, w->perpAxisY );
}

void B3_WIDE( b3StoreWideJoints )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( store_wide_joints, "StoreWideJoints", b3_colorFireBrick, true );

	const b3SphericalJointWide* sphericalJoints = context->wideSphericalJoints;
	const b3RevoluteJointWide* revoluteJoints = context->wideRevoluteJoints;
	b3JointSim* const* groupJoints = context->wideJointSims;
	int sphericalCount = context->wideSphericalCount;
	int endIndex = block.startIndex + block.count;

	for ( int i = block.startIndex; i < endIndex; ++i )
	{
		// Pull in the part of the next joints that is written
		if ( i + 1 < endIndex )
		{
			b3JointSim* const* nextJoints = groupJoints + B3_SIMD_WIDTH * ( i + 1 );
			for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
			{
				const char* joint = (const char*)nextJoints[lane];
				if ( joint != NULL )
				{
					b3Prefetch( joint + offsetof( b3JointSim, revoluteJoint ) );
					b3Prefetch( joint + offsetof( b3JointSim, revoluteJoint ) + 63 );
					if ( i + 1 >= sphericalCount )
					{
						b3Prefetch( joint + offsetof( b3JointSim, revoluteJoint.perpAxisY ) + sizeof( b3Vec3 ) - 1 );
					}
				}
			}
		}

		if ( i < sphericalCount )
		{
			b3StoreSphericalJointW( sphericalJoints + i );
		}
		else
		{
			b3StoreRevoluteJointW( revoluteJoints + ( i - sphericalCount ) );
		}
	}

	b3TracyCZoneEnd( store_wide_joints );
}

// b3WarmStartSphericalJoint
void B3_WIDE( b3WarmStartSphericalJoints )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( warm_spherical, "WarmSpherical", b3_colorGold, true );

	b3BodyState* states = context->states;
	b3SphericalJointWide* joints = context->graph->colors[block.colorIndex].wideSphericalJoints;

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3SphericalJointWide* joint = joints + i;
		b3JointBaseWide* base = &joint->base;

		b3BodyStateW bA = b3GatherBodies( states, base->indexA );
		b3BodyStateW bB = b3GatherBodies( states, base->indexB );

		b3Vec3W rA = b3RotateVectorJW( bA.dq, joint->frameAp );
		b3Vec3W rB = b3RotateVectorJW( bB.dq, joint->frameBp );

		b3Vec3W angularImpulse = b3AddVW( joint->springImpulse, joint->motorImpulse );
		angularImpulse = b3MulSubSVW( angularImpulse, joint->swingImpulse, joint->swingAxis );
		angularImpulse =
			b3MulAddSVW( angularImpulse, b3SubW( joint->lowerTwistImpulse, joint->upperTwistImpulse ), joint->twistJacobian );

		bA.v = b3MulSubSVW( bA.v, base->invMassA, joint->linearImpulse );
		bA.w = b3SubVW( bA.w, b3MulMVJW( base->invIA, b3AddVW( b3CrossW( rA, joint->linearImpulse ), angularImpulse ) ) );

		bB.v = b3MulAddSVW( bB.v, base->invMassB, joint->linearImpulse );
		bB.w = b3AddVW( bB.w, b3MulMVJW( base->invIB, b3AddVW( b3CrossW( rB, joint->linearImpulse ), angularImpulse ) ) );

		b3ScatterBodies( states, base->indexA, &bA );
		b3ScatterBodies( states, base->indexB, &bB );
	}

	b3TracyCZoneEnd( warm_spherical );
}

// b3SolveSphericalJoint
void B3_WIDE( b3SolveSphericalJoints )( b3SolverBlock block, b3StepContext* context, bool useBias, int workerIndex )
{
	b3TracyCZoneNC( solve_spherical, "SolveSpherical", b3_colorLemonChiffon, true );

	b3BodyState* states = context->states;
	b3SphericalJointWide* joints = context->graph->colors[block.colorIndex].wideSphericalJoints;
	b3BitSet* jointStateBitSet = &context->world->taskContexts.data[workerIndex].jointStateBitSet;

	b3FloatW zero = b3ZeroW();
	b3FloatW one = b3SplatW( 1.0f );
	b3FloatW h = b3SplatW( context->h );
	b3FloatW inv_h = b3SplatW( context->inv_h );

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3SphericalJointWide* joint = joints + i;
		b3JointBaseWide* base = &joint->base;

		b3FloatW mA = base->invMassA;
		b3FloatW mB = base->invMassB;
		b3Matrix3W iA = base->invIA;
		b3Matrix3W iB = base->invIB;

		b3BodyStateW bA = b3GatherBodies( states, base->indexA );
		b3BodyStateW bB = b3GatherBodies( states, base->indexB );

		b3Vec3W vA = bA.v;
		b3Vec3W wA = bA.w;
		b3Vec3W vB = bB.v;
		b3Vec3W wB = bB.w;

		b3QuatW quatA = b3MulQuatJW( bA.dq, joint->frameAq );
		b3QuatW quatB = b3MulQuatJW( bB.dq, joint->frameBq );

		b3QuatW relQ = b3InvMulQuatJW( quatA, quatB );

		// The scalar solver skips this for a fixed rotation and when no angular feature is enabled
		b3FloatW anyAngular =
			b3OrW( b3OrW( joint->enableSpring, joint->enableMotor ), b3OrW( joint->enableTwistLimit, joint->enableConeLimit ) );
		b3FloatW angularMask = b3AndNotW( anyAngular, base->fixedRotation );

		if ( b3AnyTrueW( angularMask ) )
		{
			b3Matrix3W invInertiaSum = b3AddMMJW( iA, iB );
			b3Vec3W cdot = b3SubVW( wB, wA );
			b3Vec3W angularImpulse = { zero, zero, zero };

			// Solve spring
			b3FloatW springMask = b3AndW( angularMask, joint->enableSpring );
			if ( b3AnyTrueW( springMask ) )
			{
				// Rotation constraint error
				b3Vec3W deltaRotation = b3DeltaQuatToRotationJW( relQ, joint->targetRotation );
				b3Vec3W c = b3NegVW( b3RotateVectorJW( quatA, deltaRotation ) );

				b3Vec3W bias = b3MulSVW( joint->springBiasRate, c );
				b3FloatW massScale = joint->springMassScale;
				b3FloatW impulseScale = joint->springImpulseScale;

				b3Vec3W impulse =
					b3MulSubSVW( b3MulSVW( b3NegW( massScale ), b3MulMVJW( joint->rotationMass, b3AddVW( cdot, bias ) ) ),
								 impulseScale, joint->springImpulse );

				joint->springImpulse = b3BlendVW( joint->springImpulse, b3AddVW( joint->springImpulse, impulse ), springMask );
				cdot = b3BlendVW( cdot, b3AddVW( cdot, b3MulMVJW( invInertiaSum, impulse ) ), springMask );
				angularImpulse = b3BlendVW( angularImpulse, b3AddVW( angularImpulse, impulse ), springMask );
			}

			b3FloatW motorMask = b3AndW( angularMask, joint->enableMotor );
			if ( b3AnyTrueW( motorMask ) )
			{
				b3Vec3W lambda = b3NegVW( b3MulMVJW( joint->rotationMass, b3SubVW( cdot, joint->motorVelocity ) ) );
				b3Vec3W newImpulse = b3AddVW( joint->motorImpulse, lambda );
				b3FloatW length = b3SqrtW( b3DotW( newImpulse, newImpulse ) );
				b3FloatW maxImpulse = b3MulW( joint->maxMotorTorque, h );
				b3Vec3W clampedImpulse = b3MulSVW( b3DivW( maxImpulse, length ), newImpulse );
				newImpulse = b3BlendVW( newImpulse, clampedImpulse, b3GreaterThanW( length, maxImpulse ) );

				lambda = b3SubVW( newImpulse, joint->motorImpulse );

				joint->motorImpulse = b3BlendVW( joint->motorImpulse, newImpulse, motorMask );
				cdot = b3BlendVW( cdot, b3AddVW( cdot, b3MulMVJW( invInertiaSum, lambda ) ), motorMask );
				angularImpulse = b3BlendVW( angularImpulse, b3AddVW( angularImpulse, lambda ), motorMask );
			}

			b3FloatW twistMask = b3AndW( angularMask, joint->enableTwistLimit );
			if ( b3AnyTrueW( twistMask ) )
			{
				b3FloatW twistAngle = b3GetTwistAngleJW( relQ );

				b3Vec3W twistJacobian = joint->twistJacobian;
				b3Vec3W twistResponse = b3MulMVJW( invInertiaSum, twistJacobian );

				// Lower limit
				{
					b3FloatW c = b3SubW( twistAngle, joint->lowerTwistAngle );

					// Speculation when the limit is not reached, otherwise the soft constraint
					b3FloatW speculative = b3GreaterThanW( c, zero );
					b3FloatW bias = b3BlendW( useBias ? b3MulW( base->biasRate, c ) : zero, b3MulW( c, inv_h ), speculative );
					b3FloatW massScale = b3BlendW( useBias ? base->massScale : one, one, speculative );
					b3FloatW impulseScale = b3BlendW( useBias ? base->impulseScale : zero, zero, speculative );

					b3FloatW twistCdot = b3DotW( cdot, twistJacobian );
					b3FloatW oldImpulse = joint->lowerTwistImpulse;
					b3FloatW deltaImpulse =
						b3MaxJW( b3SubW( b3MulW( b3MulW( b3NegW( massScale ), joint->twistMass ), b3AddW( twistCdot, bias ) ),
										 b3MulW( impulseScale, oldImpulse ) ),
								 b3NegW( oldImpulse ) );

					joint->lowerTwistImpulse = b3BlendW( oldImpulse, b3AddW( oldImpulse, deltaImpulse ), twistMask );
					cdot = b3BlendVW( cdot, b3MulAddSVW( cdot, deltaImpulse, twistResponse ), twistMask );
					angularImpulse = b3BlendVW( angularImpulse, b3MulAddSVW( angularImpulse, deltaImpulse, twistJacobian ), twistMask );
				}

				// Upper limit
				{
					b3FloatW c = b3SubW( joint->upperTwistAngle, twistAngle );

					b3FloatW speculative = b3GreaterThanW( c, zero );
					b3FloatW bias = b3BlendW( useBias ? b3MulW( base->biasRate, c ) : zero, b3MulW( c, inv_h ), speculative );
					b3FloatW massScale = b3BlendW( useBias ? base->massScale : one, one, speculative );
					b3FloatW impulseScale = b3BlendW( useBias ? base->impulseScale : zero, zero, speculative );

					// sign flipped on Cdot
					b3FloatW twistCdot = b3NegW( b3DotW( cdot, twistJacobian ) );
					b3FloatW oldImpulse = joint->upperTwistImpulse;
					b3FloatW deltaImpulse =
						b3MaxJW( b3SubW( b3MulW( b3MulW( b3NegW( massScale ), joint->twistMass ), b3AddW( twistCdot, bias ) ),
										 b3MulW( impulseScale, oldImpulse ) ),
								 b3NegW( oldImpulse ) );

					joint->upperTwistImpulse = b3BlendW( oldImpulse, b3AddW( oldImpulse, deltaImpulse ), twistMask );

					// sign flipped on applied impulse
					cdot = b3BlendVW( cdot, b3MulSubSVW( cdot, deltaImpulse, twistResponse ), twistMask );
					angularImpulse = b3BlendVW( angularImpulse, b3MulSubSVW( angularImpulse, deltaImpulse, twistJacobian ), twistMask );
				}
			}

			b3FloatW coneMask = b3AndW( angularMask, joint->enableConeLimit );
			if ( b3AnyTrueW( coneMask ) )
			{
				b3FloatW swingAngle = b3GetSwingAngleJW( relQ );
				b3Vec3W swingAxis = joint->swingAxis;

				b3FloatW c = b3SubW( joint->coneAngle, swingAngle );

				b3FloatW speculative = b3GreaterThanW( c, zero );
				b3FloatW bias = b3BlendW( useBias ? b3MulW( base->biasRate, c ) : zero, b3MulW( c, inv_h ), speculative );
				b3FloatW massScale = b3BlendW( useBias ? base->massScale : one, one, speculative );
				b3FloatW impulseScale = b3BlendW( useBias ? base->impulseScale : zero, zero, speculative );

				// sign flipped on Cdot
				b3FloatW swingCdot = b3NegW( b3DotW( cdot, swingAxis ) );
				b3FloatW oldImpulse = joint->swingImpulse;
				b3FloatW deltaImpulse =
					b3MaxJW( b3SubW( b3MulW( b3MulW( b3NegW( massScale ), joint->swingMass ), b3AddW( swingCdot, bias ) ),
									 b3MulW( impulseScale, oldImpulse ) ),
							 b3NegW( oldImpulse ) );

				joint->swingImpulse = b3BlendW( oldImpulse, b3AddW( oldImpulse, deltaImpulse ), coneMask );

				// sign flipped on applied impulse
				angularImpulse = b3BlendVW( angularImpulse, b3MulSubSVW( angularImpulse, deltaImpulse, swingAxis ), coneMask );
			}

			wA = b3BlendVW( wA, b3SubVW( wA, b3MulMVJW( iA, angularImpulse ) ), angularMask );
			wB = b3BlendVW( wB, b3AddVW( wB, b3MulMVJW( iB, angularImpulse ) ), angularMask );
		}

		// Solve point-to-point constraint
		{
			b3Vec3W rA = b3RotateVectorJW( bA.dq, joint->frameAp );
			b3Vec3W rB = b3RotateVectorJW( bB.dq, joint->frameBp );

			b3Vec3W cdot = b3SubVW( b3SubVW( b3AddVW( vB, b3CrossW( wB, rB ) ), vA ), b3CrossW( wA, rA ) );

			b3Vec3W bias = { zero, zero, zero };
			b3FloatW massScale = one;
			b3FloatW impulseScale = zero;
			if ( useBias )
			{
				b3Vec3W separation = b3AddVW( b3SubVW( bB.dp, bA.dp ), b3SubVW( rB, rA ) );
				separation = b3AddVW( separation, joint->deltaCenter );

				bias = b3MulSVW( base->biasRate, separation );
				massScale = base->massScale;
				impulseScale = base->impulseScale;
			}

			b3Vec3W b = b3SolvePointConstraintJW( mA, mB, iA, iB, rA, rB, b3AddVW( cdot, bias ) );

			b3Vec3W impulse = b3MulSubSVW( b3MulSVW( b3NegW( massScale ), b ), impulseScale, joint->linearImpulse );
			joint->linearImpulse = b3AddVW( joint->linearImpulse, impulse );

			vA = b3MulSubSVW( vA, mA, impulse );
			wA = b3SubVW( wA, b3MulMVJW( iA, b3CrossW( rA, impulse ) ) );
			vB = b3MulAddSVW( vB, mB, impulse );
			wB = b3AddVW( wB, b3MulMVJW( iB, b3CrossW( rB, impulse ) ) );
		}

		bA.v = vA;
		bA.w = wA;
		bB.v = vB;
		bB.w = wB;

		b3ScatterBodies( states, base->indexA, &bA );
		b3ScatterBodies( states, base->indexB, &bB );

		// Joint events are tested on the impulses of the biased solve
		if ( useBias && base->eventLanes != 0 )
		{
			for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
			{
				if ( base->eventLanes & ( 1 << lane ) )
				{
					b3StoreSphericalJointLaneW( joint, lane );
					b3CheckJointEventW( base->joints[lane], context, jointStateBitSet );
				}
			}
		}
	}

	b3TracyCZoneEnd( solve_spherical );
}

// b3WarmStartRevoluteJoint
void B3_WIDE( b3WarmStartRevoluteJoints )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( warm_revolute, "WarmRevolute", b3_colorGold, true );

	b3BodyState* states = context->states;
	b3RevoluteJointWide* joints = context->graph->colors[block.colorIndex].wideRevoluteJoints;

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3RevoluteJointWide* joint = joints + i;
		b3JointBaseWide* base = &joint->base;

		b3BodyStateW bA = b3GatherBodies( states, base->indexA );
		b3BodyStateW bB = b3GatherBodies( states, base->indexB );

		b3Vec3W rA = b3RotateVectorJW( bA.dq, joint->frameAp );
		b3Vec3W rB = b3RotateVectorJW( bB.dq, joint->frameBp );

		b3FloatW axialImpulse =
			b3SubW( b3AddW( b3AddW( joint->springImpulse, joint->motorImpulse ), joint->lowerImpulse ), joint->upperImpulse );
		b3Vec3W angularImpulse =
			b3AddVW( b3MulSVW( joint->perpImpulseX, joint->perpAxisX ), b3MulSVW( joint->perpImpulseY, joint->perpAxisY ) );
		angularImpulse = b3MulAddSVW( angularImpulse, axialImpulse, joint->rotationAxisZ );

		bA.v = b3MulSubSVW( bA.v, base->invMassA, joint->linearImpulse );
		bA.w = b3SubVW( bA.w, b3MulMVJW( base->invIA, b3AddVW( b3CrossW( rA, joint->linearImpulse ), angularImpulse ) ) );

		bB.v = b3MulAddSVW( bB.v, base->invMassB, joint->linearImpulse );
		bB.w = b3AddVW( bB.w, b3MulMVJW( base->invIB, b3AddVW( b3CrossW( rB, joint->linearImpulse ), angularImpulse ) ) );

		b3ScatterBodies( states, base->indexA, &bA );
		b3ScatterBodies( states, base->indexB, &bB );
	}

	b3TracyCZoneEnd( warm_revolute );
}

// b3SolveRevoluteJoint
void B3_WIDE( b3SolveRevoluteJoints )( b3SolverBlock block, b3StepContext* context, bool useBias, int workerIndex )
{
	b3TracyCZoneNC( solve_revolute, "SolveRevolute", b3_colorLemonChiffon, true );

	b3BodyState* states = context->states;
	b3RevoluteJointWide* joints = context->graph->colors[block.colorIndex].wideRevoluteJoints;
	b3BitSet* jointStateBitSet = &context->world->taskContexts.data[workerIndex].jointStateBitSet;

	b3FloatW zero = b3ZeroW();
	b3FloatW one = b3SplatW( 1.0f );
	b3FloatW half = b3SplatW( 0.5f );
	b3FloatW h = b3SplatW( context->h );
	b3FloatW inv_h = b3SplatW( context->inv_h );
	b3Vec3W axisX = { one, zero, zero };
	b3Vec3W axisY = { zero, one, zero };

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3RevoluteJointWide* joint = joints + i;
		b3JointBaseWide* base = &joint->base;

		b3FloatW mA = base->invMassA;
		b3FloatW mB = base->invMassB;
		b3Matrix3W iA = base->invIA;
		b3Matrix3W iB = base->invIB;

		b3BodyStateW bA = b3GatherBodies( states, base->indexA );
		b3BodyStateW bB = b3GatherBodies( states, base->indexB );

		b3Vec3W vA = bA.v;
		b3Vec3W wA = bA.w;
		b3Vec3W vB = bB.v;
		b3Vec3W wB = bB.w;

		b3QuatW quatA = b3MulQuatJW( bA.dq, joint->frameAq );
		b3QuatW quatB = b3MulQuatJW( bB.dq, joint->frameBq );

		// this keeps the rotation angle in the range [-pi, pi]
		b3FloatW flip = b3LessThanW( b3DotQuatJW( quatA, quatB ), zero );
		b3QuatW negQuatB = b3NegateQuatJW( quatB );
		quatB.V = b3BlendVW( quatB.V, negQuatB.V, flip );
		quatB.S = b3BlendW( quatB.S, negQuatB.S, flip );

		b3QuatW relQ = b3InvMulQuatJW( quatA, quatB );

		b3FloatW notFixed = b3AndNotW( b3EqualsW( zero, zero ), base->fixedRotation );

		// The scalar solver skips this for a fixed rotation and when no axial feature is enabled
		b3FloatW anyAxial = b3OrW( b3OrW( joint->enableSpring, joint->enableMotor ), joint->enableLimit );
		b3FloatW axialMask = b3AndW( anyAxial, notFixed );

		if ( b3AnyTrueW( axialMask ) )
		{
			b3Vec3W axis = joint->rotationAxisZ;
			b3Vec3W gA = b3MulMVJW( iA, axis );
			b3Vec3W gB = b3MulMVJW( iB, axis );
			b3FloatW k = b3DotW( axis, b3AddVW( gA, gB ) );
			b3FloatW axialCdot = b3DotW( b3SubVW( wB, wA ), axis );
			b3FloatW axialImpulse = zero;

			// Solve spring
			b3FloatW springMask = b3AndW( axialMask, joint->enableSpring );
			if ( b3AnyTrueW( springMask ) )
			{
				// Get the substep relative rotation
				b3FloatW angle = b3GetTwistAngleJW( relQ );
				b3FloatW c = b3SubW( angle, joint->targetAngle );

				b3FloatW bias = b3MulW( joint->springBiasRate, c );
				b3FloatW massScale = joint->springMassScale;
				b3FloatW impulseScale = joint->springImpulseScale;

				b3FloatW deltaImpulse =
					b3SubW( b3MulW( b3MulW( b3NegW( massScale ), joint->axialMass ), b3AddW( axialCdot, bias ) ),
							b3MulW( impulseScale, joint->springImpulse ) );

				joint->springImpulse = b3BlendW( joint->springImpulse, b3AddW( joint->springImpulse, deltaImpulse ), springMask );
				axialCdot = b3BlendW( axialCdot, b3AddW( axialCdot, b3MulW( k, deltaImpulse ) ), springMask );
				axialImpulse = b3BlendW( axialImpulse, b3AddW( axialImpulse, deltaImpulse ), springMask );
			}

			b3FloatW motorMask = b3AndW( axialMask, joint->enableMotor );
			if ( b3AnyTrueW( motorMask ) )
			{
				b3FloatW cdot = b3SubW( axialCdot, joint->motorSpeed );
				b3FloatW oldImpulse = joint->motorImpulse;
				b3FloatW maxImpulse = b3MulW( joint->maxMotorTorque, h );
				b3FloatW deltaImpulse = b3ClampJW( b3MulW( b3NegW( joint->axialMass ), cdot ),
												   b3SubW( b3NegW( maxImpulse ), oldImpulse ), b3SubW( maxImpulse, oldImpulse ) );

				joint->motorImpulse = b3BlendW( oldImpulse, b3AddW( oldImpulse, deltaImpulse ), motorMask );
				axialCdot = b3BlendW( axialCdot, b3AddW( axialCdot, b3MulW( k, deltaImpulse ) ), motorMask );
				axialImpulse = b3BlendW( axialImpulse, b3AddW( axialImpulse, deltaImpulse ), motorMask );
			}

			b3FloatW limitMask = b3AndW( axialMask, joint->enableLimit );
			if ( b3AnyTrueW( limitMask ) )
			{
				b3FloatW angle = b3GetTwistAngleJW( relQ );

				// Lower limit
				{
					b3FloatW c = b3SubW( angle, joint->lowerAngle );

					// Speculation when the limit is not reached, otherwise the soft constraint
					b3FloatW speculative = b3GreaterThanW( c, zero );
					b3FloatW bias = b3BlendW( useBias ? b3MulW( base->biasRate, c ) : zero, b3MulW( c, inv_h ), speculative );
					b3FloatW massScale = b3BlendW( useBias ? base->massScale : one, one, speculative );
					b3FloatW impulseScale = b3BlendW( useBias ? base->impulseScale : zero, zero, speculative );

					b3FloatW oldImpulse = joint->lowerImpulse;
					b3FloatW deltaImpulse =
						b3MaxJW( b3SubW( b3MulW( b3MulW( b3NegW( massScale ), joint->axialMass ), b3AddW( axialCdot, bias ) ),
										 b3MulW( impulseScale, oldImpulse ) ),
								 b3NegW( oldImpulse ) );

					joint->lowerImpulse = b3BlendW( oldImpulse, b3AddW( oldImpulse, deltaImpulse ), limitMask );
					axialCdot = b3BlendW( axialCdot, b3AddW( axialCdot, b3MulW( k, deltaImpulse ) ), limitMask );
					axialImpulse = b3BlendW( axialImpulse, b3AddW( axialImpulse, deltaImpulse ), limitMask );
				}

				// Upper limit
				{
					b3FloatW c = b3SubW( joint->upperAngle, angle );

					b3FloatW speculative = b3GreaterThanW( c, zero );
					b3FloatW bias = b3BlendW( useBias ? b3MulW( base->biasRate, c ) : zero, b3MulW( c, inv_h ), speculative );
					b3FloatW massScale = b3BlendW( useBias ? base->massScale : one, one, speculative );
					b3FloatW impulseScale = b3BlendW( useBias ? base->impulseScale : zero, zero, speculative );

					// sign flipped on Cdot
					b3FloatW cdot = b3NegW( axialCdot );
					b3FloatW oldImpulse = joint->upperImpulse;
					b3FloatW deltaImpulse =
						b3MaxJW( b3SubW( b3MulW( b3MulW( b3NegW( massScale ), joint->axialMass ), b3AddW( cdot, bias ) ),
										 b3MulW( impulseScale, oldImpulse ) ),
								 b3NegW( oldImpulse ) );

					joint->upperImpulse = b3BlendW( oldImpulse, b3AddW( oldImpulse, deltaImpulse ), limitMask );

					// sign flipped on applied impulse
					axialImpulse = b3BlendW( axialImpulse, b3SubW( axialImpulse, deltaImpulse ), limitMask );
				}
			}

			wA = b3BlendVW( wA, b3MulSubSVW( wA, axialImpulse, gA ), axialMask );
			wB = b3BlendVW( wB, b3MulAddSVW( wB, axialImpulse, gB ), axialMask );
		}

		// Collinearity constraint
		if ( b3AnyTrueW( notFixed ) )
		{
			b3FloatW biasX = zero;
			b3FloatW biasY = zero;
			b3FloatW massScale = one;
			b3FloatW impulseScale = zero;

			if ( useBias )
			{
				biasX = b3MulW( base->biasRate, relQ.V.X );
				biasY = b3MulW( base->biasRate, relQ.V.Y );
				massScale = base->massScale;
				impulseScale = base->impulseScale;
			}

			// Collinearity constraint as 2-by-2
			b3Vec3W perpAxisX =
				b3MulSVW( half, b3RotateVectorJW( quatA, b3AddVW( b3MulSVW( relQ.S, axisX ), b3CrossW( relQ.V, axisX ) ) ) );
			b3Vec3W perpAxisY =
				b3MulSVW( half, b3RotateVectorJW( quatA, b3AddVW( b3MulSVW( relQ.S, axisY ), b3CrossW( relQ.V, axisY ) ) ) );
			joint->perpAxisX = b3BlendVW( joint->perpAxisX, perpAxisX, notFixed );
			joint->perpAxisY = b3BlendVW( joint->perpAxisY, perpAxisY, notFixed );

			b3Matrix3W invInertiaSum = b3AddMMJW( iA, iB );
			b3FloatW kxx = b3DotW( perpAxisX, b3MulMVJW( invInertiaSum, perpAxisX ) );
			b3FloatW kyy = b3DotW( perpAxisY, b3MulMVJW( invInertiaSum, perpAxisY ) );
			b3FloatW kxy = b3DotW( perpAxisX, b3MulMVJW( invInertiaSum, perpAxisY ) );

			b3Vec3W wRel = b3SubVW( wB, wA );
			b3FloatW cdotX = b3DotW( wRel, perpAxisX );
			b3FloatW cdotY = b3DotW( wRel, perpAxisY );
			b3FloatW oldImpulseX = joint->perpImpulseX;
			b3FloatW oldImpulseY = joint->perpImpulseY;

			// b3Solve2 with the matrix { { kxx, kxy }, { kxy, kyy } }
			b3FloatW rhsX = b3AddW( cdotX, biasX );
			b3FloatW rhsY = b3AddW( cdotY, biasY );
			b3FloatW det = b3SubW( b3MulW( kxx, kyy ), b3MulW( kxy, kxy ) );
			b3FloatW valid = b3GreaterThanW( det, b3SplatW( 1000.0f * FLT_MIN ) );
			b3FloatW invDet = b3DivW( one, det );
			b3FloatW solX = b3SubW( b3MulW( b3MulW( invDet, kyy ), rhsX ), b3MulW( b3MulW( invDet, kxy ), rhsY ) );
			b3FloatW solY = b3AddW( b3MulW( b3MulW( b3NegW( invDet ), kxy ), rhsX ), b3MulW( b3MulW( invDet, kxx ), rhsY ) );
			solX = b3BlendW( zero, solX, valid );
			solY = b3BlendW( zero, solY, valid );

			b3FloatW deltaImpulseX = b3SubW( b3MulW( b3NegW( massScale ), solX ), b3MulW( impulseScale, oldImpulseX ) );
			b3FloatW deltaImpulseY = b3SubW( b3MulW( b3NegW( massScale ), solY ), b3MulW( impulseScale, oldImpulseY ) );
			joint->perpImpulseX = b3BlendW( oldImpulseX, b3AddW( oldImpulseX, deltaImpulseX ), notFixed );
			joint->perpImpulseY = b3BlendW( oldImpulseY, b3AddW( oldImpulseY, deltaImpulseY ), notFixed );

			b3Vec3W angularImpulse = b3AddVW( b3MulSVW( deltaImpulseX, perpAxisX ), b3MulSVW( deltaImpulseY, perpAxisY ) );
			wA = b3BlendVW( wA, b3SubVW( wA, b3MulMVJW( iA, angularImpulse ) ), notFixed );
			wB = b3BlendVW( wB, b3AddVW( wB, b3MulMVJW( iB, angularImpulse ) ), notFixed );
		}

		// Solve point-to-point constraint
		{
			b3Vec3W rA = b3RotateVectorJW( bA.dq, joint->frameAp );
			b3Vec3W rB = b3RotateVectorJW( bB.dq, joint->frameBp );

			b3Vec3W cdot = b3SubVW( b3SubVW( b3AddVW( vB, b3CrossW( wB, rB ) ), vA ), b3CrossW( wA, rA ) );

			b3Vec3W bias = { zero, zero, zero };
			b3FloatW massScale = one;
			b3FloatW impulseScale = zero;
			if ( useBias )
			{
				b3Vec3W separation = b3AddVW( b3AddVW( b3SubVW( bB.dp, bA.dp ), b3SubVW( rB, rA ) ), joint->deltaCenter );

				bias = b3MulSVW( base->biasRate, separation );
				massScale = base->massScale;
				impulseScale = base->impulseScale;
			}

			b3Vec3W b = b3SolvePointConstraintJW( mA, mB, iA, iB, rA, rB, b3AddVW( cdot, bias ) );

			b3Vec3W impulse = b3SubVW( b3MulSVW( b3NegW( massScale ), b ), b3MulSVW( impulseScale, joint->linearImpulse ) );
			joint->linearImpulse = b3AddVW( joint->linearImpulse, impulse );

			vA = b3MulSubSVW( vA, mA, impulse );
			wA = b3SubVW( wA, b3MulMVJW( iA, b3CrossW( rA, impulse ) ) );
			vB = b3MulAddSVW( vB, mB, impulse );
			wB = b3AddVW( wB, b3MulMVJW( iB, b3CrossW( rB, impulse ) ) );
		}

		bA.v = vA;
		bA.w = wA;
		bB.v = vB;
		bB.w = wB;

		b3ScatterBodies( states, base->indexA, &bA );
		b3ScatterBodies( states, base->indexB, &bB );

		// Joint events are tested on the impulses of the biased solve
		if ( useBias && base->eventLanes != 0 )
		{
			for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
			{
				if ( base->eventLanes & ( 1 << lane ) )
				{
					b3StoreRevoluteJointLaneW( joint, lane );
					b3CheckJointEventW( base->joints[lane], context, jointStateBitSet );
				}
			}
		}
	}

	b3TracyCZoneEnd( solve_revolute );
}

#undef B3_GATHER_JOINT
#undef B3_GATHER_JOINT_V3
#undef B3_GATHER_JOINT_QUAT
#undef B3_GATHER_JOINT_M3
#undef B3_GATHER_JOINT_MASK
#undef B3_SCATTER_JOINT
#undef B3_SCATTER_JOINT_V3
#undef B3_STORE_LANE
#undef B3_STORE_LANE_V3
