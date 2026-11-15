// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

// Wide body integration. B3_SIMD_WIDTH bodies per iteration.
//
// Velocity and position integration run once per sub-step on every awake body. The scalar versions in
// solver.c read most of the body sim each time, but what they read is constant for the whole time step:
// damping, applied force and torque, inertia, and the rotation at the start of the step. So this
// works in two parts
// 1. b3PrepareIntegration runs once per step and reduces each body sim to the few values the sub-steps
//    need, stored one group of bodies per wide struct
// 2. b3IntegrateVelocities and b3IntegratePositions run every sub-step on those groups and on the body
//    states, which are contiguous and can be loaded a group at a time
//
// Like the wide joint solver this must be bit for bit identical to the scalar code it replaces, here
// b3IntegrateVelocitiesTask and b3IntegratePositionsTask in solver.c. Every operation is transcribed in
// the same order with the same grouping, branches become blends, and there is no fused multiply-add.
// The values computed once per step are computed with the same scalar expressions the scalar task
// evaluates every sub-step, from inputs that don't change during the step, so they are the same values.
// When changing the scalar tasks the matching code here must change with it, and the reverse.
//
// A group is B3_SIMD_WIDTH bodies that are consecutive in the awake set. The last group may be partial.
// Its unused lanes read the identity body state and are never written.
//
// This file is included after joint_solver_wide.inl and shares its wide math.

#include "body.h"
#include "physics_world.h"
#include "simd.h"
#include "simd_wide.h"
#include "solver.h"

#include <float.h>

// The values of a group of bodies that velocity integration needs and that are constant over the step
typedef struct B3_WIDE( b3IntegrateData )
{
	// 1 / (1 + h * damping)
	_Alignas( B3_WIDE_ALIGNMENT ) b3FloatW linearDamping;
	b3FloatW angularDamping;

	// Velocity change from applied force, gravity, and applied torque over a sub-step, before damping
	b3Vec3W linearDelta;
	b3Vec3W angularDelta;

	// For gyroscopic torque. Rotation at the start of the step and the unique terms of the local inertia.
	b3QuatW q0;
	b3FloatW i00, i01, i02, i11, i12, i22;

	// Mask of the bodies with gyroscopic torque
	b3FloatW gyroscopic;
	int anyGyroscopic;

	// Unused lanes are at the end
	int laneCount;
} b3IntegrateDataWide;

int B3_WIDE( b3GetIntegrateDataByteCount )( void )
{
	return sizeof( b3IntegrateDataWide );
}

// b3InvRotateVector
static inline b3Vec3W b3InvRotateVectorJW( b3QuatW q, b3Vec3W v )
{
	b3Vec3W t1 = b3CrossW( q.V, v );
	b3Vec3W t2 = b3MulSubSVW( t1, q.S, v );
	b3Vec3W t3 = b3CrossW( q.V, t2 );
	return b3MulAddSVW( v, b3SplatW( 2.0f ), t3 );
}

// b3Solve3
static inline b3Vec3W b3Solve3JW( b3Matrix3W m, b3Vec3W a )
{
	// b3Det
	b3FloatW det = b3DotW( m.cx, b3CrossW( m.cy, m.cz ) );
	b3FloatW valid = b3GreaterThanW( b3AbsW( det ), b3SplatW( 1000.0f * FLT_MIN ) );
	b3FloatW invDet = b3DivW( b3SplatW( 1.0f ), det );

	b3Vec3W sx = b3CrossW( m.cy, m.cz );
	b3Vec3W sy = b3CrossW( m.cz, m.cx );
	b3Vec3W sz = b3CrossW( m.cx, m.cy );

	b3FloatW zero = b3ZeroW();
	return (b3Vec3W){
		b3BlendW( zero, b3MulW( invDet, b3DotW( sx, a ) ), valid ),
		b3BlendW( zero, b3MulW( invDet, b3DotW( sy, a ) ), valid ),
		b3BlendW( zero, b3MulW( invDet, b3DotW( sz, a ) ), valid ),
	};
}

// b3NormalizeQuat
static inline b3QuatW b3NormalizeQuatJW( b3QuatW q )
{
	b3FloatW lengthSq = b3DotQuatJW( q, q );
	b3FloatW valid = b3GreaterThanW( lengthSq, b3SplatW( 1000.0f * FLT_MIN ) );
	b3FloatW s = b3DivW( b3SplatW( 1.0f ), b3SqrtW( lengthSq ) );

	b3FloatW zero = b3ZeroW();
	b3FloatW one = b3SplatW( 1.0f );
	b3QuatW qn;
	qn.V.X = b3BlendW( zero, b3MulW( s, q.V.X ), valid );
	qn.V.Y = b3BlendW( zero, b3MulW( s, q.V.Y ), valid );
	qn.V.Z = b3BlendW( zero, b3MulW( s, q.V.Z ), valid );
	qn.S = b3BlendW( one, b3MulW( s, q.S ), valid );
	return qn;
}

// b3Length
static inline b3FloatW b3LengthJW( b3Vec3W v )
{
	return b3SqrtW( b3DotW( v, v ) );
}

// b3IntegrateRotation
static inline b3QuatW b3IntegrateRotationJW( b3QuatW q1, b3Vec3W deltaRotation )
{
	b3QuatW qd = { b3MulSVW( b3SplatW( 0.5f ), deltaRotation ), b3ZeroW() };
	qd = b3MulQuatJW( qd, q1 );
	b3QuatW q2 = { b3AddVW( q1.V, qd.V ), b3AddW( qd.S, q1.S ) };

	return b3NormalizeQuatJW( q2 );
}

// The body states of a group as wide values. The flags column carries the flags of each body as raw bits
// so that the state can be written back a whole row at a time without disturbing them.
typedef struct B3_WIDE( b3BodyGroup )
{
	b3Vec3W v;
	b3Vec3W w;
	b3Vec3W dp;
	b3FloatW flags;
	b3QuatW dq;
} b3BodyGroupW;

// Load a group of consecutive body states. Unused lanes get the identity state.
B3_FORCE_INLINE b3BodyGroupW b3LoadBodyGroup( const b3BodyState* B3_RESTRICT states, int laneCount )
{
	const float* identity = (const float*)&b3_identityBodyState;
	const float* p[B3_SIMD_WIDTH];
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		p[lane] = lane < laneCount ? (const float*)( states + lane ) : identity;
	}

	b3BodyGroupW g;
	b3FloatW pad;

#if B3_SIMD_WIDTH == 8
	b3TransposeW( b3LoadW( p[0] ), b3LoadW( p[1] ), b3LoadW( p[2] ), b3LoadW( p[3] ), b3LoadW( p[4] ), b3LoadW( p[5] ),
				  b3LoadW( p[6] ), b3LoadW( p[7] ), &g.v.X, &g.v.Y, &g.v.Z, &pad, &g.w.X, &g.w.Y, &g.w.Z, &pad );
	b3TransposeW( b3LoadW( p[0] + 8 ), b3LoadW( p[1] + 8 ), b3LoadW( p[2] + 8 ), b3LoadW( p[3] + 8 ), b3LoadW( p[4] + 8 ),
				  b3LoadW( p[5] + 8 ), b3LoadW( p[6] + 8 ), b3LoadW( p[7] + 8 ), &g.dp.X, &g.dp.Y, &g.dp.Z, &g.flags, &g.dq.V.X,
				  &g.dq.V.Y, &g.dq.V.Z, &g.dq.S );
#else
	b3TransposeW( b3LoadW( p[0] ), b3LoadW( p[1] ), b3LoadW( p[2] ), b3LoadW( p[3] ), &g.v.X, &g.v.Y, &g.v.Z, &pad );
	b3TransposeW( b3LoadW( p[0] + 4 ), b3LoadW( p[1] + 4 ), b3LoadW( p[2] + 4 ), b3LoadW( p[3] + 4 ), &g.w.X, &g.w.Y, &g.w.Z, &pad );
	b3TransposeW( b3LoadW( p[0] + 8 ), b3LoadW( p[1] + 8 ), b3LoadW( p[2] + 8 ), b3LoadW( p[3] + 8 ), &g.dp.X, &g.dp.Y, &g.dp.Z,
				  &g.flags );
	b3TransposeW( b3LoadW( p[0] + 12 ), b3LoadW( p[1] + 12 ), b3LoadW( p[2] + 12 ), b3LoadW( p[3] + 12 ), &g.dq.V.X, &g.dq.V.Y,
				  &g.dq.V.Z, &g.dq.S );
#endif

	return g;
}

// Write the velocities of a group. The scalar task writes every body, including kinematic bodies.
B3_FORCE_INLINE void b3StoreBodyGroupVelocities( b3BodyState* B3_RESTRICT states, int laneCount, const b3BodyGroupW* g )
{
	b3FloatW zero = b3ZeroW();

#if B3_SIMD_WIDTH == 8
	b3FloatW rows[8];
	b3TransposeW( g->v.X, g->v.Y, g->v.Z, zero, g->w.X, g->w.Y, g->w.Z, zero, rows + 0, rows + 1, rows + 2, rows + 3, rows + 4,
				  rows + 5, rows + 6, rows + 7 );

	for ( int lane = 0; lane < laneCount; ++lane )
	{
		b3StoreW( (float*)( states + lane ), rows[lane] );
	}
#else
	b3FloatW rowsV[4], rowsW[4];
	b3TransposeW( g->v.X, g->v.Y, g->v.Z, zero, rowsV + 0, rowsV + 1, rowsV + 2, rowsV + 3 );
	b3TransposeW( g->w.X, g->w.Y, g->w.Z, zero, rowsW + 0, rowsW + 1, rowsW + 2, rowsW + 3 );

	for ( int lane = 0; lane < laneCount; ++lane )
	{
		b3StoreW( (float*)( states + lane ), rowsV[lane] );
		b3StoreW( (float*)( states + lane ) + 4, rowsW[lane] );
	}
#endif
}

// Write the whole state of a group. The flags column goes back unchanged.
B3_FORCE_INLINE void b3StoreBodyGroup( b3BodyState* B3_RESTRICT states, int laneCount, const b3BodyGroupW* g )
{
	b3StoreBodyGroupVelocities( states, laneCount, g );

#if B3_SIMD_WIDTH == 8
	b3FloatW rows[8];
	b3TransposeW( g->dp.X, g->dp.Y, g->dp.Z, g->flags, g->dq.V.X, g->dq.V.Y, g->dq.V.Z, g->dq.S, rows + 0, rows + 1, rows + 2,
				  rows + 3, rows + 4, rows + 5, rows + 6, rows + 7 );

	for ( int lane = 0; lane < laneCount; ++lane )
	{
		b3StoreW( (float*)( states + lane ) + 8, rows[lane] );
	}
#else
	b3FloatW rowsP[4], rowsQ[4];
	b3TransposeW( g->dp.X, g->dp.Y, g->dp.Z, g->flags, rowsP + 0, rowsP + 1, rowsP + 2, rowsP + 3 );
	b3TransposeW( g->dq.V.X, g->dq.V.Y, g->dq.V.Z, g->dq.S, rowsQ + 0, rowsQ + 1, rowsQ + 2, rowsQ + 3 );

	for ( int lane = 0; lane < laneCount; ++lane )
	{
		b3StoreW( (float*)( states + lane ) + 8, rowsP[lane] );
		b3StoreW( (float*)( states + lane ) + 12, rowsQ[lane] );
	}
#endif
}

// A mask that is on in the lanes of the group where the body state has any of the flag bits
B3_FORCE_INLINE b3FloatW b3BodyFlagMask( const b3BodyState* B3_RESTRICT states, int laneCount, uint32_t flagBits )
{
	float lanes[B3_SIMD_WIDTH];
	for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
	{
		lanes[lane] = lane < laneCount && ( states[lane].flags & flagBits ) != 0 ? 1.0f : 0.0f;
	}

	return b3GreaterThanW( b3LoadW( lanes ), b3ZeroW() );
}

// The block is a range of bodies that starts on a group boundary
void B3_WIDE( b3PrepareIntegration )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( prepare_integration, "PrepIntegrate", b3_colorDeepPink, true );

	const b3BodySim* sims = context->sims;
	b3IntegrateDataWide* groups = context->wideIntegrateData;
	b3Vec3 gravity = context->world->gravity;
	float h = context->h;

	B3_ASSERT( block.startIndex % B3_SIMD_WIDTH == 0 );
	int endIndex = block.startIndex + block.count;

	for ( int first = block.startIndex; first < endIndex; first += B3_SIMD_WIDTH )
	{
		b3IntegrateDataWide* group = groups + first / B3_SIMD_WIDTH;
		int laneCount = b3MinInt( B3_SIMD_WIDTH, endIndex - first );

		// Unused lanes stay zero
		float data[20][B3_SIMD_WIDTH] = { 0 };
		int anyGyroscopic = 0;

		for ( int lane = 0; lane < laneCount; ++lane )
		{
			const b3BodySim* sim = sims + first + lane;

			// These are the expressions of b3IntegrateVelocitiesTask
			float linearDamping = 1.0f / ( 1.0f + h * sim->linearDamping );
			float angularDamping = 1.0f / ( 1.0f + h * sim->angularDamping );

			// Gravity scale will be zero for kinematic bodies
			float gravityScale = sim->invMass > 0.0f ? sim->gravityScale : 0.0f;

			b3Vec3 linearVelocityDelta = b3Blend2( h * sim->invMass, sim->force, h * gravityScale, gravity );
			b3Vec3 angularVelocityDelta = b3MulSV( h, b3MulMV( sim->invInertiaWorld, sim->torque ) );

			bool gyroscopic = ( sim->flags & b3_enableGyroscopicTorque ) != 0;
			anyGyroscopic |= gyroscopic ? 1 : 0;

			data[0][lane] = linearDamping;
			data[1][lane] = angularDamping;
			data[2][lane] = linearVelocityDelta.x;
			data[3][lane] = linearVelocityDelta.y;
			data[4][lane] = linearVelocityDelta.z;
			data[5][lane] = angularVelocityDelta.x;
			data[6][lane] = angularVelocityDelta.y;
			data[7][lane] = angularVelocityDelta.z;
			data[8][lane] = sim->transform.q.v.x;
			data[9][lane] = sim->transform.q.v.y;
			data[10][lane] = sim->transform.q.v.z;
			data[11][lane] = sim->transform.q.s;

			// Symmetric inertia tensor: 6 unique entries (column-major)
			data[12][lane] = sim->inertia.cx.x;
			data[13][lane] = sim->inertia.cy.x;
			data[14][lane] = sim->inertia.cz.x;
			data[15][lane] = sim->inertia.cy.y;
			data[16][lane] = sim->inertia.cz.y;
			data[17][lane] = sim->inertia.cz.z;
			data[18][lane] = gyroscopic ? 1.0f : 0.0f;
		}

		group->linearDamping = b3LoadW( data[0] );
		group->angularDamping = b3LoadW( data[1] );
		group->linearDelta = (b3Vec3W){ b3LoadW( data[2] ), b3LoadW( data[3] ), b3LoadW( data[4] ) };
		group->angularDelta = (b3Vec3W){ b3LoadW( data[5] ), b3LoadW( data[6] ), b3LoadW( data[7] ) };
		group->q0 = (b3QuatW){ { b3LoadW( data[8] ), b3LoadW( data[9] ), b3LoadW( data[10] ) }, b3LoadW( data[11] ) };
		group->i00 = b3LoadW( data[12] );
		group->i01 = b3LoadW( data[13] );
		group->i02 = b3LoadW( data[14] );
		group->i11 = b3LoadW( data[15] );
		group->i12 = b3LoadW( data[16] );
		group->i22 = b3LoadW( data[17] );
		group->gyroscopic = b3GreaterThanW( b3LoadW( data[18] ), b3ZeroW() );
		group->anyGyroscopic = anyGyroscopic;
		group->laneCount = laneCount;
	}

	b3TracyCZoneEnd( prepare_integration );
}

// b3IntegrateVelocitiesTask
void B3_WIDE( b3IntegrateVelocities )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( integrate_velocity, "IntVel", b3_colorDeepPink, true );

	b3BodyState* states = context->states;
	const b3IntegrateDataWide* groups = context->wideIntegrateData;
	b3FloatW h = b3SplatW( context->h );

	B3_ASSERT( block.startIndex % B3_SIMD_WIDTH == 0 );
	int endIndex = block.startIndex + block.count;

	for ( int first = block.startIndex; first < endIndex; first += B3_SIMD_WIDTH )
	{
		const b3IntegrateDataWide* group = groups + first / B3_SIMD_WIDTH;
		int laneCount = group->laneCount;
		b3BodyGroupW b = b3LoadBodyGroup( states + first, laneCount );

		b3Vec3W v = b3MulAddSVW( group->linearDelta, group->linearDamping, b.v );
		b3Vec3W w = b3MulAddSVW( group->angularDelta, group->angularDamping, b.w );

		// Gyroscopic torque by solving this nonlinear equation using Newton-Raphson.
		// I * (w2 - w1) + h * cross(w2, I * w2) = 0
		// See b3IntegrateVelocitiesTask for details.
		if ( group->anyGyroscopic )
		{
			// Get current rotation.
			b3QuatW q = b3MulQuatJW( b.dq, group->q0 );

			// Compute local angular velocity
			b3Vec3W omega1 = b3InvRotateVectorJW( q, w );
			b3Vec3W omega2 = omega1;

			b3FloatW i00 = group->i00;
			b3FloatW i01 = group->i01;
			b3FloatW i02 = group->i02;
			b3FloatW i11 = group->i11;
			b3FloatW i12 = group->i12;
			b3FloatW i22 = group->i22;

			for ( int gyroIteration = 0; gyroIteration < B3_GYROSCOPIC_ITERATIONS; ++gyroIteration )
			{
				b3FloatW w1 = omega2.X;
				b3FloatW w2 = omega2.Y;
				b3FloatW w3 = omega2.Z;

				// Iw = I * omega2 (shared between residual and Jacobian)
				b3FloatW Iw1 = b3AddW( b3AddW( b3MulW( i00, w1 ), b3MulW( i01, w2 ) ), b3MulW( i02, w3 ) );
				b3FloatW Iw2 = b3AddW( b3AddW( b3MulW( i01, w1 ), b3MulW( i11, w2 ) ), b3MulW( i12, w3 ) );
				b3FloatW Iw3 = b3AddW( b3AddW( b3MulW( i02, w1 ), b3MulW( i12, w2 ) ), b3MulW( i22, w3 ) );

				// Residual: b = I * (omega2 - omega1) + h * cross(omega2, I * omega2)
				b3Vec3W dw = b3SubVW( omega2, omega1 );
				b3Vec3W residual = {
					b3AddW( b3AddW( b3AddW( b3MulW( i00, dw.X ), b3MulW( i01, dw.Y ) ), b3MulW( i02, dw.Z ) ),
							b3MulW( h, b3SubW( b3MulW( w2, Iw3 ), b3MulW( w3, Iw2 ) ) ) ),
					b3AddW( b3AddW( b3AddW( b3MulW( i01, dw.X ), b3MulW( i11, dw.Y ) ), b3MulW( i12, dw.Z ) ),
							b3MulW( h, b3SubW( b3MulW( w3, Iw1 ), b3MulW( w1, Iw3 ) ) ) ),
					b3AddW( b3AddW( b3AddW( b3MulW( i02, dw.X ), b3MulW( i12, dw.Y ) ), b3MulW( i22, dw.Z ) ),
							b3MulW( h, b3SubW( b3MulW( w1, Iw2 ), b3MulW( w2, Iw1 ) ) ) ),
				};

				// Jacobian J = I + h * (skew(omega2) * I - skew(I * omega2))
				b3Matrix3W J;
				J.cx.X = b3AddW( i00, b3MulW( h, b3SubW( b3MulW( w2, i02 ), b3MulW( w3, i01 ) ) ) );
				J.cx.Y = b3AddW( i01, b3MulW( h, b3SubW( b3SubW( b3MulW( w3, i00 ), b3MulW( w1, i02 ) ), Iw3 ) ) );
				J.cx.Z = b3AddW( i02, b3MulW( h, b3AddW( b3SubW( b3MulW( w1, i01 ), b3MulW( w2, i00 ) ), Iw2 ) ) );
				J.cy.X = b3AddW( i01, b3MulW( h, b3AddW( b3SubW( b3MulW( w2, i12 ), b3MulW( w3, i11 ) ), Iw3 ) ) );
				J.cy.Y = b3AddW( i11, b3MulW( h, b3SubW( b3MulW( w3, i01 ), b3MulW( w1, i12 ) ) ) );
				J.cy.Z = b3AddW( i12, b3MulW( h, b3SubW( b3SubW( b3MulW( w1, i11 ), b3MulW( w2, i01 ) ), Iw1 ) ) );
				J.cz.X = b3AddW( i02, b3MulW( h, b3SubW( b3SubW( b3MulW( w2, i22 ), b3MulW( w3, i12 ) ), Iw2 ) ) );
				J.cz.Y = b3AddW( i12, b3MulW( h, b3AddW( b3SubW( b3MulW( w3, i02 ), b3MulW( w1, i22 ) ), Iw1 ) ) );
				J.cz.Z = b3AddW( i22, b3MulW( h, b3SubW( b3MulW( w1, i12 ), b3MulW( w2, i02 ) ) ) );

				omega2 = b3SubVW( omega2, b3Solve3JW( J, residual ) );
			}

			w = b3BlendVW( w, b3RotateVectorJW( q, omega2 ), group->gyroscopic );
		}

		b.v = v;
		b.w = w;
		b3StoreBodyGroupVelocities( states + first, laneCount, &b );
	}

	b3TracyCZoneEnd( integrate_velocity );
}

// b3IntegratePositionsTask
void B3_WIDE( b3IntegratePositions )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( integrate_positions, "IntPos", b3_colorDarkSeaGreen, true );

	b3BodyState* states = context->states;
	float maxLinearSpeed = context->maxLinearVelocity;
	float maxAngularSpeed = B3_MAX_ROTATION * context->inv_dt;
	b3FloatW h = b3SplatW( context->h );
	b3FloatW maxLinearSpeedW = b3SplatW( maxLinearSpeed );
	b3FloatW maxAngularSpeedW = b3SplatW( maxAngularSpeed );
	b3FloatW maxLinearSpeedSquared = b3SplatW( maxLinearSpeed * maxLinearSpeed );
	b3FloatW maxAngularSpeedSquared = b3SplatW( maxAngularSpeed * maxAngularSpeed );
	b3FloatW zero = b3ZeroW();

	B3_ASSERT( block.startIndex % B3_SIMD_WIDTH == 0 );
	int endIndex = block.startIndex + block.count;

	for ( int first = block.startIndex; first < endIndex; first += B3_SIMD_WIDTH )
	{
		b3BodyState* group = states + first;
		int laneCount = b3MinInt( B3_SIMD_WIDTH, endIndex - first );
		b3BodyGroupW b = b3LoadBodyGroup( group, laneCount );

		b3Vec3W v = b.v;
		b3Vec3W w = b.w;

		// Motion locks - these can be viewed as a constraint that come last. Most bodies have none.
		uint32_t lockFlags = 0;
		for ( int lane = 0; lane < laneCount; ++lane )
		{
			lockFlags |= group[lane].flags;
		}

		if ( lockFlags & b3_allLocks )
		{
			v.X = b3BlendW( v.X, zero, b3BodyFlagMask( group, laneCount, b3_lockLinearX ) );
			v.Y = b3BlendW( v.Y, zero, b3BodyFlagMask( group, laneCount, b3_lockLinearY ) );
			v.Z = b3BlendW( v.Z, zero, b3BodyFlagMask( group, laneCount, b3_lockLinearZ ) );
			w.X = b3BlendW( w.X, zero, b3BodyFlagMask( group, laneCount, b3_lockAngularX ) );
			w.Y = b3BlendW( w.Y, zero, b3BodyFlagMask( group, laneCount, b3_lockAngularY ) );
			w.Z = b3BlendW( w.Z, zero, b3BodyFlagMask( group, laneCount, b3_lockAngularZ ) );
		}

		// Clamp to max linear speed
		b3FloatW linearSpeedSquared = b3DotW( v, v );
		b3FloatW linearCapped = b3GreaterThanW( linearSpeedSquared, maxLinearSpeedSquared );
		if ( b3AnyTrueW( linearCapped ) )
		{
			b3FloatW ratio = b3DivW( maxLinearSpeedW, b3SqrtW( linearSpeedSquared ) );
			v = b3BlendVW( v, b3MulSVW( ratio, v ), linearCapped );
		}

		// Clamp to max angular speed
		b3FloatW angularSpeedSquared = b3DotW( w, w );
		b3FloatW angularCapped = b3GreaterThanW( angularSpeedSquared, maxAngularSpeedSquared );
		if ( b3AnyTrueW( angularCapped ) )
		{
			angularCapped = b3AndNotW( angularCapped, b3BodyFlagMask( group, laneCount, b3_allowFastRotation ) );
			b3FloatW ratio = b3DivW( maxAngularSpeedW, b3SqrtW( angularSpeedSquared ) );
			w = b3BlendVW( w, b3MulSVW( ratio, w ), angularCapped );
		}

		b.v = v;
		b.w = w;
		b.dp = b3MulAddSVW( b.dp, h, v );
		b.dq = b3IntegrateRotationJW( b.dq, b3MulSVW( h, w ) );

		b3StoreBodyGroup( group, laneCount, &b );

		// The wide store put the old flags back, so flag the capped bodies now
		b3FloatW capped = b3OrW( linearCapped, angularCapped );
		if ( b3AnyTrueW( capped ) )
		{
			// A lane is capped if its ratio was applied, which is where the masked value is non-zero
			float lanes[B3_SIMD_WIDTH];
			b3StoreW( lanes, b3BlendW( zero, b3SplatW( 1.0f ), capped ) );
			for ( int lane = 0; lane < laneCount; ++lane )
			{
				if ( lanes[lane] != 0.0f )
				{
					group[lane].flags |= b3_isSpeedCapped;
				}
			}
		}
	}

	b3TracyCZoneEnd( integrate_positions );
}

// The math of b3FinalizeBodiesTask for a group of consecutive bodies: the new rotation, how fast the body
// moved for sleeping and continuous collision, and the world space inverse inertia. Nothing is written to
// the bodies here. The task takes the results from the lanes and does the rest per body in the same order
// as before. Unused lanes hold garbage.
void B3_WIDE( b3FinalizeBodyMath )( const b3BodySim* sims, const b3BodyState* states, int laneCount, float invTimeStep,
									b3FinalizeLanes* lanes )
{
	b3BodyGroupW g = b3LoadBodyGroup( states, laneCount );

	// Unused lanes stay zero
	_Alignas( B3_WIDE_ALIGNMENT ) float data[19][B3_SIMD_WIDTH] = { 0 };
	for ( int lane = 0; lane < laneCount; ++lane )
	{
		const b3BodySim* sim = sims + lane;
		data[0][lane] = sim->transform.q.v.x;
		data[1][lane] = sim->transform.q.v.y;
		data[2][lane] = sim->transform.q.v.z;
		data[3][lane] = sim->transform.q.s;
		data[4][lane] = sim->maxExtent.x;
		data[5][lane] = sim->maxExtent.y;
		data[6][lane] = sim->maxExtent.z;
		data[7][lane] = sim->localCenter.x;
		data[8][lane] = sim->localCenter.y;
		data[9][lane] = sim->localCenter.z;
		data[10][lane] = sim->invInertiaLocal.cx.x;
		data[11][lane] = sim->invInertiaLocal.cx.y;
		data[12][lane] = sim->invInertiaLocal.cx.z;
		data[13][lane] = sim->invInertiaLocal.cy.x;
		data[14][lane] = sim->invInertiaLocal.cy.y;
		data[15][lane] = sim->invInertiaLocal.cy.z;
		data[16][lane] = sim->invInertiaLocal.cz.x;
		data[17][lane] = sim->invInertiaLocal.cz.y;
		data[18][lane] = sim->invInertiaLocal.cz.z;
	}

	b3QuatW q0 = { { b3LoadW( data[0] ), b3LoadW( data[1] ), b3LoadW( data[2] ) }, b3LoadW( data[3] ) };
	b3Vec3W maxExtent = { b3LoadW( data[4] ), b3LoadW( data[5] ), b3LoadW( data[6] ) };
	b3Vec3W localCenter = { b3LoadW( data[7] ), b3LoadW( data[8] ), b3LoadW( data[9] ) };
	b3Matrix3W invInertiaLocal = {
		{ b3LoadW( data[10] ), b3LoadW( data[11] ), b3LoadW( data[12] ) },
		{ b3LoadW( data[13] ), b3LoadW( data[14] ), b3LoadW( data[15] ) },
		{ b3LoadW( data[16] ), b3LoadW( data[17] ), b3LoadW( data[18] ) },
	};

	// These use the rotation before it is advanced
	b3Vec3W localOmega = b3InvRotateVectorJW( q0, g.w );
	b3Vec3W localDeltaRotation = b3InvRotateVectorJW( q0, g.dq.V );

	b3QuatW q = b3NormalizeQuatJW( b3MulQuatJW( g.dq, q0 ) );

	// Use the velocity of the farthest point on the body to account for rotation.
	b3Vec3W absOmega = { b3AbsJW( localOmega.X ), b3AbsJW( localOmega.Y ), b3AbsJW( localOmega.Z ) };
	b3Vec3W velocityArc = b3ModifiedCrossJW( absOmega, maxExtent );
	b3FloatW maxVelocity = b3AddW( b3LengthJW( g.v ), b3LengthJW( velocityArc ) );

	// Sleep needs to observe position correction as well as true velocity.
	b3Vec3W absDeltaRotation = { b3AbsJW( localDeltaRotation.X ), b3AbsJW( localDeltaRotation.Y ), b3AbsJW( localDeltaRotation.Z ) };
	b3Vec3W rotationArc = b3ModifiedCrossJW( absDeltaRotation, maxExtent );
	b3FloatW maxDeltaPosition = b3AddW( b3LengthJW( g.dp ), b3MulW( b3SplatW( 2.0f ), b3LengthJW( rotationArc ) ) );

	// Position correction is not as important for sleep as true velocity.
	float positionSleepFactor = 0.5f;
	b3FloatW sleepVelocity = b3MaxJW( maxVelocity, b3MulW( b3SplatW( positionSleepFactor * invTimeStep ), maxDeltaPosition ) );

	b3Vec3W rotatedCenter = b3RotateVectorJW( q, localCenter );

	// Update world space inverse inertia tensor: R * invInertiaLocal * transpose( R ) as two b3MulMM
	b3Matrix3W rotation = b3MakeMatrixFromQuatJW( q );
	b3Matrix3W transposed = {
		{ rotation.cx.X, rotation.cy.X, rotation.cz.X },
		{ rotation.cx.Y, rotation.cy.Y, rotation.cz.Y },
		{ rotation.cx.Z, rotation.cy.Z, rotation.cz.Z },
	};
	b3Matrix3W temp = {
		b3MulMVJW( rotation, invInertiaLocal.cx ),
		b3MulMVJW( rotation, invInertiaLocal.cy ),
		b3MulMVJW( rotation, invInertiaLocal.cz ),
	};
	b3Matrix3W invInertiaWorld = {
		b3MulMVJW( temp, transposed.cx ),
		b3MulMVJW( temp, transposed.cy ),
		b3MulMVJW( temp, transposed.cz ),
	};

	b3StoreW( lanes->qx, q.V.X );
	b3StoreW( lanes->qy, q.V.Y );
	b3StoreW( lanes->qz, q.V.Z );
	b3StoreW( lanes->qs, q.S );
	b3StoreW( lanes->centerX, rotatedCenter.X );
	b3StoreW( lanes->centerY, rotatedCenter.Y );
	b3StoreW( lanes->centerZ, rotatedCenter.Z );
	b3StoreW( lanes->invInertia[0], invInertiaWorld.cx.X );
	b3StoreW( lanes->invInertia[1], invInertiaWorld.cx.Y );
	b3StoreW( lanes->invInertia[2], invInertiaWorld.cx.Z );
	b3StoreW( lanes->invInertia[3], invInertiaWorld.cy.X );
	b3StoreW( lanes->invInertia[4], invInertiaWorld.cy.Y );
	b3StoreW( lanes->invInertia[5], invInertiaWorld.cy.Z );
	b3StoreW( lanes->invInertia[6], invInertiaWorld.cz.X );
	b3StoreW( lanes->invInertia[7], invInertiaWorld.cz.Y );
	b3StoreW( lanes->invInertia[8], invInertiaWorld.cz.Z );
	b3StoreW( lanes->maxVelocity, maxVelocity );
	b3StoreW( lanes->maxDeltaPosition, maxDeltaPosition );
	b3StoreW( lanes->sleepVelocity, sleepVelocity );
}
