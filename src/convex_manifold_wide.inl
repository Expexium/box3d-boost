// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "algorithm.h"
#include "hull.h"
#include "manifold.h"
#include "shape.h"
#include "simd.h"
#include "simd_wide.h"

_Static_assert( B3_SIMD_WIDTH <= B3_HULL_SOA_PADDING, "hull SoA padding must cover the SIMD width" );

static inline int b3RoundUpToSIMDWidth( int count )
{
	return ( count + B3_SIMD_WIDTH - 1 ) & ~( B3_SIMD_WIDTH - 1 );
}

// Transform a SoA point/normal stream (already split into X/Y/Z) by out = -(R*v (+t)).
// The inputs come straight from the hull's stored SoA arrays, so there's no transpose here.
static inline void b3NegativeTransformFromSoA( b3Matrix3 R, b3Vec3 p, const float* inX, const float* inY, const float* inZ, int n,
											   float* outX, float* outY, float* outZ, bool isPoint )
{
	// row-column
	b3Vec3W r0 = { b3SplatW( R.cx.x ), b3SplatW( R.cy.x ), b3SplatW( R.cz.x ) };
	b3Vec3W r1 = { b3SplatW( R.cx.y ), b3SplatW( R.cy.y ), b3SplatW( R.cz.y ) };
	b3Vec3W r2 = { b3SplatW( R.cx.z ), b3SplatW( R.cy.z ), b3SplatW( R.cz.z ) };

	b3Vec3W t = { b3ZeroW(), b3ZeroW(), b3ZeroW() };
	if ( isPoint )
	{
		t = b3SplatVW( p );
	}

	for ( int i = 0; i < n; i += B3_SIMD_WIDTH )
	{
		b3Vec3W v = b3LoadVW( inX + i, inY + i, inZ + i );

		// Rotate one vector per lane
		b3Vec3W out = { b3DotW( r0, v ), b3DotW( r1, v ), b3DotW( r2, v ) };

		if ( isPoint )
		{
			out = b3AddVW( out, t );
		}

		b3StoreVW( outX + i, outY + i, outZ + i, b3NegVW( out ) );
	}
}

_Static_assert( B3_MAX_HULL_VERTICES == 128, "must be 128" );

#define B3_HULL_BIT_COUNT 7

// SIMD support point calculation using a SoA vertex array padded with repeats of the first vertex
// to a multiple of 8.
//
// This minimizes (bias - dot), where the caller is expected to provide a bias that makes this always positive.
// It can be direction dependent. The bias should be just big enough to ensure the value is positive because
// an excessive bias causes a precision loss in the support calculation.
//
// The vertex index is embedded in the low B3_HULL_BIT_COUNT mantissa bits of the value. By minimizing a value that
// is always positive, the minimum carries the smallest index so that padded SoA values will never win. This is the
// purpose of using the bias instead of maximizing the dot directly.
//
// The support is then recomputed exactly as dot(normal, vertex), without the embedded index.
// todo consider using this for GJK
static inline void b3GetSupportWide( b3Vec3 normal, const float* vx, const float* vy, const float* vz, int n, float bias,
									 float* support, int* vertexIndex )
{
	const b3Vec3W normalW = b3SplatVW( normal );
	const b3FloatW biasV = b3SplatW( bias );

	// Start the minimum at a large value.
	b3FloatW minValue = b3SplatW( INFINITY );

	// Tail lanes hold vertex 0 with index bits >= vertexCount, so they never become the min value.
	for ( int i = 0; i < n; i += B3_SIMD_WIDTH )
	{
		b3Vec3W v = b3LoadVW( vx + i, vy + i, vz + i );
		b3FloatW d = b3DotW( normalW, v );

		// This is always positive.
		b3FloatW value = b3SubW( biasV, d );
		b3FloatW augmentedValue = b3EmbedIndexW( value, i, B3_HULL_BIT_COUNT );
		minValue = b3MinW( minValue, augmentedValue );
	}

	// One horizontal min, the winning lane's value and index bits ride through.
	int vi = b3MinIndexW( minValue, B3_HULL_BIT_COUNT );

	// Exact support for the chosen vertex.
	*vertexIndex = vi;

	// Dot product
	*support = normal.x * vx[vi] + normal.y * vy[vi] + normal.z * vz[vi];
}

static inline float b3GetFaceSeparation( b3Vec3 direction, float planeSeparation, const float* vx, const float* vy,
										 const float* vz, int n, b3Vec3 center, b3Vec3 extents, int* vertexIndex )
{
	float bias = b3Dot( direction, center ) + 1.0625f * b3Dot( b3Abs( direction ), extents );
	float support;
	b3GetSupportWide( direction, vx, vy, vz, n, bias, &support, vertexIndex );
	return planeSeparation - support;
}

// Wide dot(n, d) for all face normals n of the hull, padded to the SIMD width.
static inline void b3GetFaceDots( const b3HullData* hull, b3Vec3 d, float* dots )
{
	int faceStride = b3GetHullSoaStride( hull->faceCount );
	const float* nx = b3GetHullSoaNormals( hull );
	const float* ny = nx + faceStride;
	const float* nz = ny + faceStride;

	b3Vec3W dW = b3SplatVW( d );

	int wideFaceCount = b3RoundUpToSIMDWidth( hull->faceCount );
	for ( int i = 0; i < wideFaceCount; i += B3_SIMD_WIDTH )
	{
		// dot product per lane
		b3FloatW m = b3DotW( b3LoadVW( nx + i, ny + i, nz + i ), dW );
		b3StoreW( dots + i, m );
	}
}

#define B3_PARALLEL_TOL 1e-4f

// Inscribed sphere edge test. https://box2d.org/posts/2026/09/inscribed-spheres/
// The implementation here is a mix of the versions from Cairn Overturf and Dirk Gregorius:
// https://gist.github.com/cairnc/dee7a2866da0709f2d9a77b6493b57b5
// https://gist.github.com/dgregorius/e6751b5c00937cd21af63cba3c53c861
// This benchmarks faster than the version from the blog post.
static inline int b3TestEdgeCandidateSorted( float a1, float a2, float c, float bound )
{
	// We want to maximize
	//
	//   f(t) = dot(d, nlerp(n1, n2, t))
	//
	// over t in [0, 1], where
	//
	//   a1 = dot(n1, d)
	//   a2 = dot(n2, d)
	//   c  = dot(n1, n2).
	//
	// The maximum can occur either at an endpoint or at an interior
	// critical point where f'(t) = 0.

	// Since the endpoint values are a1 and a2, only the larger endpoint
	// needs to be tested against the bound.
	float hi = b3MaxFloat( a1, a2 );
	float lo = b3MinFloat( a1, a2 );
	int exterior = hi >= bound;

	// Differentiating f(t) gives
	//
	//           a2 - c*a1 - (1 - c)*(a1 + a2)*t
	//   f'(t) = --------------------------------------
	//           ((1 - t)^2 + t^2 + 2*c*t*(1 - t))^(3/2)
	//
	// The denominator is positive and the numerator is linear in t.
	// An interior maximum therefore requires
	//
	//   f'(0) >= 0  ->  a2 >= c*a1
	//   f'(1) <= 0  ->  a1 >= c*a2.
	//
	// After sorting these become
	//
	//   hi >= c*lo
	//   lo >= c*hi.
	//
	// The second is always the stricter condition since
	//
	//   (hi - c*lo) - (lo - c*hi)
	//       = (1 + c)*(hi - lo) >= 0.
	//
	// Thus both conditions reduce to u >= 0.
	float u = lo - c * hi;
	int maxIsInterior = u >= 0.0f;

	// Solving f'(t) = 0 and evaluating f(t) at tmax gives
	//
	//   f(tmax)^2 =
	//       (a1^2 + a2^2 - 2*c*a1*a2) / (1 - c^2).
	//
	// After sorting,
	//
	//   hi^2 + lo^2 - 2*c*hi*lo
	//       = hi^2*(1 - c^2) + u^2.
	//
	// Let s = 1 - c^2 and rearrange f(tmax) >= bound:
	//
	//   u^2 >= (bound^2 - hi^2)*s.
	float s = 1.0f - c * c;
	float boundTerm = ( bound - hi ) * ( bound + hi );
	float lhs = u * u;
	float rhs = boundTerm * s;
	int maxBeatsBound = lhs >= rhs;

	// The interior expression contains 1 - c^2 in its denominator.
	// When this approaches zero the arc is degenerate or ill-conditioned,
	// so conservatively keep the edge as a candidate.
	int nearlyParallel = s < B3_PARALLEL_TOL;

	int interior = maxIsInterior & ( maxBeatsBound | nearlyParallel );

	return exterior | interior;
}

// dot(n, otherCenter) - offset for all faces of the hull, padded to the SIMD width.
static inline void b3GetFacePlaneSeparations( const b3HullData* hull, b3Vec3 otherCenter, float* separations )
{
	b3GetFaceDots( hull, otherCenter, separations );

	const b3Plane* planes = b3GetHullPlanes( hull );
	int faceCount = hull->faceCount;
	for ( int i = 0; i < faceCount; ++i )
	{
		separations[i] -= planes[i].offset;
	}
}

// Collect the edges of a hull that pass b3TestEdgeCandidateSorted, in order. Returns the count.
// planeDots: the value of each face plane, the first two arguments of the test
static inline int b3GatherEdgeCandidates( const b3HullData* hull, const float* planeDots, float bound, int* edgeIndices )
{
	const b3HullHalfEdge* halfEdges = b3GetHullEdges( hull );
	const float* cosines = b3GetHullEdgeCosines( hull );
	int halfEdgeCount = hull->edgeCount;
	int count = 0;
	int i = 0;

#if B3_SIMD_WIDTH == 8
	// Eight edges at a time. An edge is a pair of half edges and each half edge is four bytes with the
	// face index in the last byte. This computes the same test as b3TestEdgeCandidateSorted with the
	// same operations, so it keeps exactly the same edges.
	_Static_assert( sizeof( b3HullHalfEdge ) == 4 && offsetof( b3HullHalfEdge, face ) == 3, "half edge layout" );

	const __m256i evenThenOdd = _mm256_setr_epi32( 0, 2, 4, 6, 1, 3, 5, 7 );
	const __m256 boundW = _mm256_set1_ps( bound );
	const __m256 oneW = _mm256_set1_ps( 1.0f );
	const __m256 zeroW = _mm256_setzero_ps();
	const __m256 parallelTol = _mm256_set1_ps( B3_PARALLEL_TOL );

	for ( ; i + 16 <= halfEdgeCount; i += 16 )
	{
		// Split the sixteen half edges into the first and the second half edge of each edge
		__m256i h0 = _mm256_permutevar8x32_epi32( _mm256_loadu_si256( (const __m256i*)( halfEdges + i ) ), evenThenOdd );
		__m256i h1 = _mm256_permutevar8x32_epi32( _mm256_loadu_si256( (const __m256i*)( halfEdges + i + 8 ) ), evenThenOdd );
		__m256i face1 = _mm256_srli_epi32( _mm256_permute2x128_si256( h0, h1, 0x20 ), 24 );
		__m256i face2 = _mm256_srli_epi32( _mm256_permute2x128_si256( h0, h1, 0x31 ), 24 );

		__m256 a1 = _mm256_i32gather_ps( planeDots, face1, 4 );
		__m256 a2 = _mm256_i32gather_ps( planeDots, face2, 4 );
		__m256 c = _mm256_loadu_ps( cosines + ( i >> 1 ) );

		// b3MaxFloat( a1, a2 ) and b3MinFloat( a1, a2 )
		__m256 hi = _mm256_blendv_ps( a2, a1, _mm256_cmp_ps( a1, a2, _CMP_GT_OQ ) );
		__m256 lo = _mm256_blendv_ps( a2, a1, _mm256_cmp_ps( a1, a2, _CMP_LT_OQ ) );
		__m256 exterior = _mm256_cmp_ps( hi, boundW, _CMP_GE_OQ );

		__m256 u = _mm256_sub_ps( lo, _mm256_mul_ps( c, hi ) );
		__m256 maxIsInterior = _mm256_cmp_ps( u, zeroW, _CMP_GE_OQ );

		__m256 s = _mm256_sub_ps( oneW, _mm256_mul_ps( c, c ) );
		__m256 boundTerm = _mm256_mul_ps( _mm256_sub_ps( boundW, hi ), _mm256_add_ps( boundW, hi ) );
		__m256 lhs = _mm256_mul_ps( u, u );
		__m256 rhs = _mm256_mul_ps( boundTerm, s );
		__m256 maxBeatsBound = _mm256_cmp_ps( lhs, rhs, _CMP_GE_OQ );
		__m256 nearlyParallel = _mm256_cmp_ps( s, parallelTol, _CMP_LT_OQ );

		__m256 keep = _mm256_or_ps( exterior, _mm256_and_ps( maxIsInterior, _mm256_or_ps( maxBeatsBound, nearlyParallel ) ) );
		int keepMask = _mm256_movemask_ps( keep );

		for ( int lane = 0; lane < 8; ++lane )
		{
			edgeIndices[count] = i + 2 * lane;
			count += ( keepMask >> lane ) & 1;
		}
	}
#endif

	for ( ; i < halfEdgeCount; i += 2 )
	{
		int i1 = halfEdges[i].face;
		int i2 = halfEdges[i + 1].face;
		float c = cosines[i >> 1];
		edgeIndices[count] = i;
		count += b3TestEdgeCandidateSorted( planeDots[i1], planeDots[i2], c, bound );
	}

	return count;
}

// The number of edge pair tests needed to make the extra culling pass worthwhile.
#define B3_EDGE_PROBE_MIN_TESTS 10

// Re-test edge candidates against the plane separations of a probe point on the other hull
static inline int b3FilterEdgeCandidates( const b3HullData* hull, const float* planeSeparations, float bound, int* edgeIndices,
										  int count )
{
	const b3HullHalfEdge* halfEdges = b3GetHullEdges( hull );
	const float* cosines = b3GetHullEdgeCosines( hull );
	int keptCount = 0;

	for ( int k = 0; k < count; ++k )
	{
		int i = edgeIndices[k];
		int i1 = halfEdges[i].face;
		int i2 = halfEdges[i + 1].face;
		float c = cosines[i >> 1];
		edgeIndices[keptCount] = i;
		keptCount += b3TestEdgeCandidateSorted( planeSeparations[i1], planeSeparations[i2], c, bound );
	}

	return keptCount;
}

// Temporary abbreviations for convenience.
#define NE ( B3_MAX_HULL_EDGES + B3_SIMD_WIDTH )
#define NF ( B3_MAX_HULL_FACES + B3_SIMD_WIDTH )
#define NV ( B3_MAX_HULL_VERTICES + B3_SIMD_WIDTH )

// SIMD separating axis test based on an implementation developed by Cairn Overturf.
// See his article: https://cairnc.github.io/posts/improvements-to-the-separating-axis/
b3AxisQuery B3_WIDE( b3ComputeSeparatingAxis )( const b3HullData* hullA, const b3HullData* hullB, b3Transform xfB,
												bool earlyReturn )
{
	b3Matrix3 R = b3MakeMatrixFromQuat( xfB.q );
	b3Matrix3 invR = b3Transpose( R );

	float speculativeDistance = B3_SPECULATIVE_DISTANCE;

	b3AxisQuery res = {
		.faceA =
			{
				.normal = b3Vec3_zero,
				.separation = -INFINITY,
				.indexA = B3_NULL_INDEX,
				.indexB = B3_NULL_INDEX,
				.type = b3_faceAxisA,
			},
		.faceB =
			{
				.normal = b3Vec3_zero,
				.separation = -INFINITY,
				.indexA = B3_NULL_INDEX,
				.indexB = B3_NULL_INDEX,
				.type = b3_faceAxisB,
			},
		.edge =
			{
				.normal = b3Vec3_zero,
				.separation = -INFINITY,
				.indexA = B3_NULL_INDEX,
				.indexB = B3_NULL_INDEX,
				.type = b3_edgePairAxis,
			},
		.separatedFeature = b3_invalidAxis,
	};

	int faceCountA = hullA->faceCount;
	const b3Plane* planesA = b3GetHullPlanes( hullA );

	int vertexStrideB = b3GetHullSoaStride( hullB->vertexCount );
	const float* vxB = b3GetHullSoaVertices( hullB );
	const float* vyB = vxB + vertexStrideB;
	const float* vzB = vyB + vertexStrideB;
	int wideVertexCountB = b3RoundUpToSIMDWidth( hullB->vertexCount );

	b3Vec3 cB = b3AABB_Center( hullB->aabb );
	b3Vec3 hB = b3AABB_Extents( hullB->aabb );

	// The hulls have a precomputed inner radius and centroid.
	// A given axis cannot achieve a separation larger than:
	// dot(axis, centerB - centerA) - innerRadiusA - innerRadiusB
	// So this is the upper bound for the separation of a candidate axis.
	// An axis can be skipped if it has an upper bound that is less than the current
	// best separation. This lets me skip many of the candidates without computing
	// support points.

	b3Vec3 deltaCenter = b3Sub( b3Add( b3MulMV( R, hullB->center ), xfB.p ), hullA->center );
	float centerDistance = b3Length( deltaCenter );
	float radius = hullA->innerRadius + hullB->innerRadius;

	// Adjust the radius to ensure the best axis isn't skipped.
	float radiusBound = radius - ( B3_LINEAR_SLOP + 0.001f * ( centerDistance + radius ) );

	// Compute dot(normalA, centerDelta) for all face normals of hullA.
	_Alignas( B3_WIDE_ALIGNMENT ) float dotA[NF];
	b3GetFaceDots( hullA, deltaCenter, dotA );

	// Find the face of hullA that most aligns with centerDelta.
	int seedIndexA = 0;
	float maxDotA = dotA[0];
	for ( int i = 1; i < faceCountA; ++i )
	{
		if ( dotA[i] > maxDotA )
		{
			maxDotA = dotA[i];
			seedIndexA = i;
		}
	}

	// Use the seed to get a lower bound on the separation for the faces of hullA.
	float floorA = -INFINITY;
	float seedSeparationA = -INFINITY;
	int seedVertexB = 0;
	if ( earlyReturn )
	{
		b3Plane plane = planesA[seedIndexA];
		b3Vec3 direction = b3Neg( b3MulMV( invR, plane.normal ) );
		float planeSeparation = b3Dot( plane.normal, xfB.p ) - plane.offset;
		seedSeparationA = b3GetFaceSeparation( direction, planeSeparation, vxB, vyB, vzB, wideVertexCountB, cB, hB, &seedVertexB );
		floorA = b3MinFloat( seedSeparationA, speculativeDistance );
	}

	// Test A's face planes against B's vertices.
	for ( int i = 0; i < faceCountA; ++i )
	{
		// The bound offset ensures the seed will be evaluated.
		if ( dotA[i] - radiusBound < b3MaxFloat( floorA, res.faceA.separation ) )
		{
			continue;
		}

		b3Plane plane = planesA[i];
		int vertexIndex = seedVertexB;
		float separation = seedSeparationA;

		// Avoid recomputing the seed face separation.
		if ( earlyReturn == false || i != seedIndexA )
		{
			b3Vec3 direction = b3Neg( b3MulMV( invR, plane.normal ) );
			float planeSeparation = b3Dot( plane.normal, xfB.p ) - plane.offset;
			separation = b3GetFaceSeparation( direction, planeSeparation, vxB, vyB, vzB, wideVertexCountB, cB, hB, &vertexIndex );
		}

		if ( separation > res.faceA.separation )
		{
			res.faceA.normal = plane.normal;
			res.faceA.separation = separation;
			res.faceA.indexA = i;
			res.faceA.indexB = vertexIndex;
			if ( separation > speculativeDistance && earlyReturn )
			{
				res.separatedFeature = b3_faceAxisA;
				return res;
			}
		}
	}

	B3_VALIDATE( res.faceA.indexA != B3_NULL_INDEX );

	int faceCountB = hullB->faceCount;
	const b3Plane* planesB = b3GetHullPlanes( hullB );

	int vertexStrideA = b3GetHullSoaStride( hullA->vertexCount );
	const float* vxA = b3GetHullSoaVertices( hullA );
	const float* vyA = vxA + vertexStrideA;
	const float* vzA = vyA + vertexStrideA;
	int wideVertexCountA = b3RoundUpToSIMDWidth( hullA->vertexCount );

	b3Vec3 cA = b3AABB_Center( hullA->aabb );
	b3Vec3 hA = b3AABB_Extents( hullA->aabb );

	// Similarly, find the face of hullB that most aligns with the vector pointing from centerB to centerA.
	_Alignas( B3_WIDE_ALIGNMENT ) float dotB[NF];
	b3GetFaceDots( hullB, b3Neg( b3MulMV( invR, deltaCenter ) ), dotB );

	int seedIndexB = 0;
	float maxDotB = dotB[0];
	for ( int i = 1; i < faceCountB; ++i )
	{
		if ( dotB[i] > maxDotB )
		{
			maxDotB = dotB[i];
			seedIndexB = i;
		}
	}

	// Get a lower bound on the separation for the faces of hullB.
	float floorB = -INFINITY;
	float seedSeparationB = -INFINITY;
	int seedVertexA = 0;
	if ( earlyReturn )
	{
		b3Plane plane = planesB[seedIndexB];
		b3Vec3 direction = b3Neg( b3MulMV( R, plane.normal ) );
		float planeSeparation = b3Dot( direction, xfB.p ) - plane.offset;
		seedSeparationB = b3GetFaceSeparation( direction, planeSeparation, vxA, vyA, vzA, wideVertexCountA, cA, hA, &seedVertexA );

		// Include the floor set by hull A faces.
		floorB = b3MaxFloat( seedSeparationB, res.faceA.separation );
		floorB = b3MinFloat( floorB, speculativeDistance );
	}

	// Test B's face planes against A's vertices.
	for ( int i = 0; i < faceCountB; ++i )
	{
		if ( dotB[i] - radiusBound < b3MaxFloat( floorB, res.faceB.separation ) )
		{
			continue;
		}

		b3Plane plane = planesB[i];
		b3Vec3 direction = b3Neg( b3MulMV( R, plane.normal ) );
		int vertexIndex = seedVertexA;
		float separation = seedSeparationB;

		// Avoid recomputing the seed face separation.
		if ( earlyReturn == false || i != seedIndexB )
		{
			float planeSeparation = b3Dot( direction, xfB.p ) - plane.offset;
			separation = b3GetFaceSeparation( direction, planeSeparation, vxA, vyA, vzA, wideVertexCountA, cA, hA, &vertexIndex );
		}

		if ( separation > res.faceB.separation )
		{
			res.faceB.normal = direction;
			res.faceB.separation = separation;
			res.faceB.indexA = vertexIndex;
			res.faceB.indexB = i;
			if ( separation > speculativeDistance && earlyReturn )
			{
				res.separatedFeature = b3_faceAxisB;
				return res;
			}
		}
	}

	// Transform B into A's space once, into SoA arrays. Extra space so tail can be set to zero.
	_Static_assert( ( B3_MAX_HULL_EDGES & ( B3_SIMD_WIDTH - 1 ) ) == 0, "must be multiple of SIMD width" );
	_Static_assert( ( B3_MAX_HULL_FACES & ( B3_HULL_SOA_PADDING - 1 ) ) == 0, "must be multiple of hull SoA padding" );
	_Static_assert( ( B3_MAX_HULL_VERTICES & ( B3_HULL_SOA_PADDING - 1 ) ) == 0, "must be multiple of hull SoA padding" );

	B3_VALIDATE( earlyReturn == false ||
				 centerDistance >= b3MaxFloat( res.faceA.separation, res.faceB.separation ) + radiusBound );

	// Gather edges of A that can feasibly create a separating axis that beats the maximum face separation.
	const b3HullHalfEdge* halfEdgesA = b3GetHullEdges( hullA );
	int edgeIndicesA[NE];

	const b3HullHalfEdge* halfEdgesB = b3GetHullEdges( hullB );
	int edgeIndicesB[B3_MAX_HULL_EDGES];

	float maxFaceSeparation = b3MaxFloat( res.faceA.separation, res.faceB.separation );
	float boundSlack = radius - radiusBound;
	float thresholdA = earlyReturn ? maxFaceSeparation + hullB->innerRadius - boundSlack : -INFINITY;
	float thresholdB = earlyReturn ? maxFaceSeparation + hullA->innerRadius - boundSlack : -INFINITY;

	b3Vec3 centerBinA = b3Add( b3MulMV( R, hullB->center ), xfB.p );
	b3Vec3 centerAinB = b3MulMV( invR, b3Sub( hullA->center, xfB.p ) );

	_Alignas( B3_WIDE_ALIGNMENT ) float planeDotA[NF];
	_Alignas( B3_WIDE_ALIGNMENT ) float planeDotB[NF];
	b3GetFacePlaneSeparations( hullA, centerBinA, planeDotA );
	b3GetFacePlaneSeparations( hullB, centerAinB, planeDotB );

	int na = b3GatherEdgeCandidates( hullA, planeDotA, thresholdA, edgeIndicesA );
	int nb = b3GatherEdgeCandidates( hullB, planeDotB, thresholdB, edgeIndicesB );

	// Apply additional culling use the support points for the best face axes.
	// The support vertices of the best faces are points on the other hull, so an edge pair axis cannot
	// have a larger separation than the plane separation of these points over the arc of the edge.
	// This can slow down boxes so skip this if there are not enough edge candidates.
	if ( earlyReturn && nb * ( ( na + 3 ) >> 2 ) >= B3_EDGE_PROBE_MIN_TESTS )
	{
		float probeBound = maxFaceSeparation - ( 0.1f * B3_LINEAR_SLOP + 0.001f * b3AbsFloat( centerDistance + radius ) );

		int vertexB = res.faceA.indexB != B3_NULL_INDEX ? res.faceA.indexB : seedVertexB;
		b3Vec3 probeB = { vxB[vertexB], vyB[vertexB], vzB[vertexB] };
		b3GetFacePlaneSeparations( hullA, b3Add( b3MulMV( R, probeB ), xfB.p ), planeDotA );
		na = b3FilterEdgeCandidates( hullA, planeDotA, probeBound, edgeIndicesA, na );

		int vertexA = res.faceB.indexA != B3_NULL_INDEX ? res.faceB.indexA : seedVertexA;
		b3Vec3 probeA = { vxA[vertexA], vyA[vertexA], vzA[vertexA] };
		b3GetFacePlaneSeparations( hullB, b3MulMV( invR, b3Sub( probeA, xfB.p ) ), planeDotB );
		nb = b3FilterEdgeCandidates( hullB, planeDotB, probeBound, edgeIndicesB, nb );
	}

	if ( na == 0 || nb == 0 )
	{
		// No edge candidates found.
		return res;
	}

	// The alignments below are not necessary, but they don't hurt.

	// B face normals in A space, negated.
	_Alignas( B3_WIDE_ALIGNMENT ) float bFNx[NF];
	_Alignas( B3_WIDE_ALIGNMENT ) float bFNy[NF];
	_Alignas( B3_WIDE_ALIGNMENT ) float bFNz[NF];

	// B vertices in A space, negated.
	_Alignas( B3_WIDE_ALIGNMENT ) float bWx[NV];
	_Alignas( B3_WIDE_ALIGNMENT ) float bWy[NV];
	_Alignas( B3_WIDE_ALIGNMENT ) float bWz[NV];

	int faceStrideB = b3GetHullSoaStride( faceCountB );
	const float* nxB = b3GetHullSoaNormals( hullB );
	const float* nyB = nxB + faceStrideB;
	const float* nzB = nyB + faceStrideB;

	int wideFaceCountB = b3RoundUpToSIMDWidth( faceCountB );
	b3NegativeTransformFromSoA( R, xfB.p, nxB, nyB, nzB, wideFaceCountB, bFNx, bFNy, bFNz, false );
	b3NegativeTransformFromSoA( R, xfB.p, vxB, vyB, vzB, wideVertexCountB, bWx, bWy, bWz, true );

	// Per A edge data, already in A's space so just gathered. n0 and n1 are the two face
	// normals, d the edge vector av1-av0, v0 the first vertex. Tol is the
	// parallel edge tolerance, scaled by the edge length.
	_Alignas( B3_WIDE_ALIGNMENT ) float aN0x[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aN0y[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aN0z[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aN1x[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aN1y[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aN1z[NE];
	// dir = av1 - av0
	_Alignas( B3_WIDE_ALIGNMENT ) float aDx[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aDy[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aDz[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aV0x[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aV0y[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aV0z[NE];
	_Alignas( B3_WIDE_ALIGNMENT ) float aTol[NE];

	float squaredTol = B3_PARALLEL_EDGE_TOL * B3_PARALLEL_EDGE_TOL;
	for ( int k = 0; k < na; ++k )
	{
		const b3HullHalfEdge* edge = halfEdgesA + edgeIndicesA[k];
		const b3HullHalfEdge* twin = edge + 1;

		b3Vec3 A = planesA[edge->face].normal;
		b3Vec3 B = planesA[twin->face].normal;
		aN0x[k] = A.x;
		aN0y[k] = A.y;
		aN0z[k] = A.z;
		aN1x[k] = B.x;
		aN1y[k] = B.y;
		aN1z[k] = B.z;

		int v0 = edge->origin;
		int v1 = twin->origin;

		aDx[k] = vxA[v1] - vxA[v0];
		aDy[k] = vyA[v1] - vyA[v0];
		aDz[k] = vzA[v1] - vzA[v0];
		aV0x[k] = vxA[v0];
		aV0y[k] = vyA[v0];
		aV0z[k] = vzA[v0];

		aTol[k] = squaredTol * ( aDx[k] * aDx[k] + aDy[k] * aDy[k] + aDz[k] * aDz[k] );
	}

	// Zero the tail lanes.
	b3FloatW zero = b3ZeroW();
	b3Vec3W zeroV = { zero, zero, zero };
	b3StoreVW( aN0x + na, aN0y + na, aN0z + na, zeroV );
	b3StoreVW( aN1x + na, aN1y + na, aN1z + na, zeroV );
	b3StoreVW( aDx + na, aDy + na, aDz + na, zeroV );
	b3StoreVW( aV0x + na, aV0y + na, aV0z + na, zeroV );

	b3StoreW( aTol + na, zero );

	float linearSlop = B3_LINEAR_SLOP;

#if defined( B3_SIMD_NONE )

	// The SIMD emulated version of this code is very slow. This is a purely scalar version
	// for platforms that don't have SIMD capability. It is much faster than SIMD emulation.
	// WARNING: this math needs to match the SIMD version for cross platform determinism.

	const float EPS = -linearSlop * linearSlop;

	for ( int j = 0; j < nb; ++j )
	{
		const b3HullHalfEdge* edge = halfEdgesB + edgeIndicesB[j];
		const b3HullHalfEdge* twin = edge + 1;
		int f0 = edge->face;
		int f1 = twin->face;
		int v0 = edge->origin;
		int v1 = twin->origin;

		b3Vec3 C = { bFNx[f0], bFNy[f0], bFNz[f0] };
		b3Vec3 D = { bFNx[f1], bFNy[f1], bFNz[f1] };
		b3Vec3 bv0 = { bWx[v0], bWy[v0], bWz[v0] };
		b3Vec3 bv1 = { bWx[v1], bWy[v1], bWz[v1] };
		b3Vec3 DC = b3Sub( bv1, bv0 );

		for ( int i = 0; i < na; ++i )
		{
			b3Vec3 d = { aDx[i], aDy[i], aDz[i] };

			// CBA = C.dir, DBA = D.dir, where dir = B_x_A
			float CBA = b3Dot( C, d );
			float DBA = b3Dot( D, d );
			if ( CBA * DBA >= EPS )
			{
				continue;
			}

			b3Vec3 n0 = { aN0x[i], aN0y[i], aN0z[i] };
			b3Vec3 n1 = { aN1x[i], aN1y[i], aN1z[i] };

			// ADC = n0.DC, BDC = n1.DC, where DC = D_x_C
			float ADC = b3Dot( n0, DC );
			float BDC = b3Dot( n1, DC );
			if ( ADC * BDC >= EPS || CBA * BDC >= EPS )
			{
				continue;
			}

			// Reject near parallel edges
			float maxCD = b3MaxFloat( CBA * CBA, DBA * DBA );
			if ( maxCD <= aTol[i] )
			{
				continue;
			}

			// t = -CBA / (DBA - CBA)
			float t = -CBA / ( DBA - CBA );

			// normal = lerp(t, C, D) = C + (D-C)*t
			b3Vec3 n = b3MulAdd( C, t, b3Sub( D, C ) );
			float len2 = b3Dot( n, n );

			float inv = 1.0f / sqrtf( len2 );
			n = b3MulSV( inv, n );

			// separation = -dot(normal, av0 + bv0)
			b3Vec3 av0 = { aV0x[i], aV0y[i], aV0z[i] };
			float separation = -b3Dot( b3Add( av0, bv0 ), n );

			if ( separation > res.edge.separation )
			{
				res.edge.normal = n;
				res.edge.separation = separation;

				// Half edge index
				res.edge.indexA = edgeIndicesA[i];
				res.edge.indexB = edgeIndicesB[j];

				if ( separation > speculativeDistance && earlyReturn )
				{
					res.separatedFeature = b3_edgePairAxis;
					return res;
				}
			}
		}
	}

#else

	// This tolerance can skip edges shorter than 1cm.
	const b3FloatW EPS = b3SplatW( -linearSlop * linearSlop );
	const b3FloatW INF = b3SplatW( INFINITY );

	for ( int j = 0; j < nb; ++j )
	{
		const b3HullHalfEdge* edge = halfEdgesB + edgeIndicesB[j];
		const b3HullHalfEdge* twin = edge + 1;
		int f0 = edge->face;
		int f1 = twin->face;
		int v0 = edge->origin;
		int v1 = twin->origin;

		b3Vec3 nC = { bFNx[f0], bFNy[f0], bFNz[f0] };
		b3Vec3 nD = { bFNx[f1], bFNy[f1], bFNz[f1] };
		b3Vec3 pB = { bWx[v0], bWy[v0], bWz[v0] };
		b3Vec3 qB = { bWx[v1], bWy[v1], bWz[v1] };

		const b3Vec3W C = b3SplatVW( nC );
		const b3Vec3W D = b3SplatVW( nD );
		const b3Vec3W DC = b3SplatVW( b3Sub( qB, pB ) );
		const b3Vec3W bv0 = b3SplatVW( pB );

		for ( int i = 0; i < na; i += B3_SIMD_WIDTH )
		{
			b3Vec3W n0 = b3LoadVW( aN0x + i, aN0y + i, aN0z + i );
			b3Vec3W n1 = b3LoadVW( aN1x + i, aN1y + i, aN1z + i );
			b3Vec3W d = b3LoadVW( aDx + i, aDy + i, aDz + i );
			b3Vec3W av0 = b3LoadVW( aV0x + i, aV0y + i, aV0z + i );

			b3FloatW tol = b3LoadW( aTol + i );

			// CBA = C.dir, DBA = D.dir, where dir = B_x_A
			b3FloatW CBA = b3DotW( C, d );
			b3FloatW DBA = b3DotW( D, d );
			// ADC = n0.DC, BDC = n1.DC, where DC = D_x_C
			b3FloatW ADC = b3DotW( n0, DC );
			b3FloatW BDC = b3DotW( n1, DC );

			// Gauss map arc crossing test, CBA*DBA<eps and ADC*BDC<eps and CBA*BDC<eps
			b3FloatW m1 = b3LessThanW( b3MulW( CBA, DBA ), EPS );
			b3FloatW m2 = b3LessThanW( b3MulW( ADC, BDC ), EPS );
			b3FloatW m3 = b3LessThanW( b3MulW( CBA, BDC ), EPS );

			// Reject near parallel edges. The arc lerp is ill conditioned when both of B's normals are nearly
			// perpendicular to edge A, a scale invariant sine threshold relative to the edge length.
			b3FloatW maxCD = b3MaxW( b3MulW( CBA, CBA ), b3MulW( DBA, DBA ) );
			b3FloatW notParallel = b3GreaterThanW( maxCD, tol );
			b3FloatW mask = b3AndW( b3AndW( m1, m2 ), b3AndW( m3, notParallel ) );

			// Most A-edges fail the Gauss test, so skip the divide, sqrt and support work when no
			// lane passed.
			if ( b3AnyTrueW( mask ) == false )
			{
				continue;
			}

			// t = -CBA / (DBA - CBA)
			b3FloatW t = b3DivW( b3NegW( CBA ), b3SubW( DBA, CBA ) );

			// normal = lerp(t, C, D) = C + (D-C)*t
			b3Vec3W n = b3MulAddSVW( C, t, b3SubVW( D, C ) );

			// normalize
			b3FloatW len2 = b3DotW( n, n );
			b3FloatW inv = b3DivW( b3SplatW( 1.0f ), b3SqrtW( len2 ) );
			n = b3MulSVW( inv, n );

			// support = dot(normal, av0 + bv0)
			b3FloatW support = b3DotW( b3AddVW( av0, bv0 ), n );

			// Lanes that fail the Gauss test can never win.
			support = b3BlendW( INF, support, mask );
			b3FloatW separation = b3NegW( support );

			// Test all supports against the running best at once. If none beats it, skip the
			// store and scalar reduction.
			b3FloatW improves = b3GreaterThanW( separation, b3SplatW( res.edge.separation ) );
			if ( b3AnyTrueW( improves ) == false )
			{
				continue;
			}

			_Alignas( B3_WIDE_ALIGNMENT ) float sA[B3_SIMD_WIDTH];
			_Alignas( B3_WIDE_ALIGNMENT ) float nxA[B3_SIMD_WIDTH];
			_Alignas( B3_WIDE_ALIGNMENT ) float nyA[B3_SIMD_WIDTH];
			_Alignas( B3_WIDE_ALIGNMENT ) float nzA[B3_SIMD_WIDTH];
			b3StoreW( sA, separation );
			b3StoreVW( nxA, nyA, nzA, n );

			// Reduce in lane order so ties keep the first edge and the early out takes the first
			// improving support below zero. Padded tail lanes carry +INF support, so they never
			// update or index edges out of range.
			for ( int lane = 0; lane < B3_SIMD_WIDTH; lane++ )
			{
				int ei = i + lane;
				float s = sA[lane];
				if ( s > res.edge.separation )
				{
					res.edge.normal = (b3Vec3){ nxA[lane], nyA[lane], nzA[lane] };
					res.edge.separation = s;

					// Half edge index
					res.edge.indexA = edgeIndicesA[ei];
					res.edge.indexB = edgeIndicesB[j];

					if ( s > speculativeDistance && earlyReturn )
					{
						res.separatedFeature = b3_edgePairAxis;
						return res;
					}
				}
			}
		}
	}
#endif

	return res;
}

#undef NE
#undef NF
#undef NV

#if B3_SIMD_WIDTH == 8

// The search of b3QueryTriangleAndHullEdges with eight hull edges per iteration: over all pairs of a hull
// edge and a triangle edge, find the valid pair with the largest separation. The separations are computed
// with the same operations as the scalar code, so this finds the same pair, including the scalar rule
// that the first pair wins a tie. The caller computes the result for that pair with the scalar code.
// Returns 4 * halfEdgeIndex + triangleEdgeIndex, or -1 if no pair is valid.
// processedCount: the number of half edges covered, the caller continues from there
int b3FindTriangleHullEdgePairW8( const b3Vec3* trianglePoints, const b3Vec3* triangleEdges, b3Vec3 triNormal,
								  const b3HullData* hull, int* processedCount )
{
	_Static_assert( sizeof( b3HullHalfEdge ) == 4 && offsetof( b3HullHalfEdge, origin ) == 2 && offsetof( b3HullHalfEdge, face ) == 3,
					"half edge layout" );
	_Static_assert( sizeof( b3Vec3 ) == 12 && sizeof( b3Plane ) == 16 && offsetof( b3Plane, normal ) == 0, "hull data layout" );

	const b3HullHalfEdge* hullEdges = b3GetHullEdges( hull );
	const float* points = (const float*)b3GetHullPoints( hull );
	const float* planes = (const float*)b3GetHullPlanes( hull );
	int edgeCount = hull->edgeCount;
	float squaredTolerance = 0.005f * 0.005f;

	const __m256i evenThenOdd = _mm256_setr_epi32( 0, 2, 4, 6, 1, 3, 5, 7 );
	const __m256i byteMask = _mm256_set1_epi32( 0xFF );
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps( 1.0f );
	const __m256 minLengthSq = _mm256_set1_ps( 1000.0f * FLT_MIN );

	// Per triangle edge constants
	__m256 edgeX[3], edgeY[3], edgeZ[3], pointX[3], pointY[3], pointZ[3], tolerance[3];
	for ( int j = 0; j < 3; ++j )
	{
		edgeX[j] = _mm256_set1_ps( triangleEdges[j].x );
		edgeY[j] = _mm256_set1_ps( triangleEdges[j].y );
		edgeZ[j] = _mm256_set1_ps( triangleEdges[j].z );
		pointX[j] = _mm256_set1_ps( trianglePoints[j].x );
		pointY[j] = _mm256_set1_ps( trianglePoints[j].y );
		pointZ[j] = _mm256_set1_ps( trianglePoints[j].z );
		tolerance[j] = _mm256_set1_ps( squaredTolerance * b3LengthSquared( triangleEdges[j] ) );
	}

	const __m256 normalX = _mm256_set1_ps( triNormal.x );
	const __m256 normalY = _mm256_set1_ps( triNormal.y );
	const __m256 normalZ = _mm256_set1_ps( triNormal.z );

	__m256 bestSeparation = _mm256_set1_ps( -INFINITY );
	__m256 bestCode = _mm256_set1_ps( -1.0f );

	// The code of the pair in each lane for triangle edge 0 of the current group
	__m256 code = _mm256_setr_ps( 0.0f, 8.0f, 16.0f, 24.0f, 32.0f, 40.0f, 48.0f, 56.0f );
	const __m256 codeStep = _mm256_set1_ps( 64.0f );

	int i = 0;
	for ( ; i + 16 <= edgeCount; i += 16 )
	{
		// Split the sixteen half edges into the first and the second half edge of each edge
		__m256i h0 = _mm256_permutevar8x32_epi32( _mm256_loadu_si256( (const __m256i*)( hullEdges + i ) ), evenThenOdd );
		__m256i h1 = _mm256_permutevar8x32_epi32( _mm256_loadu_si256( (const __m256i*)( hullEdges + i + 8 ) ), evenThenOdd );
		__m256i edge = _mm256_permute2x128_si256( h0, h1, 0x20 );
		__m256i twin = _mm256_permute2x128_si256( h0, h1, 0x31 );

		// Indices into the point and plane arrays, in floats
		__m256i origin1 = _mm256_and_si256( _mm256_srli_epi32( edge, 16 ), byteMask );
		__m256i origin2 = _mm256_and_si256( _mm256_srli_epi32( twin, 16 ), byteMask );
		origin1 = _mm256_add_epi32( origin1, _mm256_slli_epi32( origin1, 1 ) );
		origin2 = _mm256_add_epi32( origin2, _mm256_slli_epi32( origin2, 1 ) );
		__m256i face1 = _mm256_slli_epi32( _mm256_srli_epi32( edge, 24 ), 2 );
		__m256i face2 = _mm256_slli_epi32( _mm256_srli_epi32( twin, 24 ), 2 );

		__m256 hullPointX = _mm256_i32gather_ps( points + 0, origin1, 4 );
		__m256 hullPointY = _mm256_i32gather_ps( points + 1, origin1, 4 );
		__m256 hullPointZ = _mm256_i32gather_ps( points + 2, origin1, 4 );

		// hullEdge = b3Sub( hullPoints[twin->origin], hullPoint )
		__m256 hullEdgeX = _mm256_sub_ps( _mm256_i32gather_ps( points + 0, origin2, 4 ), hullPointX );
		__m256 hullEdgeY = _mm256_sub_ps( _mm256_i32gather_ps( points + 1, origin2, 4 ), hullPointY );
		__m256 hullEdgeZ = _mm256_sub_ps( _mm256_i32gather_ps( points + 2, origin2, 4 ), hullPointZ );

		__m256 normal1X = _mm256_i32gather_ps( planes + 0, face1, 4 );
		__m256 normal1Y = _mm256_i32gather_ps( planes + 1, face1, 4 );
		__m256 normal1Z = _mm256_i32gather_ps( planes + 2, face1, 4 );
		__m256 normal2X = _mm256_i32gather_ps( planes + 0, face2, 4 );
		__m256 normal2Y = _mm256_i32gather_ps( planes + 1, face2, 4 );
		__m256 normal2Z = _mm256_i32gather_ps( planes + 2, face2, 4 );

		// bcd = b3Dot( triNormal, hullEdge )
		__m256 bcd = _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps( normalX, hullEdgeX ), _mm256_mul_ps( normalY, hullEdgeY ) ),
									_mm256_mul_ps( normalZ, hullEdgeZ ) );

		for ( int j = 0; j < 3; ++j )
		{
			// cab = b3Dot( hullNormal1, triEdge ), dab = b3Dot( hullNormal2, triEdge )
			__m256 cab = _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps( normal1X, edgeX[j] ), _mm256_mul_ps( normal1Y, edgeY[j] ) ),
										_mm256_mul_ps( normal1Z, edgeZ[j] ) );
			__m256 dab = _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps( normal2X, edgeX[j] ), _mm256_mul_ps( normal2Y, edgeY[j] ) ),
										_mm256_mul_ps( normal2Z, edgeZ[j] ) );

			// if ( cab * dab >= 0.0f || cab * bcd <= 0.0f ) continue
			__m256 skip = _mm256_or_ps( _mm256_cmp_ps( _mm256_mul_ps( cab, dab ), zero, _CMP_GE_OQ ),
										_mm256_cmp_ps( _mm256_mul_ps( cab, bcd ), zero, _CMP_LE_OQ ) );

			// if ( b3MaxFloat( cab * cab, dab * dab ) < squaredTolerance * b3LengthSquared( triEdge ) ) continue
			__m256 cabSq = _mm256_mul_ps( cab, cab );
			__m256 dabSq = _mm256_mul_ps( dab, dab );
			__m256 maxSq = _mm256_blendv_ps( dabSq, cabSq, _mm256_cmp_ps( cabSq, dabSq, _CMP_GT_OQ ) );
			skip = _mm256_or_ps( skip, _mm256_cmp_ps( maxSq, tolerance[j], _CMP_LT_OQ ) );

			// axis = b3Normalize( b3Lerp( hullNormal1, hullNormal2, cab / ( cab - dab ) ) )
			__m256 t = _mm256_div_ps( cab, _mm256_sub_ps( cab, dab ) );
			__m256 s = _mm256_sub_ps( one, t );
			__m256 axisX = _mm256_add_ps( _mm256_mul_ps( s, normal1X ), _mm256_mul_ps( t, normal2X ) );
			__m256 axisY = _mm256_add_ps( _mm256_mul_ps( s, normal1Y ), _mm256_mul_ps( t, normal2Y ) );
			__m256 axisZ = _mm256_add_ps( _mm256_mul_ps( s, normal1Z ), _mm256_mul_ps( t, normal2Z ) );

			__m256 lengthSq = _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps( axisX, axisX ), _mm256_mul_ps( axisY, axisY ) ),
											 _mm256_mul_ps( axisZ, axisZ ) );
			__m256 canNormalize = _mm256_cmp_ps( lengthSq, minLengthSq, _CMP_GT_OQ );
			__m256 scale = _mm256_div_ps( one, _mm256_sqrt_ps( lengthSq ) );
			axisX = _mm256_and_ps( _mm256_mul_ps( scale, axisX ), canNormalize );
			axisY = _mm256_and_ps( _mm256_mul_ps( scale, axisY ), canNormalize );
			axisZ = _mm256_and_ps( _mm256_mul_ps( scale, axisZ ), canNormalize );

			// separation = b3Dot( axis, b3Sub( trianglePoints[j], hullPoint ) )
			__m256 separation =
				_mm256_add_ps( _mm256_add_ps( _mm256_mul_ps( axisX, _mm256_sub_ps( pointX[j], hullPointX ) ),
											  _mm256_mul_ps( axisY, _mm256_sub_ps( pointY[j], hullPointY ) ) ),
							   _mm256_mul_ps( axisZ, _mm256_sub_ps( pointZ[j], hullPointZ ) ) );

			// if ( separation > result.separation ) take this pair
			__m256 take = _mm256_andnot_ps( skip, _mm256_cmp_ps( separation, bestSeparation, _CMP_GT_OQ ) );
			bestSeparation = _mm256_blendv_ps( bestSeparation, separation, take );
			bestCode = _mm256_blendv_ps( bestCode, _mm256_add_ps( code, _mm256_set1_ps( (float)j ) ), take );
		}

		code = _mm256_add_ps( code, codeStep );
	}

	*processedCount = i;

	// Each lane holds its largest separation and the first pair that reached it. Take the largest
	// across the lanes and then the first pair among the lanes that have it.
	__m128 max4 = _mm_max_ps( _mm256_castps256_ps128( bestSeparation ), _mm256_extractf128_ps( bestSeparation, 1 ) );
	__m128 max2 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
	__m128 max1 = _mm_max_ss( max2, _mm_shuffle_ps( max2, max2, 1 ) );
	__m256 isBest = _mm256_cmp_ps( bestSeparation, _mm256_broadcastss_ps( max1 ), _CMP_EQ_OQ );

	// A lane without a valid pair has the code -1, which must not win
	isBest = _mm256_and_ps( isBest, _mm256_cmp_ps( bestCode, zero, _CMP_GE_OQ ) );
	__m256 candidates = _mm256_blendv_ps( _mm256_set1_ps( FLT_MAX ), bestCode, isBest );
	__m128 code4 = _mm_min_ps( _mm256_castps256_ps128( candidates ), _mm256_extractf128_ps( candidates, 1 ) );
	__m128 code2 = _mm_min_ps( code4, _mm_movehl_ps( code4, code4 ) );
	__m128 code1 = _mm_min_ss( code2, _mm_shuffle_ps( code2, code2, 1 ) );

	float best = _mm_cvtss_f32( code1 );
	return best == FLT_MAX ? -1 : (int)best;
}

// b3FindHullSupportVertex with eight vertices per iteration. Returns the same vertex: the one with
// the largest dot product and, among equals, the lowest index. The vertex arrays are padded to a
// multiple of eight with copies of the first vertex, which lose every tie because of their higher index.
int b3FindHullSupportVertexW8( const b3HullData* hull, b3Vec3 direction )
{
	int vertexCount = hull->vertexCount;
	const float* vx = b3GetHullSoaVertices( hull );

	int soaVertexCount = b3GetHullSoaStride( vertexCount );
	const float* vy = vx + soaVertexCount;
	const float* vz = vy + soaVertexCount;

	b3FloatW dx = b3SplatW( direction.x );
	b3FloatW dy = b3SplatW( direction.y );
	b3FloatW dz = b3SplatW( direction.z );
	b3FloatW eight = b3SplatW( 8.0f );
	b3FloatW index = b3SetW( 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f );
	b3FloatW bestDotW = b3SplatW( -FLT_MAX );
	b3FloatW bestIndexW = b3SplatW( -1.0f );

	int loopCount = b3RoundUpToSIMDWidth( vertexCount );
	for ( int i = 0; i < loopCount; i += 8 )
	{
		b3FloatW dot = b3AddW( b3AddW( b3MulW( dx, b3LoadW( vx + i ) ), b3MulW( dy, b3LoadW( vy + i ) ) ), b3MulW( dz, b3LoadW( vz + i ) ) );
		b3FloatW mask = b3GreaterThanW( dot, bestDotW );
		bestDotW = b3BlendW( bestDotW, dot, mask );
		bestIndexW = b3BlendW( bestIndexW, index, mask );
		index = b3AddW( index, eight );
	}

	// Each lane holds its largest dot product and the first index that reached it. Take the largest
	// across the lanes and then the lowest index among the lanes that have it.
	__m128 dot4 = _mm_max_ps( _mm256_castps256_ps128( bestDotW ), _mm256_extractf128_ps( bestDotW, 1 ) );
	__m128 dot2 = _mm_max_ps( dot4, _mm_movehl_ps( dot4, dot4 ) );
	__m128 dot1 = _mm_max_ss( dot2, _mm_shuffle_ps( dot2, dot2, 1 ) );
	__m256 bestDot = _mm256_broadcastss_ps( dot1 );

	__m256 isBest = _mm256_cmp_ps( bestDotW, bestDot, _CMP_EQ_OQ );
	__m256 candidates = _mm256_blendv_ps( _mm256_set1_ps( FLT_MAX ), bestIndexW, isBest );
	__m128 index4 = _mm_min_ps( _mm256_castps256_ps128( candidates ), _mm256_extractf128_ps( candidates, 1 ) );
	__m128 index2 = _mm_min_ps( index4, _mm_movehl_ps( index4, index4 ) );
	__m128 index1 = _mm_min_ss( index2, _mm_shuffle_ps( index2, index2, 1 ) );

	int bestIndex = _mm_cvttss_si32( index1 );
	B3_ASSERT( 0 <= bestIndex && bestIndex < vertexCount );
	return bestIndex;
}

#endif
