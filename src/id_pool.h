// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "container.h"

typedef struct b3IdPool
{
	b3Array( int ) freeArray;
	int nextIndex;
} b3IdPool;

b3IdPool b3CreateIdPool( void );

// The id b3AllocId will return after `skip` more calls, as long as no id is freed before then.
// Useful for prefetching the slot of an id before it is allocated.
static inline int b3PeekId( const b3IdPool* pool, int skip )
{
	int freeCount = pool->freeArray.count;
	return skip < freeCount ? pool->freeArray.data[freeCount - 1 - skip] : pool->nextIndex + ( skip - freeCount );
}
void b3DestroyIdPool( b3IdPool* pool );

int b3AllocId( b3IdPool* pool );
void b3FreeId( b3IdPool* pool, int id );
void b3ValidateFreeId( const b3IdPool* pool, int id );

static inline int b3GetIdCount( const b3IdPool* pool )
{
	return pool->nextIndex - pool->freeArray.count;
}

static inline int b3GetIdCapacity( const b3IdPool* pool )
{
	return pool->nextIndex;
}

static inline int b3GetIdBytes( const b3IdPool* pool )
{
	return b3Array_ByteCount( pool->freeArray );
}
