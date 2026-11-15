// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "destruction_scenes.h"
#include "gfx/debug_adapter.h"
#include "imgui.h"
#include "sample.h"

#include "box3d/box3d.h"

// Shared parts of the destruction samples: the stats text, the draw toggles and a ball to throw.
class DestructionSample : public Sample
{
public:
	explicit DestructionSample( SampleContext* context )
		: Sample( context )
	{
		m_ballSpeed = 30.0f;
		m_ballRadius = 0.5f;
	}

	// Counters and timings of the destruction solver
	void DrawStats()
	{
		b3DestructionCounters counters = b3World_GetDestructionCounters( m_worldId );
		b3DestructionProfile profile = b3World_GetDestructionProfile( m_worldId );

		DrawTextLine( "bodies %d, pieces %d, intact bonds %d", counters.bodyCount, counters.pieceCount, counters.bondCount );
		DrawTextLine( "solved pieces %d, iterations %d, broken bonds %d, new bodies %d, pending splits %d", counters.solvedPieceCount,
					  counters.iterationCount, counters.brokenBondCount, counters.newBodyCount, counters.pendingSplitCount );
		DrawTextLine( "destruction %.2f ms = loads %.2f + solve %.2f + fracture %.2f", profile.total, profile.loads, profile.solve,
					  profile.fracture );
	}

	// Toggles every destruction sample has. The stress view colors each bond by how close it is to breaking.
	void DrawCommonControls()
	{
		b3DebugDraw* draw = GetGuiDraw();
		ImGui::Checkbox( "Draw bonds", &draw->drawBonds );
		ImGui::Checkbox( "Draw bond stress", &draw->drawBondStress );

		b3DestructionSettings settings = b3World_GetDestructionSettings( m_worldId );
		bool changed = false;
		changed |= ImGui::Checkbox( "Stress solve", &settings.enableStress );
		changed |= ImGui::SliderFloat( "Impact fracture", &settings.impactFractureScale, 0.0f, 8.0f, "%.1f" );
		changed |= ImGui::SliderInt( "Work budget", &settings.workBudget, 10000, 1000000 );
		if ( changed )
		{
			b3World_SetDestructionSettings( m_worldId, &settings );
		}

		ImGui::SliderFloat( "Ball speed", &m_ballSpeed, 5.0f, 100.0f, "%.0f m/s" );
		ImGui::SliderFloat( "Ball radius", &m_ballRadius, 0.2f, 1.5f, "%.1f m" );
	}

	// Throw a heavy ball from a point toward a target
	void ThrowBall( b3Pos from, b3Pos target )
	{
		b3Vec3 direction = b3Normalize( b3Vec3{ float( target.x - from.x ), float( target.y - from.y ), float( target.z - from.z ) } );

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = from;
		bodyDef.linearVelocity = b3MulSV( m_ballSpeed, direction );
		bodyDef.isBullet = true;
		b3BodyId ballId = b3CreateBody( m_worldId, &bodyDef );

		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.density = 7800.0f;
		b3Sphere sphere = { b3Vec3_zero, m_ballRadius };
		b3CreateSphereShape( ballId, &shapeDef, &sphere );
	}

	float m_ballSpeed;
	float m_ballRadius;
};

// The primary scene. Buildings stand until charges take out their ground floor columns, line by line,
// then they come down floor by floor and end as rubble.
class DestructionBuilding : public DestructionSample
{
public:
	explicit DestructionBuilding( SampleContext* context )
		: DestructionSample( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 35.0f, -15.0f, 70.0f, { 0.0f, 8.0f, 0.0f } );
		}

		m_shadowSplitFar = 150.0f;
		m_buildingCount = 1;
		m_floorCount = m_isDebug ? 2 : 5;
		m_autoDetonate = true;
		m_detonateStep = -1;
		m_isCreated = false;

		CreateScene();
	}

	~DestructionBuilding() override
	{
		DestroyScene();
	}

	void CreateScene()
	{
		DestroyScene();

		SetBuildingCollapseScale( m_buildingCount, m_floorCount );

		b3Capacity capacity = {};
		GetBuildingCollapseCapacity( &capacity );
		CreateWorld( &capacity );

		CreateBuildingCollapse( m_worldId );
		m_isCreated = true;
		m_detonateStep = -1;
	}

	void DestroyScene()
	{
		// The structure must outlive the destructibles, which this takes care of
		if ( m_isCreated )
		{
			DestroyBuildingCollapse();
			m_isCreated = false;
		}
	}

	void Step() override
	{
		if ( m_autoDetonate && m_detonateStep < 0 && m_stepCount >= 60 )
		{
			m_detonateStep = m_stepCount;
		}

		if ( m_detonateStep >= 0 )
		{
			// The scene sets its first charge off at step 30 of its own clock
			StepBuildingCollapse( m_worldId, 30 + m_stepCount - m_detonateStep );
		}

		Sample::Step();
		DrawStats();
	}

	bool DrawControls() override
	{
		bool rebuild = false;
		rebuild |= ImGui::SliderInt( "Buildings", &m_buildingCount, 1, 8 );
		rebuild |= ImGui::SliderInt( "Floors", &m_floorCount, 1, 16 );
		if ( rebuild )
		{
			CreateScene();
		}

		ImGui::Checkbox( "Auto detonate", &m_autoDetonate );
		if ( ImGui::Button( "Detonate" ) && m_detonateStep < 0 )
		{
			m_detonateStep = m_stepCount;
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Throw ball" ) )
		{
			ThrowBall( { 0.0f, 0.5f * 3.2f * float( m_floorCount ), 40.0f }, { 0.0f, 0.5f * 3.2f * float( m_floorCount ), 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionBuilding( context );
	}

	int m_buildingCount;
	int m_floorCount;
	int m_detonateStep;
	bool m_autoDetonate;
	bool m_isCreated;
};

static int sampleDestructionBuilding = RegisterSample( "Destruction", "Building", DestructionBuilding::Create );

// A destruction sample that owns one structure and one destructible made from it.
class SingleDestructible : public DestructionSample
{
public:
	explicit SingleDestructible( SampleContext* context )
		: DestructionSample( context )
	{
		m_structure = nullptr;
		m_destructibleId = b3_nullDestructibleId;
	}

	~SingleDestructible() override
	{
		DestroyDestructible();
	}

	// The structure must outlive the destructible
	void DestroyDestructible()
	{
		if ( b3Destructible_IsValid( m_destructibleId ) )
		{
			b3DestroyDestructible( m_destructibleId );
		}

		m_destructibleId = b3_nullDestructibleId;

		if ( m_structure != nullptr )
		{
			b3DestroyStructure( m_structure );
			m_structure = nullptr;
		}
	}

	// Start over with a fresh world and ground
	void ResetWorld()
	{
		DestroyDestructible();

		b3Capacity capacity = {};
		CreateWorld( &capacity );
		AddGroundBox( 100.0f );
	}

	void Step() override
	{
		Sample::Step();
		DrawStats();
	}

	b3StructureData* m_structure;
	b3DestructibleId m_destructibleId;
};

// A free standing wall. Blast a hole in it or throw a ball at it.
class DestructionWall : public SingleDestructible
{
public:
	explicit DestructionWall( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 20.0f, -10.0f, 22.0f, { 0.0f, 2.5f, 0.0f } );
		}

		m_blastRadius = 1.5f;
		m_blastImpulse = 8000.0f;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.5f;
		fracture.childSize = 0.25f;
		fracture.tilt = 0.2f;
		m_structure = CreateWallStructure( 12.0f, 5.0f, 0.3f, &fracture, 21 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "wall";
		m_destructibleId = b3CreateDestructible( m_worldId, &def );
	}

	bool DrawControls() override
	{
		ImGui::SliderFloat( "Blast radius", &m_blastRadius, 0.5f, 4.0f, "%.1f m" );
		ImGui::SliderFloat( "Blast impulse", &m_blastImpulse, 0.0f, 30000.0f, "%.0f" );

		if ( ImGui::Button( "Blast" ) )
		{
			// In front of the wall, so the debris flies out the back
			ApplyBlast( m_worldId, { 0.0f, 2.5f, 0.4f }, m_blastRadius, 2.0e5f, m_blastImpulse );
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Throw ball" ) )
		{
			ThrowBall( { 0.0f, 2.5f, 25.0f }, { 0.0f, 2.5f, 0.0f } );
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Reset" ) )
		{
			CreateScene();
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionWall( context );
	}

	float m_blastRadius;
	float m_blastImpulse;
};

static int sampleDestructionWall = RegisterSample( "Destruction", "Wall", DestructionWall::Create );

// A beam fixed at one end. Make it longer until its own weight snaps it at the root, where the
// bending is the largest.
class DestructionCantilever : public SingleDestructible
{
public:
	explicit DestructionCantilever( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 20.0f, -10.0f, 25.0f, { 5.0f, 5.0f, 0.0f } );
			GetGuiDraw()->drawBonds = true;
			GetGuiDraw()->drawBondStress = true;
		}

		m_length = 6.0f;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		b3StructureMaterial material = b3DefaultStructureMaterial();
		int cellCount = b3MaxInt( 2, int( m_length / 0.5f ) );
		m_structure = CreateBeamStructure( m_length, 0.5f, 0.5f, cellCount, true, &material );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, 6.0f, 0.0f };
		def.name = "beam";
		m_destructibleId = b3CreateDestructible( m_worldId, &def );
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Length", &m_length, 2.0f, 20.0f, "%.1f m" ) )
		{
			CreateScene();
		}

		if ( ImGui::Button( "Throw ball" ) )
		{
			ThrowBall( { 0.75f * m_length, 6.0f, 20.0f }, { 0.75f * m_length, 6.0f, 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionCantilever( context );
	}

	float m_length;
};

static int sampleDestructionCantilever = RegisterSample( "Destruction", "Cantilever", DestructionCantilever::Create );

// A tall tower. Blow out one side of its base and it leans over, and the bending breaks it apart
// on the way down.
class DestructionTower : public SingleDestructible
{
public:
	explicit DestructionTower( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, -10.0f, 60.0f, { 0.0f, 12.0f, 0.0f } );
		}

		m_shadowSplitFar = 120.0f;
		m_height = 24.0f;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 1.0f;
		fracture.childSize = 0.5f;
		m_structure = CreateTowerStructure( 4.0f, m_height, &fracture, 9 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "tower";
		m_destructibleId = b3CreateDestructible( m_worldId, &def );
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Height", &m_height, 8.0f, 48.0f, "%.0f m" ) )
		{
			CreateScene();
		}

		if ( ImGui::Button( "Blast base" ) )
		{
			ApplyBlast( m_worldId, { 2.0f, 1.0f, 0.0f }, 2.5f, 2.0e6f, 6000.0f );
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Throw ball" ) )
		{
			ThrowBall( { 0.0f, 0.6f * m_height, 40.0f }, { 0.0f, 0.6f * m_height, 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionTower( context );
	}

	float m_height;
};

static int sampleDestructionTower = RegisterSample( "Destruction", "Tower", DestructionTower::Create );

// A loose boulder. A short drop leaves it whole, a long one has the energy to shatter it.
class DestructionBoulder : public SingleDestructible
{
public:
	explicit DestructionBoulder( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 20.0f, -10.0f, 40.0f, { 0.0f, 8.0f, 0.0f } );
		}

		m_dropHeight = 20.0f;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		b3StructureMaterial material = b3DefaultStructureMaterial();
		m_structure = CreateBoulderStructure( 1.5f, 0.6f, &material, 3 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, m_dropHeight + 1.5f, 0.0f };
		def.anchorToStatic = false;
		def.name = "boulder";
		m_destructibleId = b3CreateDestructible( m_worldId, &def );
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Drop height", &m_dropHeight, 1.0f, 40.0f, "%.0f m" ) )
		{
			CreateScene();
		}

		if ( ImGui::Button( "Drop again" ) )
		{
			CreateScene();
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionBoulder( context );
	}

	float m_dropHeight;
};

static int sampleDestructionBoulder = RegisterSample( "Destruction", "Boulder", DestructionBoulder::Create );
