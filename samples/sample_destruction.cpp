// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "destruction_scenes.h"
#include "gfx/debug_adapter.h"
#include "gfx/keycodes.h"
#include "imgui.h"
#include "sample.h"

#include "box3d/box3d.h"

#include <math.h>

// Shared parts of the destruction samples: the stats text and the controls every scene has.
// A scene registers its destructibles and names its strength layers, and gets sliders for them.
class DestructionSample : public Sample
{
public:
	explicit DestructionSample( SampleContext* context )
		: Sample( context )
	{
		m_ballSpeed = 30.0f;
		m_ballRadius = 0.5f;
		m_explosionRadius = 2.5f;
		m_explosionDamage = 1.0e6f;
		m_explosionImpulse = 8000.0f;
		m_strengthScale = 1.0f;
		m_destructibleCount = 0;
		m_layerCount = 0;

		for ( int i = 0; i < B3_MAX_STRENGTH_LAYERS; ++i )
		{
			m_layerNames[i] = "Layer";
			m_layerStrength[i] = 1.0f;
		}
	}

	// The scene tells the controls which destructibles it made
	void ClearDestructibles()
	{
		m_destructibleCount = 0;
	}

	void AddDestructible( b3DestructibleId destructibleId )
	{
		if ( m_destructibleCount < m_maxDestructibles && b3Destructible_IsValid( destructibleId ) )
		{
			m_destructibles[m_destructibleCount++] = destructibleId;
		}
	}

	// Name the strength layers from the coarsest to the finest. The last one is the inside of the
	// fragments.
	void SetLayerNames( const char* const* names, int count )
	{
		m_layerCount = count < B3_MAX_STRENGTH_LAYERS ? count : B3_MAX_STRENGTH_LAYERS;
		for ( int i = 0; i < m_layerCount; ++i )
		{
			m_layerNames[i] = names[i];
		}
	}

	// Read the sliders back from the first destructible, after a scene was built
	void SyncControls()
	{
		if ( m_destructibleCount == 0 )
		{
			return;
		}

		int structureLayers = b3Destructible_GetStructure( m_destructibles[0] )->strengthLayerCount;
		if ( m_layerCount > structureLayers )
		{
			m_layerCount = structureLayers;
		}

		for ( int i = 0; i < m_layerCount; ++i )
		{
			m_layerStrength[i] = b3Destructible_GetLayerStrength( m_destructibles[0], i );
		}

		m_strengthScale = b3Destructible_GetStrengthScale( m_destructibles[0] );
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

	// How the structure comes apart. Weak outer layers and strong inner ones leave large pieces,
	// equal layers grind it down to single fragments and debris.
	void DrawLayerControls()
	{
		ImGui::SeparatorText( "Connection strength" );

		for ( int i = 0; i < m_layerCount; ++i )
		{
			ImGui::PushID( i );
			if ( ImGui::SliderFloat( m_layerNames[i], m_layerStrength + i, 0.05f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic ) )
			{
				for ( int j = 0; j < m_destructibleCount; ++j )
				{
					if ( b3Destructible_IsValid( m_destructibles[j] ) )
					{
						b3Destructible_SetLayerStrength( m_destructibles[j], i, m_layerStrength[i] );
					}
				}
			}
			ImGui::PopID();
		}

		if ( ImGui::SliderFloat( "Everything", &m_strengthScale, 0.05f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic ) )
		{
			for ( int j = 0; j < m_destructibleCount; ++j )
			{
				if ( b3Destructible_IsValid( m_destructibles[j] ) )
				{
					b3Destructible_SetStrengthScale( m_destructibles[j], m_strengthScale );
				}
			}
		}
	}

	// Settings of the solver, the debug draw and the ball. Every destruction sample has these.
	void DrawCommonControls()
	{
		DrawLayerControls();

		ImGui::SeparatorText( "Solver" );

		b3DestructionSettings settings = b3World_GetDestructionSettings( m_worldId );
		bool changed = false;
		changed |= ImGui::Checkbox( "Stress solve", &settings.enableStress );
		changed |= ImGui::SliderFloat( "Impact fracture", &settings.impactFractureScale, 0.0f, 8.0f, "%.1f" );
		changed |= ImGui::SliderFloat( "Impact crush", &settings.impactDamageScale, 0.0f, 8.0f, "%.1f" );
		changed |= ImGui::SliderFloat( "Momentum transfer", &settings.momentumTransferScale, 0.0f, 1.0f, "%.2f" );
		changed |= ImGui::SliderInt( "Work budget", &settings.workBudget, 10000, 1000000 );
		changed |= ImGui::SliderInt( "New bodies per step", &settings.maxNewBodiesPerStep, 8, 1024 );
		if ( changed )
		{
			b3World_SetDestructionSettings( m_worldId, &settings );
		}

		ImGui::SeparatorText( "View" );

		// The stress view colors each bond by how close it is to breaking
		b3DebugDraw* draw = GetGuiDraw();
		ImGui::Checkbox( "Draw bonds", &draw->drawBonds );
		ImGui::Checkbox( "Draw bond stress", &draw->drawBondStress );

		ImGui::SeparatorText( "Ball: shift + click" );
		ImGui::SliderFloat( "Ball speed", &m_ballSpeed, 5.0f, 100.0f, "%.0f m/s" );
		ImGui::SliderFloat( "Ball radius", &m_ballRadius, 0.2f, 1.5f, "%.1f m" );

		ImGui::SeparatorText( "Explosion: ctrl + shift + click" );
		ImGui::SliderFloat( "Explosion radius", &m_explosionRadius, 0.5f, 8.0f, "%.1f m" );
		ImGui::SliderFloat( "Explosion damage", &m_explosionDamage, 0.0f, 1.0e7f, "%.0f J" );
		ImGui::SliderFloat( "Explosion impulse", &m_explosionImpulse, 0.0f, 30000.0f, "%.0f" );

		ImGui::SeparatorText( "Drag: ctrl + drag" );
		ImGui::SliderFloat( "Drag force", &m_mouseForceScale, 1.0f, 1000.0f, "%.0f x weight", ImGuiSliderFlags_Logarithmic );
	}

	// Throw a steel ball from a point toward a target
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

	// Mouse controls of every destruction sample:
	// - shift + click fires a steel ball from the camera at the point under the cursor
	// - ctrl + shift + click sets off an explosion at the point under the cursor (alt is the camera)
	// - ctrl + drag grabs a piece and drags it, which the base sample provides
	void MouseDown( b3Vec2 p, int button, int modifiers ) override
	{
		if ( button == 0 && modifiers == MOD_SHIFT )
		{
			PickRay pickRay = m_camera->BuildPickRay( p.x, p.y );
			b3Vec3 direction = b3Normalize( pickRay.translation );
			b3Pos from = pickRay.origin + 2.0f * direction;
			ThrowBall( from, from + direction );
			return;
		}

		if ( button == 0 && modifiers == ( MOD_CTRL | MOD_SHIFT ) )
		{
			PickRay pickRay = m_camera->BuildPickRay( p.x, p.y );
			b3QueryFilter filter = b3DefaultQueryFilter();
			b3RayResult result = b3World_CastRayClosest( m_worldId, pickRay.origin, pickRay.translation, filter );
			if ( result.hit )
			{
				// A little in front of the surface, so the push goes into it
				b3Vec3 direction = b3Normalize( pickRay.translation );
				b3Pos center = result.point - 0.3f * direction;
				ApplyBlast( m_worldId, center, m_explosionRadius, m_explosionDamage, m_explosionImpulse );
			}

			return;
		}

		Sample::MouseDown( p, button, modifiers );
	}

	// Is the simulation going to advance in this frame?
	bool IsStepping() const
	{
		return m_context->pause == false || m_context->singleStep > 0;
	}

	static constexpr int m_maxDestructibles = 16;
	b3DestructibleId m_destructibles[m_maxDestructibles];
	int m_destructibleCount;

	const char* m_layerNames[B3_MAX_STRENGTH_LAYERS];
	float m_layerStrength[B3_MAX_STRENGTH_LAYERS];
	int m_layerCount;
	float m_strengthScale;

	float m_ballSpeed;
	float m_ballRadius;
	float m_explosionRadius;
	float m_explosionDamage;
	float m_explosionImpulse;
};

static const char* const s_buildingLayerNames[] = { "Between stories", "Between members", "Inside members", "Inside fragments" };
static const char* const s_panelLayerNames[] = { "Between panels", "Inside panels", "Inside fragments" };
static const char* const s_plainLayerNames[] = { "Bonds", "Inside fragments" };

// The primary scene. Buildings stand until charges take out the columns of the ground floor and of
// every third floor above it, then they come down in sections and end as rubble.
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

		ClearDestructibles();
		for ( int i = 0; i < m_buildingCount; ++i )
		{
			AddDestructible( GetBuildingCollapseBuilding( i ) );
		}

		SetLayerNames( s_buildingLayerNames, 4 );
		SyncControls();
	}

	void DestroyScene()
	{
		// The structure must outlive the destructibles, which this takes care of
		if ( m_isCreated )
		{
			ClearDestructibles();
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

		if ( m_detonateStep >= 0 && IsStepping() )
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
		ClearDestructibles();

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
	b3BodyId ResetWorld()
	{
		DestroyDestructible();

		b3Capacity capacity = {};
		CreateWorld( &capacity );
		return AddGroundBox( 100.0f );
	}

	// Make the destructible from the structure and hand it to the controls
	void CreateDestructible( const b3DestructibleDef& def, const char* const* layerNames, int layerCount )
	{
		m_destructibleId = b3CreateDestructible( m_worldId, &def );
		AddDestructible( m_destructibleId );
		SetLayerNames( layerNames, layerCount );
		SyncControls();
	}

	void Step() override
	{
		Sample::Step();
		DrawStats();
	}

	b3StructureData* m_structure;
	b3DestructibleId m_destructibleId;
};

// A free standing wall cut into panels. Blast a hole in it or throw a ball at it. With strong panels
// it comes apart panel by panel, with equal layers it crumbles around the hit.
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
		m_blastDamage = 2.0e5f;
		m_panelSize = 2.0f;
		m_debrisLifetime = 0.0f;
		m_freezeDebris = false;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		DestructionBuilder* builder = CreateDestructionBuilder( 21 );
		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.5f;
		fracture.childSize = 0.25f;
		fracture.tilt = 0.2f;

		b3Transform transform = { { 0.0f, 2.5f, 0.0f }, b3Quat_identity };
		AddFracturedBox( builder, transform, { 6.0f, 2.5f, 0.15f }, &fracture );

		// One cluster layer: panels along the wall
		ClusterFragmentsByGrid( builder, 0, 0, { m_panelSize, 0.0f, 0.0f } );
		float layerStrength[3] = { 1.0f, 5.0f, 3.0f };
		SetDestructionLayers( builder, 1, layerStrength );

		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.debrisLifetime = m_debrisLifetime;
		def.freezeSleepingDebris = m_freezeDebris;
		def.name = "wall";
		CreateDestructible( def, s_panelLayerNames, 3 );
	}

	bool DrawControls() override
	{
		bool rebuild = false;
		rebuild |= ImGui::SliderFloat( "Panel size", &m_panelSize, 1.0f, 6.0f, "%.1f m" );
		rebuild |= ImGui::SliderFloat( "Debris lifetime", &m_debrisLifetime, 0.0f, 10.0f, "%.1f s" );
		rebuild |= ImGui::Checkbox( "Freeze sleeping debris", &m_freezeDebris );
		if ( rebuild )
		{
			CreateScene();
		}

		ImGui::SliderFloat( "Blast radius", &m_blastRadius, 0.5f, 4.0f, "%.1f m" );
		ImGui::SliderFloat( "Blast impulse", &m_blastImpulse, 0.0f, 30000.0f, "%.0f" );
		ImGui::SliderFloat( "Blast damage", &m_blastDamage, 0.0f, 2.0e6f, "%.0f J" );

		if ( ImGui::Button( "Blast" ) )
		{
			// In front of the wall, so the debris flies out the back
			ApplyBlast( m_worldId, { 0.0f, 2.5f, 0.4f }, m_blastRadius, m_blastDamage, m_blastImpulse );
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
	float m_blastDamage;
	float m_panelSize;
	float m_debrisLifetime;
	bool m_freezeDebris;
};

static int sampleDestructionWall = RegisterSample( "Destruction", "Wall", DestructionWall::Create );

// A beam fixed at one end, in two halves. Make it longer until its own weight snaps it at the root,
// where the bending is the largest. Or make the layer between the halves weak and it breaks there.
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

		int cellCount = b3MaxInt( 2, int( m_length / 0.5f ) );
		float cellLength = m_length / float( cellCount );

		DestructionBuilder* builder = CreateDestructionBuilder( 1 );
		for ( int i = 0; i < cellCount; ++i )
		{
			b3Vec3 center = { ( float( i ) + 0.5f ) * cellLength, 0.0f, 0.0f };
			b3BoxHull box = b3MakeOffsetBoxHull( 0.5f * cellLength, 0.25f, 0.25f, center );
			int index = AddFragment( builder, &box.base, B3_NULL_INDEX, 0, i == 0 );

			// One cluster layer: the half next to the wall and the half at the free end
			SetFragmentCluster( builder, index, 0, i < cellCount / 2 ? 0 : 1 );
		}

		float layerStrength[3] = { 1.0f, 1.0f, 1.0f };
		SetDestructionLayers( builder, 1, layerStrength );
		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, 6.0f, 0.0f };
		def.name = "beam";

		static const char* const names[] = { "Between halves", "Inside halves", "Inside fragments" };
		CreateDestructible( def, names, 3 );
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
		return new DestructionCantilever( context );
	}

	float m_length;
};

static int sampleDestructionCantilever = RegisterSample( "Destruction", "Cantilever", DestructionCantilever::Create );

// A tall tower in segments. Blow out one side of its base and it leans over, and the bending breaks
// it apart on the way down.
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
		m_segmentHeight = 4.0f;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		DestructionBuilder* builder = CreateDestructionBuilder( 9 );
		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 1.0f;
		fracture.childSize = 0.5f;

		b3Transform transform = { { 0.0f, 0.5f * m_height, 0.0f }, b3Quat_identity };
		AddFracturedBox( builder, transform, { 2.0f, 0.5f * m_height, 2.0f }, &fracture );

		// One cluster layer: segments up the tower
		ClusterFragmentsByGrid( builder, 0, 0, { 0.0f, m_segmentHeight, 0.0f } );
		float layerStrength[3] = { 1.0f, 4.0f, 3.0f };
		SetDestructionLayers( builder, 1, layerStrength );

		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "tower";

		static const char* const names[] = { "Between segments", "Inside segments", "Inside fragments" };
		CreateDestructible( def, names, 3 );
	}

	bool DrawControls() override
	{
		bool rebuild = false;
		rebuild |= ImGui::SliderFloat( "Height", &m_height, 8.0f, 48.0f, "%.0f m" );
		rebuild |= ImGui::SliderFloat( "Segment height", &m_segmentHeight, 1.0f, 12.0f, "%.0f m" );
		if ( rebuild )
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
		return new DestructionTower( context );
	}

	float m_height;
	float m_segmentHeight;
};

static int sampleDestructionTower = RegisterSample( "Destruction", "Tower", DestructionTower::Create );

// A loose boulder. A short drop leaves it whole, a long one has the energy to shatter it. The
// inside of the fragment is the only layer that matters here: it sets how much energy that takes.
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
		CreateDestructible( def, s_plainLayerNames, 2 );
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

// A plate of four panels falls flat onto a ridge, which bends it hard. With equal layers it breaks
// up small. With strong bonds inside the panels and tough fragments the panels come through whole.
// The layers are read when the plate is built, so set them and drop again.
class DestructionPlateDrop : public SingleDestructible
{
public:
	explicit DestructionPlateDrop( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, -15.0f, 22.0f, { 0.0f, 3.0f, 0.0f } );
		}

		m_dropHeight = 7.0f;
		m_startLayers[0] = 1.0f;
		m_startLayers[1] = 30.0f;
		m_startLayers[2] = 30.0f;
		CreateScene();
	}

	void CreateScene()
	{
		b3BodyId groundId = ResetWorld();

		// A ridge across the plate, off the border between the panels
		b3BoxHull ridge = b3MakeOffsetBoxHull( 0.2f, 0.5f, 3.0f, { 0.5f, 1.5f, 0.0f } );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateHullShape( groundId, &shapeDef, &ridge.base );

		DestructionBuilder* builder = CreateDestructionBuilder( 29 );
		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.5f;
		fracture.childSize = 0.25f;

		// Four panels of two by two meters, each a cluster
		for ( int i = 0; i < 4; ++i )
		{
			b3Transform transform = { { i % 2 == 0 ? -1.0f : 1.0f, 0.0f, i / 2 == 0 ? -1.0f : 1.0f }, b3Quat_identity };
			fracture.clusters[0] = i;
			AddFracturedBox( builder, transform, { 1.0f, 0.15f, 1.0f }, &fracture );
		}

		SetDestructionLayers( builder, 1, m_startLayers );
		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, 2.0f + m_dropHeight, 0.0f };
		def.anchorToStatic = false;
		def.name = "plate";
		CreateDestructible( def, s_panelLayerNames, 3 );
	}

	bool DrawControls() override
	{
		ImGui::SliderFloat( "Drop height", &m_dropHeight, 1.0f, 30.0f, "%.0f m" );

		if ( ImGui::Button( "Drop again" ) )
		{
			// Keep what the sliders say for the next plate
			for ( int i = 0; i < 3; ++i )
			{
				m_startLayers[i] = m_layerStrength[i];
			}

			CreateScene();
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Equal layers" ) )
		{
			m_startLayers[0] = m_startLayers[1] = m_startLayers[2] = 1.0f;
			CreateScene();
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Strong panels" ) )
		{
			m_startLayers[0] = 1.0f;
			m_startLayers[1] = m_startLayers[2] = 30.0f;
			CreateScene();
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionPlateDrop( context );
	}

	float m_dropHeight;
	float m_startLayers[3];
};

static int sampleDestructionPlateDrop = RegisterSample( "Destruction", "Plate Drop", DestructionPlateDrop::Create );

// A heavy ball against a wall. The rigid body solver stops the ball as if the wall were whole. The
// bonds that broke could not have resisted that much, so the surplus is handed back and the ball
// punches through. Set the momentum transfer to zero and it stops dead instead.
class DestructionPunch : public SingleDestructible
{
public:
	explicit DestructionPunch( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 60.0f, -10.0f, 22.0f, { 0.0f, 1.5f, 0.0f } );
		}

		m_ballSpeed = 40.0f;
		m_ballRadius = 0.4f;
		m_ballId = b3_nullBodyId;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();
		m_ballId = b3_nullBodyId;

		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.5f;
		fracture.childSize = 0.0f;
		m_structure = CreateWallStructure( 6.0f, 3.0f, 0.3f, &fracture, 11 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "wall";
		CreateDestructible( def, s_plainLayerNames, 2 );
	}

	void Step() override
	{
		SingleDestructible::Step();

		if ( b3Body_IsValid( m_ballId ) )
		{
			b3Vec3 v = b3Body_GetLinearVelocity( m_ballId );
			DrawTextLine( "ball speed %.1f m/s", b3Length( v ) );
		}
	}

	bool DrawControls() override
	{
		if ( ImGui::Button( "Fire" ) )
		{
			// Remember the latest ball to show its speed
			b3Vec3 direction = { 0.0f, 0.0f, -1.0f };
			b3BodyDef bodyDef = b3DefaultBodyDef();
			bodyDef.type = b3_dynamicBody;
			bodyDef.position = { 0.0f, 1.5f, 6.0f };
			bodyDef.linearVelocity = b3MulSV( m_ballSpeed, direction );
			bodyDef.isBullet = true;
			m_ballId = b3CreateBody( m_worldId, &bodyDef );

			b3ShapeDef shapeDef = b3DefaultShapeDef();
			shapeDef.density = 7800.0f;
			b3Sphere sphere = { b3Vec3_zero, m_ballRadius };
			b3CreateSphereShape( m_ballId, &shapeDef, &sphere );
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
		return new DestructionPunch( context );
	}

	b3BodyId m_ballId;
};

static int sampleDestructionPunch = RegisterSample( "Destruction", "Punch Through", DestructionPunch::Create );

// A tower on a table that shakes. The anchored fragments ride on a kinematic host, and the
// acceleration of the host loads the structure like extra weight from the side. Raise the
// amplitude or the frequency until the tower snaps.
class DestructionShakeTable : public SingleDestructible
{
public:
	explicit DestructionShakeTable( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, -10.0f, 35.0f, { 0.0f, 6.0f, 0.0f } );
		}

		m_amplitude = 0.3f;
		m_frequency = 1.0f;
		m_height = 10.0f;
		m_isShaking = false;
		m_time = 0.0f;
		m_hostId = b3_nullBodyId;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();
		m_time = 0.0f;

		// The table: a kinematic plate just above the ground
		b3BodyDef hostDef = b3DefaultBodyDef();
		hostDef.type = b3_kinematicBody;
		hostDef.position = { 0.0f, 0.5f, 0.0f };
		hostDef.name = "table";
		m_hostId = b3CreateBody( m_worldId, &hostDef );

		b3BoxHull plate = b3MakeBoxHull( 4.0f, 0.25f, 4.0f );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateHullShape( m_hostId, &shapeDef, &plate.base );

		// The tower stands on the table. Its foot is fixed, and the fixed fragments follow the host.
		DestructionBuilder* builder = CreateDestructionBuilder( 5 );
		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.75f;
		fracture.childSize = 0.0f;

		FractureDef foot = fracture;
		foot.isAnchored = true;
		b3Transform footTransform = { { 0.0f, 0.375f, 0.0f }, b3Quat_identity };
		AddFracturedBox( builder, footTransform, { 0.75f, 0.375f, 0.75f }, &foot );

		b3Transform towerTransform = { { 0.0f, 0.75f + 0.5f * m_height, 0.0f }, b3Quat_identity };
		AddFracturedBox( builder, towerTransform, { 0.75f, 0.5f * m_height, 0.75f }, &fracture );

		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, 0.75f, 0.0f };
		def.hostBodyId = m_hostId;
		def.anchorToStatic = false;
		def.name = "tower";
		CreateDestructible( def, s_plainLayerNames, 2 );
	}

	void Step() override
	{
		if ( IsStepping() && b3Body_IsValid( m_hostId ) )
		{
			// Drive the table with a velocity. Starting from rest at the middle of the swing would
			// be a jolt, so the swing starts at one end.
			b3Vec3 velocity = b3Vec3_zero;
			if ( m_isShaking )
			{
				float omega = 2.0f * B3_PI * m_frequency;
				velocity.x = m_amplitude * omega * sinf( omega * m_time );
				m_time += 1.0f / m_context->hertz;
			}

			b3Body_SetLinearVelocity( m_hostId, velocity );
		}

		SingleDestructible::Step();

		float omega = 2.0f * B3_PI * m_frequency;
		DrawTextLine( "peak table acceleration %.1f m/s^2", m_amplitude * omega * omega );
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Tower height", &m_height, 3.0f, 24.0f, "%.0f m" ) )
		{
			CreateScene();
		}

		ImGui::SliderFloat( "Amplitude", &m_amplitude, 0.0f, 2.0f, "%.2f m" );
		ImGui::SliderFloat( "Frequency", &m_frequency, 0.1f, 5.0f, "%.1f Hz" );
		ImGui::Checkbox( "Shake", &m_isShaking );

		ImGui::SameLine();
		if ( ImGui::Button( "Reset" ) )
		{
			m_isShaking = false;
			CreateScene();
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionShakeTable( context );
	}

	b3BodyId m_hostId;
	float m_amplitude;
	float m_frequency;
	float m_height;
	float m_time;
	bool m_isShaking;
};

static int sampleDestructionShakeTable = RegisterSample( "Destruction", "Shake Table", DestructionShakeTable::Create );

// Two walls stand close together without touching. Damage that spreads along the bonds stays in
// the wall it hit. Damage that goes straight through space reaches the other wall too.
class DestructionRadialDamage : public SingleDestructible
{
public:
	explicit DestructionRadialDamage( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 60.0f, -20.0f, 16.0f, { 0.0f, 1.5f, 0.5f } );
		}

		m_innerRadius = 0.5f;
		m_outerRadius = 2.5f;
		m_damage = 2.0e5f;
		m_spreadAlongBonds = true;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		DestructionBuilder* builder = CreateDestructionBuilder( 13 );
		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 0.5f;
		fracture.childSize = 0.25f;

		b3Transform nearWall = { { 0.0f, 1.5f, 0.0f }, b3Quat_identity };
		b3Transform farWall = { { 0.0f, 1.5f, 1.0f }, b3Quat_identity };
		AddFracturedBox( builder, nearWall, { 2.0f, 1.5f, 0.15f }, &fracture );
		AddFracturedBox( builder, farWall, { 2.0f, 1.5f, 0.15f }, &fracture );

		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "walls";
		CreateDestructible( def, s_plainLayerNames, 2 );
	}

	bool DrawControls() override
	{
		ImGui::SliderFloat( "Inner radius", &m_innerRadius, 0.1f, 2.0f, "%.1f m" );
		ImGui::SliderFloat( "Outer radius", &m_outerRadius, 0.5f, 5.0f, "%.1f m" );
		ImGui::SliderFloat( "Damage", &m_damage, 0.0f, 2.0e6f, "%.0f J" );
		ImGui::Checkbox( "Spread along bonds", &m_spreadAlongBonds );

		if ( ImGui::Button( "Damage near wall" ) )
		{
			b3RadialDamageDef damageDef = b3DefaultRadialDamageDef();
			damageDef.position = { 0.0f, 1.5f, 0.0f };
			damageDef.innerRadius = m_innerRadius;
			damageDef.outerRadius = b3MaxFloat( m_outerRadius, m_innerRadius );
			damageDef.damage = m_damage;
			damageDef.spreadAlongBonds = m_spreadAlongBonds;
			b3World_ApplyRadialDamage( m_worldId, &damageDef );
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
		return new DestructionRadialDamage( context );
	}

	float m_innerRadius;
	float m_outerRadius;
	float m_damage;
	bool m_spreadAlongBonds;
};

static int sampleDestructionRadialDamage = RegisterSample( "Destruction", "Radial Damage", DestructionRadialDamage::Create );

// A slab on four columns. Take columns away one at a time. The slab hangs on to what is left, its
// bonds overload, and after a creak it comes down.
class DestructionFrame : public SingleDestructible
{
public:
	explicit DestructionFrame( SampleContext* context )
		: SingleDestructible( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 35.0f, -15.0f, 18.0f, { 0.0f, 1.5f, 0.0f } );
			GetGuiDraw()->drawBonds = true;
			GetGuiDraw()->drawBondStress = true;
		}

		m_buildingDef = DefaultBuildingDef();
		m_buildingDef.floorCount = 1;
		m_buildingDef.bayCountX = 1;
		m_buildingDef.bayCountZ = 1;
		m_buildingDef.enableWalls = false;
		CreateScene();
	}

	void CreateScene()
	{
		ResetWorld();

		m_structure = CreateBuildingStructure( &m_buildingDef );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "frame";
		CreateDestructible( def, s_buildingLayerNames, 4 );
	}

	// Crush the fragments of one column. The columns stand at the corners of the bay.
	void RemoveColumn( int column )
	{
		float half = 0.5f * m_buildingDef.baySize;
		b3Vec3 foot = { column % 2 == 0 ? -half : half, 0.0f, column / 2 == 0 ? -half : half };
		float columnTop = m_buildingDef.floorHeight - m_buildingDef.slabThickness + 0.01f;

		const b3StructureFragment* fragments = b3GetStructureFragments( m_structure );
		for ( int i = 0; i < m_structure->fragmentCount; ++i )
		{
			const b3StructureFragment* fragment = fragments + i;
			if ( fragment->parent != B3_NULL_INDEX || fragment->aabb.upperBound.y > columnTop )
			{
				continue;
			}

			float dx = fragment->center.x - foot.x;
			float dz = fragment->center.z - foot.z;
			if ( dx * dx + dz * dz < 1.0f )
			{
				b3Destructible_DamageFragment( m_destructibleId, i, 10.0f * fragment->health );
			}
		}
	}

	bool DrawControls() override
	{
		for ( int i = 0; i < 4; ++i )
		{
			ImGui::PushID( 100 + i );
			if ( ImGui::Button( "Remove column" ) )
			{
				RemoveColumn( i );
			}

			ImGui::PopID();
			if ( i % 2 == 0 )
			{
				ImGui::SameLine();
			}
		}

		if ( ImGui::Button( "Reset" ) )
		{
			CreateScene();
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionFrame( context );
	}

	BuildingDef m_buildingDef;
};

static int sampleDestructionFrame = RegisterSample( "Destruction", "Frame", DestructionFrame::Create );
