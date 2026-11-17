// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "destruction_scenes.h"
#include "gfx/debug_adapter.h"
#include "gfx/keycodes.h"
#include "imgui.h"
#include "sample.h"

#include "box3d/box3d.h"

#include <math.h>

// The parameters every destruction sample has. A scene adds its own by deriving from this.
struct DestructionParams
{
	DestructionParams()
	{
		settings = b3DefaultDestructionSettings();
		for ( int i = 0; i < B3_MAX_STRENGTH_LAYERS; ++i )
		{
			layerStrength[i] = 1.0f;
		}
	}

	b3DestructionSettings settings;

	// Strength of the connections on each layer. Read from the structure of the scene the first
	// time it is built. After that the sliders own it and every rebuild of the scene applies it.
	float layerStrength[B3_MAX_STRENGTH_LAYERS];
	bool hasLayerStrength = false;
	float strengthScale = 1.0f;

	// Stress mode, or damage only. @see b3DestructibleDef::enableStress
	bool enableStress = true;

	float ballSpeed = 30.0f;
	float ballRadius = 0.5f;
	float explosionRadius = 2.5f;
	float explosionDamage = 1.0e6f;
	float explosionImpulse = 8000.0f;
	float dragForce = 100.0f;
};

// The parameters of a sample live here and not in the sample, so they survive a restart: the
// sample is deleted and created again on a restart, with the restart flag set. Picking the sample
// from the list starts over from the defaults. Every parameter type gets its own storage.
template <typename T>
static T* GetSampleParams( SampleContext* context )
{
	static T s_params;
	if ( context->restart == false )
	{
		s_params = T();
	}

	return &s_params;
}

// Shared parts of the destruction samples: the stats text and the controls every scene has.
// A scene registers its destructibles and names its strength layers, and gets sliders for them.
class DestructionSample : public Sample
{
public:
	DestructionSample( SampleContext* context, DestructionParams* params )
		: Sample( context )
		, m_params( params )
	{
		m_destructibleCount = 0;
		m_layerCount = 0;
		m_mouseForceScale = m_params->dragForce;

		for ( int i = 0; i < B3_MAX_STRENGTH_LAYERS; ++i )
		{
			m_layerNames[i] = "Layer";
		}
	}

	// Build the scene from the parameters: on creation, on Restart, and when a slider that shapes the
	// scene moves. A new world is made each time.
	virtual void CreateScene() = 0;

	// Set the parameters of the scene back to their defaults
	virtual void ResetParams() = 0;

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

	// Hand the parameters to a scene that was just built: the solver settings to the world, the rest
	// to every destructible. Call this after the destructibles were added.
	void ApplyParams()
	{
		b3World_SetDestructionSettings( m_worldId, &m_params->settings );
		m_mouseForceScale = m_params->dragForce;

		if ( m_destructibleCount == 0 )
		{
			return;
		}

		int structureLayers = b3Destructible_GetStructure( m_destructibles[0] )->strengthLayerCount;
		if ( m_layerCount > structureLayers )
		{
			m_layerCount = structureLayers;
		}

		if ( m_params->hasLayerStrength == false )
		{
			for ( int i = 0; i < m_layerCount; ++i )
			{
				m_params->layerStrength[i] = b3Destructible_GetLayerStrength( m_destructibles[0], i );
			}

			m_params->hasLayerStrength = true;
		}

		for ( int i = 0; i < m_destructibleCount; ++i )
		{
			b3DestructibleId destructibleId = m_destructibles[i];
			for ( int j = 0; j < m_layerCount; ++j )
			{
				b3Destructible_SetLayerStrength( destructibleId, j, m_params->layerStrength[j] );
			}

			b3Destructible_SetStrengthScale( destructibleId, m_params->strengthScale );
			b3Destructible_EnableStress( destructibleId, m_params->enableStress );
		}
	}

	// Counters and timings of the destruction solver
	void DrawStats()
	{
		b3DestructionCounters counters = b3World_GetDestructionCounters( m_worldId );
		b3DestructionProfile profile = b3World_GetDestructionProfile( m_worldId );

		DrawTextLine( "mode: %s", m_params->enableStress ? "stress" : "damage only" );
		DrawTextLine( "bodies %d, pieces %d, intact bonds %d", counters.bodyCount, counters.pieceCount, counters.bondCount );
		DrawTextLine( "solved pieces %d, iterations %d, broken bonds %d, new bodies %d, pending splits %d", counters.solvedPieceCount,
					  counters.iterationCount, counters.brokenBondCount, counters.newBodyCount, counters.pendingSplitCount );
		DrawTextLine( "destruction %.2f ms = loads %.2f + solve %.2f + fracture %.2f", profile.total, profile.loads, profile.solve,
					  profile.fracture );
	}

	// Stress lets the loads break the structure. Damage only breaks it where it is hit or damaged.
	void DrawModeControls()
	{
		const char* modes[] = { "Stress", "Damage only" };
		int mode = m_params->enableStress ? 0 : 1;
		if ( ImGui::Combo( "Mode", &mode, modes, 2 ) )
		{
			m_params->enableStress = mode == 0;
			for ( int i = 0; i < m_destructibleCount; ++i )
			{
				if ( b3Destructible_IsValid( m_destructibles[i] ) )
				{
					b3Destructible_EnableStress( m_destructibles[i], m_params->enableStress );
				}
			}
		}

		ImGui::SetItemTooltip( "Stress: the loads break the structure, it collapses when its supports go.\n"
							   "Damage only: it breaks only where it is hit or damaged. Loose pieces still fall." );
	}

	// How the structure comes apart. Weak outer layers and strong inner ones leave large pieces,
	// equal layers grind it down to single fragments and debris.
	void DrawLayerControls()
	{
		ImGui::SeparatorText( "Connection strength" );

		for ( int i = 0; i < m_layerCount; ++i )
		{
			ImGui::PushID( i );
			if ( ImGui::SliderFloat( m_layerNames[i], m_params->layerStrength + i, 0.05f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic ) )
			{
				for ( int j = 0; j < m_destructibleCount; ++j )
				{
					if ( b3Destructible_IsValid( m_destructibles[j] ) )
					{
						b3Destructible_SetLayerStrength( m_destructibles[j], i, m_params->layerStrength[i] );
					}
				}
			}
			ImGui::PopID();
		}

		if ( ImGui::SliderFloat( "Everything", &m_params->strengthScale, 0.05f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic ) )
		{
			for ( int j = 0; j < m_destructibleCount; ++j )
			{
				if ( b3Destructible_IsValid( m_destructibles[j] ) )
				{
					b3Destructible_SetStrengthScale( m_destructibles[j], m_params->strengthScale );
				}
			}
		}
	}

	// Restart, the mode, the strength, the solver, the debug draw and the ball. Every destruction
	// sample has these.
	void DrawCommonControls()
	{
		// Restart keeps every parameter. Reset parameters goes back to the defaults of this sample.
		if ( ImGui::Button( "Restart" ) )
		{
			CreateScene();
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Reset parameters" ) )
		{
			ResetParams();
			CreateScene();
		}

		DrawModeControls();
		DrawLayerControls();

		ImGui::SeparatorText( "Solver" );

		b3DestructionSettings* settings = &m_params->settings;
		bool changed = false;
		changed |= ImGui::SliderFloat( "Impact fracture", &settings->impactFractureScale, 0.0f, 8.0f, "%.1f" );
		changed |= ImGui::SliderFloat( "Impact crush", &settings->impactDamageScale, 0.0f, 8.0f, "%.1f" );
		changed |= ImGui::SliderFloat( "Momentum transfer", &settings->momentumTransferScale, 0.0f, 1.0f, "%.2f" );
		changed |= ImGui::SliderFloat( "Settle time", &settings->settleTime, 0.0f, 5.0f, "%.1f s" );
		ImGui::SetItemTooltip( "Loose pieces that rest this long stop failing under their loads. Zero lets a pile crumble on." );
		changed |= ImGui::SliderInt( "Work budget", &settings->workBudget, 10000, 1000000 );
		changed |= ImGui::SliderInt( "New bodies per step", &settings->maxNewBodiesPerStep, 8, 1024 );
		if ( changed )
		{
			b3World_SetDestructionSettings( m_worldId, settings );
		}

		ImGui::SeparatorText( "View" );

		// The stress view colors each bond by how close it is to breaking
		b3DebugDraw* draw = GetGuiDraw();
		ImGui::Checkbox( "Draw bonds", &draw->drawBonds );
		ImGui::Checkbox( "Draw bond stress", &draw->drawBondStress );

		ImGui::SeparatorText( "Ball: shift + click" );
		ImGui::SliderFloat( "Ball speed", &m_params->ballSpeed, 5.0f, 100.0f, "%.0f m/s" );
		ImGui::SliderFloat( "Ball radius", &m_params->ballRadius, 0.2f, 1.5f, "%.1f m" );

		ImGui::SeparatorText( "Explosion: ctrl + shift + click" );
		ImGui::SliderFloat( "Explosion radius", &m_params->explosionRadius, 0.5f, 8.0f, "%.1f m" );
		ImGui::SliderFloat( "Explosion damage", &m_params->explosionDamage, 0.0f, 1.0e7f, "%.0f J" );
		ImGui::SliderFloat( "Explosion impulse", &m_params->explosionImpulse, 0.0f, 30000.0f, "%.0f" );

		ImGui::SeparatorText( "Drag: ctrl + drag" );
		if ( ImGui::SliderFloat( "Drag force", &m_params->dragForce, 1.0f, 1000.0f, "%.0f x weight", ImGuiSliderFlags_Logarithmic ) )
		{
			m_mouseForceScale = m_params->dragForce;
		}
	}

	// Throw a steel ball from a point toward a target
	void ThrowBall( b3Pos from, b3Pos target )
	{
		b3Vec3 direction = b3Normalize( b3Vec3{ float( target.x - from.x ), float( target.y - from.y ), float( target.z - from.z ) } );

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = from;
		bodyDef.linearVelocity = b3MulSV( m_params->ballSpeed, direction );
		bodyDef.isBullet = true;
		b3BodyId ballId = b3CreateBody( m_worldId, &bodyDef );

		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.density = 7800.0f;
		b3Sphere sphere = { b3Vec3_zero, m_params->ballRadius };
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
				ApplyBlast( m_worldId, center, m_params->explosionRadius, m_params->explosionDamage, m_params->explosionImpulse );
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

	// Points to the storage of the derived sample, see GetSampleParams
	DestructionParams* m_params;

	static constexpr int m_maxDestructibles = 16;
	b3DestructibleId m_destructibles[m_maxDestructibles];
	int m_destructibleCount;

	const char* m_layerNames[B3_MAX_STRENGTH_LAYERS];
	int m_layerCount;
};

static const char* const s_buildingLayerNames[] = { "Between stories", "Between members", "Inside members", "Inside fragments" };
static const char* const s_panelLayerNames[] = { "Between panels", "Inside panels", "Inside fragments" };
static const char* const s_plainLayerNames[] = { "Bonds", "Inside fragments" };

// The primary scene. Buildings stand until charges take out the columns of the ground floor and of
// every third floor above it, then they come down in sections and end as rubble.
class DestructionBuilding : public DestructionSample
{
public:
	struct Params : DestructionParams
	{
		int buildingCount = 1;
		int floorCount = Sample::m_isDebug ? 2 : 5;
		bool autoDetonate = true;
	};

	explicit DestructionBuilding( SampleContext* context )
		: DestructionSample( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 35.0f, -15.0f, 70.0f, { 0.0f, 8.0f, 0.0f } );
		}

		m_shadowSplitFar = 150.0f;
		m_detonateStep = -1;
		m_isCreated = false;

		CreateScene();
	}

	~DestructionBuilding() override
	{
		DestroyScene();
	}

	void CreateScene() override
	{
		DestroyScene();

		SetBuildingCollapseScale( m_scene.buildingCount, m_scene.floorCount );

		b3Capacity capacity = {};
		GetBuildingCollapseCapacity( &capacity );
		CreateWorld( &capacity );

		CreateBuildingCollapse( m_worldId );
		m_isCreated = true;
		m_detonateStep = -1;

		ClearDestructibles();
		for ( int i = 0; i < m_scene.buildingCount; ++i )
		{
			AddDestructible( GetBuildingCollapseBuilding( i ) );
		}

		SetLayerNames( s_buildingLayerNames, 4 );
		ApplyParams();
	}

	void ResetParams() override
	{
		m_scene = Params();
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
		if ( m_scene.autoDetonate && m_detonateStep < 0 && m_stepCount >= 60 )
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
		rebuild |= ImGui::SliderInt( "Buildings", &m_scene.buildingCount, 1, 8 );
		rebuild |= ImGui::SliderInt( "Floors", &m_scene.floorCount, 1, 16 );
		if ( rebuild )
		{
			CreateScene();
		}

		ImGui::Checkbox( "Auto detonate", &m_scene.autoDetonate );
		if ( ImGui::Button( "Detonate" ) && m_detonateStep < 0 )
		{
			m_detonateStep = m_stepCount;
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Throw ball" ) )
		{
			float height = 0.5f * 3.2f * float( m_scene.floorCount );
			ThrowBall( { 0.0f, height, 40.0f }, { 0.0f, height, 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionBuilding( context );
	}

	Params& m_scene;
	int m_detonateStep;
	bool m_isCreated;
};

static int sampleDestructionBuilding = RegisterSample( "Destruction", "Building", DestructionBuilding::Create );

// A destruction sample that owns one structure and one destructible made from it.
class SingleDestructible : public DestructionSample
{
public:
	SingleDestructible( SampleContext* context, DestructionParams* params )
		: DestructionSample( context, params )
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

	// Make the destructible from the structure and hand it the parameters
	void CreateDestructible( const b3DestructibleDef& def, const char* const* layerNames, int layerCount )
	{
		m_destructibleId = b3CreateDestructible( m_worldId, &def );
		AddDestructible( m_destructibleId );
		SetLayerNames( layerNames, layerCount );
		ApplyParams();
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
	struct Params : DestructionParams
	{
		float blastRadius = 1.5f;
		float blastImpulse = 8000.0f;
		float blastDamage = 2.0e5f;
		float panelSize = 2.0f;
		float debrisLifetime = 0.0f;
		bool freezeDebris = false;
	};

	explicit DestructionWall( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 20.0f, -10.0f, 22.0f, { 0.0f, 2.5f, 0.0f } );
		}

		CreateScene();
	}

	void CreateScene() override
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
		ClusterFragmentsByGrid( builder, 0, 0, { m_scene.panelSize, 0.0f, 0.0f } );
		float layerStrength[3] = { 1.0f, 5.0f, 3.0f };
		SetDestructionLayers( builder, 1, layerStrength );

		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.debrisLifetime = m_scene.debrisLifetime;
		def.freezeSleepingDebris = m_scene.freezeDebris;
		def.name = "wall";
		CreateDestructible( def, s_panelLayerNames, 3 );
	}

	void ResetParams() override
	{
		m_scene = Params();
	}

	bool DrawControls() override
	{
		bool rebuild = false;
		rebuild |= ImGui::SliderFloat( "Panel size", &m_scene.panelSize, 1.0f, 6.0f, "%.1f m" );
		rebuild |= ImGui::SliderFloat( "Debris lifetime", &m_scene.debrisLifetime, 0.0f, 10.0f, "%.1f s" );
		rebuild |= ImGui::Checkbox( "Freeze sleeping debris", &m_scene.freezeDebris );
		if ( rebuild )
		{
			CreateScene();
		}

		ImGui::SliderFloat( "Blast radius", &m_scene.blastRadius, 0.5f, 4.0f, "%.1f m" );
		ImGui::SliderFloat( "Blast impulse", &m_scene.blastImpulse, 0.0f, 30000.0f, "%.0f" );
		ImGui::SliderFloat( "Blast damage", &m_scene.blastDamage, 0.0f, 2.0e6f, "%.0f J" );

		if ( ImGui::Button( "Blast" ) )
		{
			// In front of the wall, so the debris flies out the back
			ApplyBlast( m_worldId, { 0.0f, 2.5f, 0.4f }, m_scene.blastRadius, m_scene.blastDamage, m_scene.blastImpulse );
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Throw ball" ) )
		{
			ThrowBall( { 0.0f, 2.5f, 25.0f }, { 0.0f, 2.5f, 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionWall( context );
	}

	Params& m_scene;
};

static int sampleDestructionWall = RegisterSample( "Destruction", "Wall", DestructionWall::Create );

// A beam fixed at one end, in two halves. Make it longer until its own weight snaps it at the root,
// where the bending is the largest. Or make the layer between the halves weak and it breaks there.
class DestructionCantilever : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		float length = 6.0f;
	};

	explicit DestructionCantilever( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 20.0f, -10.0f, 25.0f, { 5.0f, 5.0f, 0.0f } );
			GetGuiDraw()->drawBonds = true;
			GetGuiDraw()->drawBondStress = true;
		}

		CreateScene();
	}

	void CreateScene() override
	{
		ResetWorld();

		float length = m_scene.length;
		int cellCount = b3MaxInt( 2, int( length / 0.5f ) );
		float cellLength = length / float( cellCount );

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

	void ResetParams() override
	{
		m_scene = Params();
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Length", &m_scene.length, 2.0f, 20.0f, "%.1f m" ) )
		{
			CreateScene();
		}

		if ( ImGui::Button( "Throw ball" ) )
		{
			float x = 0.75f * m_scene.length;
			ThrowBall( { x, 6.0f, 20.0f }, { x, 6.0f, 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionCantilever( context );
	}

	Params& m_scene;
};

static int sampleDestructionCantilever = RegisterSample( "Destruction", "Cantilever", DestructionCantilever::Create );

// A tall tower in segments. Blow out one side of its base and it leans over, and the bending breaks
// it apart on the way down.
class DestructionTower : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		float height = 24.0f;
		float segmentHeight = 4.0f;
	};

	explicit DestructionTower( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, -10.0f, 60.0f, { 0.0f, 12.0f, 0.0f } );
		}

		m_shadowSplitFar = 120.0f;
		CreateScene();
	}

	void CreateScene() override
	{
		ResetWorld();

		DestructionBuilder* builder = CreateDestructionBuilder( 9 );
		FractureDef fracture = DefaultFractureDef();
		fracture.cellSize = 1.0f;
		fracture.childSize = 0.5f;

		float height = m_scene.height;
		b3Transform transform = { { 0.0f, 0.5f * height, 0.0f }, b3Quat_identity };
		AddFracturedBox( builder, transform, { 2.0f, 0.5f * height, 2.0f }, &fracture );

		// One cluster layer: segments up the tower
		ClusterFragmentsByGrid( builder, 0, 0, { 0.0f, m_scene.segmentHeight, 0.0f } );
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

	void ResetParams() override
	{
		m_scene = Params();
	}

	bool DrawControls() override
	{
		bool rebuild = false;
		rebuild |= ImGui::SliderFloat( "Height", &m_scene.height, 8.0f, 48.0f, "%.0f m" );
		rebuild |= ImGui::SliderFloat( "Segment height", &m_scene.segmentHeight, 1.0f, 12.0f, "%.0f m" );
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
			float y = 0.6f * m_scene.height;
			ThrowBall( { 0.0f, y, 40.0f }, { 0.0f, y, 0.0f } );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionTower( context );
	}

	Params& m_scene;
};

static int sampleDestructionTower = RegisterSample( "Destruction", "Tower", DestructionTower::Create );

// A loose boulder. A short drop leaves it whole, a long one has the energy to shatter it. The
// inside of the fragment is the only layer that matters here: it sets how much energy that takes.
// Restart drops it again.
class DestructionBoulder : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		float dropHeight = 20.0f;
	};

	explicit DestructionBoulder( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 20.0f, -10.0f, 40.0f, { 0.0f, 8.0f, 0.0f } );
		}

		CreateScene();
	}

	void CreateScene() override
	{
		ResetWorld();

		b3StructureMaterial material = b3DefaultStructureMaterial();
		m_structure = CreateBoulderStructure( 1.5f, 0.6f, &material, 3 );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, m_scene.dropHeight + 1.5f, 0.0f };
		def.anchorToStatic = false;
		def.name = "boulder";
		CreateDestructible( def, s_plainLayerNames, 2 );
	}

	void ResetParams() override
	{
		m_scene = Params();
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Drop height", &m_scene.dropHeight, 1.0f, 40.0f, "%.0f m" ) )
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

	Params& m_scene;
};

static int sampleDestructionBoulder = RegisterSample( "Destruction", "Boulder", DestructionBoulder::Create );

// A plate of four panels falls flat onto a ridge, which bends it hard. With equal layers it breaks
// up small. With strong bonds inside the panels and tough fragments the panels come through whole.
// Restart drops it again.
class DestructionPlateDrop : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		float dropHeight = 7.0f;
	};

	explicit DestructionPlateDrop( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, -15.0f, 22.0f, { 0.0f, 3.0f, 0.0f } );
		}

		CreateScene();
	}

	void CreateScene() override
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

		// Strong panels to start with. The sliders and the buttons below change this.
		float layerStrength[3] = { 1.0f, 30.0f, 30.0f };
		SetDestructionLayers( builder, 1, layerStrength );
		m_structure = BuildStructure( builder );
		DestroyDestructionBuilder( builder );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.position = { 0.0f, 2.0f + m_scene.dropHeight, 0.0f };
		def.anchorToStatic = false;
		def.name = "plate";
		CreateDestructible( def, s_panelLayerNames, 3 );
	}

	void ResetParams() override
	{
		m_scene = Params();
	}

	// Drop a new plate with these connection strengths
	void DropWithLayers( float betweenPanels, float insidePanels, float insideFragments )
	{
		m_scene.layerStrength[0] = betweenPanels;
		m_scene.layerStrength[1] = insidePanels;
		m_scene.layerStrength[2] = insideFragments;
		m_scene.hasLayerStrength = true;
		CreateScene();
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Drop height", &m_scene.dropHeight, 1.0f, 30.0f, "%.0f m" ) )
		{
			CreateScene();
		}

		if ( ImGui::Button( "Equal layers" ) )
		{
			DropWithLayers( 1.0f, 1.0f, 1.0f );
		}

		ImGui::SameLine();
		if ( ImGui::Button( "Strong panels" ) )
		{
			DropWithLayers( 1.0f, 30.0f, 30.0f );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionPlateDrop( context );
	}

	Params& m_scene;
};

static int sampleDestructionPlateDrop = RegisterSample( "Destruction", "Plate Drop", DestructionPlateDrop::Create );

// A heavy ball against a wall. The rigid body solver stops the ball as if the wall were whole. The
// bonds that broke could not have resisted that much, so the surplus is handed back and the ball
// punches through. Set the momentum transfer to zero and it stops dead instead.
class DestructionPunch : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		Params()
		{
			ballSpeed = 40.0f;
			ballRadius = 0.4f;
		}
	};

	explicit DestructionPunch( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 60.0f, -10.0f, 22.0f, { 0.0f, 1.5f, 0.0f } );
		}

		m_ballId = b3_nullBodyId;
		CreateScene();
	}

	void CreateScene() override
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

	void ResetParams() override
	{
		m_scene = Params();
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
			bodyDef.linearVelocity = b3MulSV( m_scene.ballSpeed, direction );
			bodyDef.isBullet = true;
			m_ballId = b3CreateBody( m_worldId, &bodyDef );

			b3ShapeDef shapeDef = b3DefaultShapeDef();
			shapeDef.density = 7800.0f;
			b3Sphere sphere = { b3Vec3_zero, m_scene.ballRadius };
			b3CreateSphereShape( m_ballId, &shapeDef, &sphere );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionPunch( context );
	}

	Params& m_scene;
	b3BodyId m_ballId;
};

static int sampleDestructionPunch = RegisterSample( "Destruction", "Punch Through", DestructionPunch::Create );

// A tower on a table that shakes. The anchored fragments ride on a kinematic host, and the
// acceleration of the host loads the structure like extra weight from the side. Raise the
// amplitude or the frequency until the tower snaps.
class DestructionShakeTable : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		float amplitude = 0.3f;
		float frequency = 1.0f;
		float height = 10.0f;
		bool isShaking = false;
	};

	explicit DestructionShakeTable( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, -10.0f, 35.0f, { 0.0f, 6.0f, 0.0f } );
		}

		m_time = 0.0f;
		m_hostId = b3_nullBodyId;
		CreateScene();
	}

	void CreateScene() override
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

		float height = m_scene.height;
		b3Transform towerTransform = { { 0.0f, 0.75f + 0.5f * height, 0.0f }, b3Quat_identity };
		AddFracturedBox( builder, towerTransform, { 0.75f, 0.5f * height, 0.75f }, &fracture );

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

	void ResetParams() override
	{
		m_scene = Params();
	}

	void Step() override
	{
		if ( IsStepping() && b3Body_IsValid( m_hostId ) )
		{
			// Drive the table with a velocity. Starting from rest at the middle of the swing would
			// be a jolt, so the swing starts at one end.
			b3Vec3 velocity = b3Vec3_zero;
			if ( m_scene.isShaking )
			{
				float omega = 2.0f * B3_PI * m_scene.frequency;
				velocity.x = m_scene.amplitude * omega * sinf( omega * m_time );
				m_time += 1.0f / m_context->hertz;
			}

			b3Body_SetLinearVelocity( m_hostId, velocity );
		}

		SingleDestructible::Step();

		float omega = 2.0f * B3_PI * m_scene.frequency;
		DrawTextLine( "peak table acceleration %.1f m/s^2", m_scene.amplitude * omega * omega );
	}

	bool DrawControls() override
	{
		if ( ImGui::SliderFloat( "Tower height", &m_scene.height, 3.0f, 24.0f, "%.0f m" ) )
		{
			CreateScene();
		}

		ImGui::SliderFloat( "Amplitude", &m_scene.amplitude, 0.0f, 2.0f, "%.2f m" );
		ImGui::SliderFloat( "Frequency", &m_scene.frequency, 0.1f, 5.0f, "%.1f Hz" );
		ImGui::Checkbox( "Shake", &m_scene.isShaking );

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionShakeTable( context );
	}

	Params& m_scene;
	b3BodyId m_hostId;
	float m_time;
};

static int sampleDestructionShakeTable = RegisterSample( "Destruction", "Shake Table", DestructionShakeTable::Create );

// Two walls stand close together without touching. Damage that spreads along the bonds stays in
// the wall it hit. Damage that goes straight through space reaches the other wall too.
class DestructionRadialDamage : public SingleDestructible
{
public:
	struct Params : DestructionParams
	{
		float innerRadius = 0.5f;
		float outerRadius = 2.5f;
		float damage = 2.0e5f;
		bool spreadAlongBonds = true;
	};

	explicit DestructionRadialDamage( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
		, m_scene( *static_cast<Params*>( m_params ) )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 60.0f, -20.0f, 16.0f, { 0.0f, 1.5f, 0.5f } );
		}

		CreateScene();
	}

	void CreateScene() override
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

	void ResetParams() override
	{
		m_scene = Params();
	}

	bool DrawControls() override
	{
		ImGui::SliderFloat( "Inner radius", &m_scene.innerRadius, 0.1f, 2.0f, "%.1f m" );
		ImGui::SliderFloat( "Outer radius", &m_scene.outerRadius, 0.5f, 5.0f, "%.1f m" );
		ImGui::SliderFloat( "Damage", &m_scene.damage, 0.0f, 2.0e6f, "%.0f J" );
		ImGui::Checkbox( "Spread along bonds", &m_scene.spreadAlongBonds );

		if ( ImGui::Button( "Damage near wall" ) )
		{
			b3RadialDamageDef damageDef = b3DefaultRadialDamageDef();
			damageDef.position = { 0.0f, 1.5f, 0.0f };
			damageDef.innerRadius = m_scene.innerRadius;
			damageDef.outerRadius = b3MaxFloat( m_scene.outerRadius, m_scene.innerRadius );
			damageDef.damage = m_scene.damage;
			damageDef.spreadAlongBonds = m_scene.spreadAlongBonds;
			b3World_ApplyRadialDamage( m_worldId, &damageDef );
		}

		DrawCommonControls();
		return true;
	}

	static Sample* Create( SampleContext* context )
	{
		return new DestructionRadialDamage( context );
	}

	Params& m_scene;
};

static int sampleDestructionRadialDamage = RegisterSample( "Destruction", "Radial Damage", DestructionRadialDamage::Create );

// A slab on four columns. Take columns away one at a time. The slab hangs on to what is left, its
// bonds overload, and after a creak it comes down. In damage only mode it hangs on until the last
// column is gone, then it falls.
class DestructionFrame : public SingleDestructible
{
public:
	// No parameters of its own, but its own storage
	struct Params : DestructionParams
	{
	};

	explicit DestructionFrame( SampleContext* context )
		: SingleDestructible( context, GetSampleParams<Params>( context ) )
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

	void CreateScene() override
	{
		ResetWorld();

		m_structure = CreateBuildingStructure( &m_buildingDef );

		b3DestructibleDef def = b3DefaultDestructibleDef();
		def.structure = m_structure;
		def.name = "frame";
		CreateDestructible( def, s_buildingLayerNames, 4 );
	}

	void ResetParams() override
	{
		*static_cast<Params*>( m_params ) = Params();
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
