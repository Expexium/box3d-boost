# Destruction {#destruction}

Box3D can break structures apart. You give it a structure that is already cut into convex
fragments, and it decides when and where the structure fails: it works out the force every joint
between fragments has to carry, compares that with the strength of the material, breaks the joints
that are overloaded, and splits the rigid bodies to match.

This is the back end of a destruction system. It does not cut meshes and it does not render.
You fracture your assets in a tool of your choice, and you follow the events to move your render
meshes.

## The pieces

**Fragments** are the convex chunks a structure is cut into. Fragments come in levels. The
top level fragments carry the structure and are called support fragments. A support fragment can
have children, smaller chunks it crumbles into when it is crushed. Children are dormant until
then and cost nothing.

**Bonds** join two support fragments, or a fragment and the world. A bond has an area, a
centroid and a normal. Bonds to the world are anchors. You can list bonds yourself, or let the
cooking find them where fragment faces touch.

A **structure** (`b3StructureData`) is the cooked, immutable asset: fragments, hulls, bonds and
materials in one relocatable block of memory. Cook it once with `b3CreateStructure`. You can save
the block to disk and check it after loading with `b3ValidateStructure`.

A **destructible** (`b3DestructibleId`) is an instance of a structure in a world. Many
destructibles can share one structure. The structure must outlive them.

Fragments joined by intact bonds move as one rigid body with one shape per fragment. All anchored
fragments of a destructible share a single static body, or a kinematic body when the destructible
rides on a host such as a vehicle.

## Making a destructible

```c
// Describe the fragments. Each one is a convex hull in structure space.
b3FragmentDef fragmentDefs[2];
for ( int i = 0; i < 2; ++i )
{
    fragmentDefs[i] = b3DefaultFragmentDef();
    fragmentDefs[i].hulls = &hulls[i];
    fragmentDefs[i].hullCount = 1;
}

// The first fragment is fixed to the world
fragmentDefs[0].isAnchored = true;

b3StructureMaterial concrete = b3DefaultStructureMaterial();

b3StructureDef structureDef = b3DefaultStructureDef();
structureDef.fragments = fragmentDefs;
structureDef.fragmentCount = 2;
structureDef.materials = &concrete;
structureDef.materialCount = 1;

// Find bonds where fragment faces touch
structureDef.generateBonds = true;

b3StructureData* structure = b3CreateStructure( &structureDef );

b3DestructibleDef def = b3DefaultDestructibleDef();
def.structure = structure;
def.position = position;
b3DestructibleId destructibleId = b3CreateDestructible( worldId, &def );
```

With `anchorToStatic` set, which is the default, fragments that touch static shapes when the
destructible is created are anchored to them. A building placed on the ground is held by the
ground without you marking anything.

After that the destructible takes part in `b3World_Step` like everything else. Destroy the
destructibles before the structure, and the structure with `b3DestroyStructure`.

## What makes it break

**Stress.** Every step the solver finds the force and moment each bond carries under gravity,
contact forces, the motion of a host, and spin. It turns that into tension, compression and shear
stress on the bond area and compares them with the material limits. Bending counts, which is what
brings most structures down. Each limit has an elastic value and a fatal value:

- below the elastic limit nothing happens
- at the fatal limit the bond breaks
- in between the bond wears down and fails after about the material `creepTime`, so an
  overloaded structure creaks before it goes

Failure runs in rounds. Only the bonds near the worst overload break, then the load is solved
again. Removing a support therefore makes the load move to what remains, and the failure travels
through the structure the way it would in the real thing.

**Impacts.** A hard contact stresses the bonds near the hit. The stress picks which bonds fail,
and the kinetic energy lost in the hit decides how many: a bond costs its area times the material
`toughness`. A light tap cannot shatter a wall, and the result does not depend on the time step.
The same energy wears down the health of the fragment that was hit. A fragment with no health left
is crushed into its children.

**Damage you apply.** `b3World_ApplyRadialDamage` damages fragments and bonds in a sphere, with
the damage spreading along bonds and not through empty space. `b3World_Explode` pushes the
fragments. Use both for an explosion. `b3Destructible_DamageFragment`, `b3Destructible_BreakBond`
and `b3Destructible_ApplyImpulse` act on single fragments and bonds.

## Materials

`b3StructureMaterial` holds the density, the elastic and fatal limits for tension, compression
and shear, a relative stiffness, the toughness, the creep time, and a random variation of the
bond strength. The default is plain concrete. Start from real strengths and scale them for the
game. Bonds take the material of their fragments.

`b3Destructible_SetStrengthScale` scales the strength of a whole destructible at run time, and
the layers described below scale parts of it.

## Layers

How a structure comes apart is decided by how strong its connections are on each layer.

Put the support fragments into clusters, on up to three cluster layers from coarse to fine. For
a building the first layer could be the story and the second the member: a column, a wall panel,
a bay of the floor slab. Each fragment names its cluster on every layer in
`b3FragmentDef::clusters`.

Every bond then lies on one strength layer: the coarsest cluster layer on which its two fragments
are in different clusters. A bond between two stories is on the first layer, a bond between a
wall and a slab of the same story on the second, and a bond between two fragments of the same
wall on the layer after that. The last layer is the inside of a fragment, which is the health
that keeps it from being crushed into its children.

`b3StructureDef::layerStrength` gives each layer a scale on the material. It applies to the
stress limits of the bonds and to the energy it takes to break them, so it covers slow overload,
impacts and damage alike. `b3Destructible_SetLayerStrength` changes it for one destructible at
any time.

- With equal layers a structure breaks wherever the load is the highest. A large collapse then
  grinds it down to single fragments and debris.
- With strong inner layers it separates along the cluster borders and the clusters come through
  as large pieces: whole wall panels and slab bays.
- With a strong last layer the fragments are not crushed, so there is less debris.
- Strong outer layers as well give pieces that span several clusters.

For example, the same eight story building, demolished the same way, with the layers for
stories, members, inside members and inside fragments set to:

| Layers | Fragments crushed | Single fragments | In pieces of 9 or more |
|---|---|---|---|
| 1, 1, 1, 1 | 60% | 84% | 2% |
| 1, 1, 10, 5 | 29% | 33% | 21% |
| 1, 1, 20, 20 | 20% | 21% | 43% |
| 1, 20, 20, 20 | 21% | 20% | 47% |

The last two columns are shares of the fragments that were not crushed.

Bond events and `b3Destructible_GetBondState` report the layer of a bond. A break on a coarse
layer is a large part coming off, a break on the finest is a crack inside a member.

## Events

Read the events after each step with `b3World_GetDestructionEvents`. They are valid until the
next step.

- **Fragment events** tell you a fragment moved to another body, was released as debris, or is no
  longer simulated. This is what you follow to move render meshes.
- **Body events** tell you the solver created a body, and from which body it split.
- **Bond events** tell you a bond broke, where, and why: overload, creep, impact, damage or
  crush. Useful for sound and dust. Turn them on with `enableBondEvents`.

`b3Shape_GetDestructible` and `b3Shape_GetFragmentIndex` map a shape from a ray cast or a contact
event back to its fragment.

## Debris

Pieces smaller than `debrisRadius` are debris. They get the `debrisFilter`, they can be removed
after `debrisLifetime`, and they can turn into static rubble when they fall asleep
(`freezeSleepingDebris`). Fragments smaller than `minFragmentRadius` are never simulated.

## Cost and tuning

The settings are per world: `b3World_SetDestructionSettings`.

A piece gets a share of solver work per step, `workBudget`. A piece that needs more keeps solving
the same loads over the following steps and reacts a few steps late. A bond that is far over its
limit breaks before the solve is exact, a bond near its limit waits for it. Raise the budget to
make large structures react sooner.

`impactFractureScale` sets how much of the impact energy goes into breaking bonds. Raise it and
falling pieces break up more when they land. `impactDamageScale` does the same for crushing
fragments.

`maxNewBodiesPerStep` spreads large splits over several steps.

`b3World_GetDestructionProfile` and `b3World_GetDestructionCounters` report the time and the
amount of work of the latest step. Turn on `drawBonds` and `drawBondStress` in `b3DebugDraw` to
see the bonds colored by how close they are to breaking.

Pieces are solved in parallel. The result is deterministic and does not depend on the number of
workers.

## Authoring advice

- Cut the support level along the members of the structure: columns, beams, slab bays. The weak
  points of the bond graph then match those of the building.
- Cut finely where hits are likely and coarsely elsewhere. The cost follows the number of support
  fragments and bonds, not the number of children.
- Give reinforced members a stronger material, or a wide gap between the elastic and the fatal
  limit so they sag before they snap.
- Irregular cuts look better and pile up better than a regular grid of boxes.

## Recording

Everything the solver does to the world goes through the public API: it creates bodies, moves
shapes between them with `b3Shape_SetBody`, and so on. A recording therefore contains a collapse
as ordinary operations and replays without the destruction solver.
