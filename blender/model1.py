import bpy
import bmesh
import math

# Clean up old objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Parameters
num_points = 64
height = 2.0
#radius_func = lambda z: 0.2 + 0.6 * math.sin(math.pi * z / height) ** 1.5  # smooth bulb shape
#radius_func = lambda z: 2 * math.sqrt(1. - (z - 32)**2/32**2)
radius_func = lambda z: 2 * math.sqrt(max(0., 1. - (z / (height / 2))**2)) 

# Create profile curve
mesh = bpy.data.meshes.new("PodProfile")
obj = bpy.data.objects.new("PodProfile", mesh)
bpy.context.collection.objects.link(obj)

bm = bmesh.new()
for i in range(num_points):
    z = -height/2 + height * i / (num_points - 1)
    r = radius_func(z)
    print(f"z = {z}, r = {r}")
    bm.verts.new((0, r, z))
bm.verts.ensure_lookup_table()

for i in range(num_points - 1):
    bm.edges.new((bm.verts[i], bm.verts[i + 1]))

bm.to_mesh(mesh)
bm.free()

# Convert to curve object for revolution
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='OBJECT')  # Ensure we're in Object Mode
bpy.ops.object.convert(target='CURVE')

# Set curve settings
curve = obj.data
curve.dimensions = '3D'
curve.fill_mode = 'FULL'
curve.resolution_u = 16

# Set origin to (0,0,0) to ensure screw axis is centered
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Add screw modifier for azimuthal symmetry
bpy.ops.object.convert(target='MESH')  # Convert back to mesh to apply screw
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

bpy.ops.object.modifier_add(type='SCREW')
obj.modifiers["Screw"].angle = math.radians(360)
obj.modifiers["Screw"].steps = 64
obj.modifiers["Screw"].render_steps = 128
obj.modifiers["Screw"].axis = 'Z'

# Apply the screw modifier
bpy.ops.object.convert(target='MESH')

# Smooth shading
bpy.ops.object.shade_smooth()

# Add material
mat = bpy.data.materials.new(name="PodMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Metallic"].default_value = 0.3
bsdf.inputs["Roughness"].default_value = 0.4
obj.data.materials.append(mat)

# Lighting
light_data = bpy.data.lights.new(name="Light", type='AREA')
light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
light_obj.location = (5, -5, 5)
light_data.energy = 1000
bpy.context.collection.objects.link(light_obj)

# Camera
cam_data = bpy.data.cameras.new(name='Camera')
cam = bpy.data.objects.new('Camera', cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (4, -4, 2)
cam.rotation_euler = (math.radians(65), 0, math.radians(45))
bpy.context.scene.camera = cam

# Background
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.02, 0.02, 0.03, 1)

# Render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.filepath = "//flight_pod_render.png"

# Final render
bpy.ops.render.render(write_still=True)
