import taichi as ti

import ray_tracing_utils as rtu


ti.init(debug=True)

size = 512

camera = rtu.Camera(
    ti.Vector([0.0, 1.0, -5.0]),
    ti.Vector([0.0, 0.0, 1.0])
)
scene = rtu.Scene()

WALL_MAT = (rtu.MAT_DIFFUSE, ti.Vector([0.8, 0.8, 0.8]))
scene.add(rtu.Sphere(ti.Vector([0.0, -100.5, -1]), 100, WALL_MAT))
scene.add(rtu.Sphere(ti.Vector([0.0, 102.5, -1]), 100, WALL_MAT))
scene.add(rtu.Sphere(ti.Vector([0.0, 1.0, 101.0]), 100, WALL_MAT))
scene.add(rtu.Sphere(
    ti.Vector([-101.5, 0.0, -1.0]),
    100.0, (rtu.MAT_DIFFUSE, ti.Vector([0.6, 0.0, 0.0])
)))
scene.add(rtu.Sphere(
    ti.Vector([101.5, 0.0, -1.0]),
    100.0, (rtu.MAT_DIFFUSE, ti.Vector([0.0, 0.0, 0.6])
)))

scene.add(rtu.Sphere(
    ti.Vector([0.7, 0.0, -0.5]),
    0.5, (rtu.MAT_DIELECTRIC, ti.Vector([1.0, 1.0, 1.0]))
))
scene.add(rtu.Sphere(
    ti.Vector([-0.8, 0.2, -1.0]),
    0.7, (rtu.MAT_METAL, ti.Vector([0.6, 0.8, 0.8]))
))

pixels = ti.Vector.field(3, ti.f32, shape=(size, size))


@ti.kernel
def render():
    for i, j in pixels:
        u = i / size
        v = j / size
        color = rtu.zero()

        ray = camera.get_camera_ray(u, v)
        hit, hit_pos, hit_normal, mat, hit_color = scene.get_ray_hit(ray)

        if hit:
            color = hit_color

        pixels[i, j] = color


gui = ti.GUI('Path Tracing Demo', res=(size, size))

while gui.running:
    render()

    gui.set_image(pixels)
    gui.show()
