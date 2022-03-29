import taichi as ti
import numpy as np

import ray_tracing_utils as rtu


ti.init(debug=True, arch=ti.gpu)

size = 800
p_RR = 0.8
max_bounce = 10
samples_per_pixel = 20
refraction = 1.5

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
    ti.Vector([0.0, 5.4, -1.0]),
    3.0, (rtu.MAT_LIGHT, ti.Vector([10.0, 10.0, 10.0]))
))

scene.add(rtu.Sphere(
    ti.Vector([-101.5, 0.0, -1.0]),
    100.0, (rtu.MAT_DIFFUSE, ti.Vector([0.0, 0.6, 0.0])
)))
scene.add(rtu.Sphere(
    ti.Vector([101.5, 0.0, -1.0]),
    100.0, (rtu.MAT_DIFFUSE, ti.Vector([0.6, 0.0, 0.0])
)))

scene.add(rtu.Sphere(
    ti.Vector([-0.6, 0.0, -1.0]),
    0.5, (rtu.MAT_DIELECTRIC, ti.Vector([1.0, 1.0, 1.0]))
))

scene.add(rtu.Sphere(
    ti.Vector([0.7, 0.2, -0.5]),
    0.7, (rtu.MAT_METAL, ti.Vector([0.6, 0.8, 0.8]))
))

scene.add(rtu.Sphere(
    ti.Vector([0.25, -0.25, -1.5]),
    0.25, (rtu.MAT_DIFFUSE, ti.Vector([0.0, 0.0, 1.0]))
))

pixels = ti.Vector.field(3, ti.f32, shape=(size, size))


@ti.kernel
def render():
    for i, j in pixels:
        u = (i + ti.random()) / size
        v = (j + ti.random()) / size
        color = rtu.zero()

        for _ in ti.ndrange(samples_per_pixel):
            ray = camera.get_camera_ray(u, v)
            hit_color = ray_trace(ray)
            color += hit_color

        pixels[i, j] += color / samples_per_pixel


@ti.func
def ray_trace(ray):
    color = rtu.zero()
    bright = ti.Vector([1.0, 1.0, 1.0])
    origin = ray.origin
    direction = ray.direction

    for _ in ti.ndrange(max_bounce):
        if ti.random() > p_RR:
            break

        hit, hit_pos, hit_normal, inverted, mat, c = scene.get_ray_hit(
            rtu.Ray(origin, direction)
        )

        if mat == rtu.MAT_LIGHT:
            color = bright * c
            break

        elif mat == rtu.MAT_DIFFUSE:
            origin = hit_pos
            target = hit_pos + hit_normal + rtu.rand_diffuse_offset()
            direction = (target - origin).normalized()
            bright *= c

        elif mat == rtu.MAT_METAL:
            origin = hit_pos
            direction = rtu.reflect_across(direction, hit_normal)
            bright *= c

        elif mat == rtu.MAT_DIELECTRIC:
            refr = refraction
            origin = hit_pos

            if not inverted: # not inside to outside ray
                refr = 1.0 / refr

            sin = direction.cross(hit_normal).norm()
            cos = hit_normal.dot(-direction)
            if sin * refr > 1.0 or ti.random() < rtu.reflectance(cos, refr):
                direction = rtu.reflect_across(direction, hit_normal)
            else:
                direction = rtu.refract_across(direction, hit_normal, refr)



        bright /= p_RR

    return color


gui = ti.GUI('Path Tracing Demo', res=(size, size))

frame = 0.0
while gui.running:
    render()

    frame += 1.0
    gui.set_image(np.sqrt(pixels.to_numpy() / frame)) # gamma correction

    gui.show()
