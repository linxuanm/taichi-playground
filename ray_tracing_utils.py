import taichi as ti


UP = ti.Vector([0, 1, 0])


@ti.data_oriented
class Ray:

    def __init__(self, origin, dir, t_min=1e-3, t_max=1e+8):
        self.origin = origin
        self.direction = direction
        self.t_min = t_min
        self.t_max = t_max

    @ti.func
    def at(self, t):
        return self.origin + t * self.direction


@ti.data_oriented
class SceneObject:

    def __init__(self, pos, mat):
        self.pos = pos
        self.mat = mat

    @ti.func
    def hit_ray(self, ray):
        raise NotImplemented


@ti.data_oriented
class Sphere(SceneObject):

    def __init__(self, pos, radius, mat):
        super(pos, mat)
        self.radius = radius

    @ti.func
    def hit_ray(self, ray):
        to_origin = ray.origin - self.pos

        a = 1 #ray.direction.dot(ray.direction)
        b = 2 * ray.direction.dot(to_origin)
        c = to_origin.dot(to_origin) - self.radius ** 2
        discrim = b * b - 4 * a * c
        root = 0

        hit = False
        hit_pos = ti.Vector([0, 0, 0])
        hit_normal = ti.Vector([0, 0, 0])

        if discrim > 0: # ignore = 0 cuz artifacts
            offset = ti.sqrt(discrim)
            root = (-b - offset) / (2 * a) # starts off with closer solution

            if root < ray.t_min or root > ray.t_max:
                root = (-b + offset) / (2 * a)
                if root >= ray.t_min and root <= ray.t_max:
                    hit = True

            else:
                hit = True

        if hit:
            hit_pos = ray.at(root)
            hit_normal = (hit_pos - self.center).normalize()

            if hit_normal.dot(ray.direction) > 0: # view from inside of sphere
                hit_normal = -hit_normal

        return hit, hit_pos, hit_normal


@ti.data_oriented
class Scene:

    def __init__(self):
        self.objs = []

    def add(self, obj):
        self.objs.append(obj)

    @ti.func
    def get_ray_hit(self, ray):
        closest = ray.t_max
        hit = False
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])

        for i in self.objs:
            curr_hit, curr_hit_pos, curr_hit_normal = i.hit_ray(ray):

            if curr_hit:
                hit = curr_hit
                hit_pos = curr_hit_pos
                hit_normal = curr_hit_normal

        return hit, hit_pos, hit_normal


@ti.data_oriented
class Camera:

    def __init__(pos, view_dir, fov=60.0, aspect_ratio=1.0):
        self.fov = fov
        self.ratio = aspect_ratio

        self.pos = ti.Vector.field(3, ti.f32, shape=())
        self.direction = ti.Vector.field(3, ti.f32, shape=())
        self.lower_left = ti.Vector.field(3, ti.f32, shape=())

        self.u_dir = ti.Vector.field(3, ti.f32, shape=())
        self.v_dir = ti.Vector.field(3, ti.f32, shape=())

        self.initialize(pos, view_dir)

    @ti.kernel
    def initialize(self, pos, view_dir):
        self.pos[None] = pos
        self.direction[None] = view_dir

        angle = self.fov / 180 * 3.14159265
        half_height = ti.tan(angle / 2)
        half_width = self.ratio * half_height

        right = self.direction[None].cross(UP).normalize()
        up = right.cross(self.direction[None])

        offset = -half_height * up - half_width * right
        self.lower_left[None] = self.pos[None] + offset

        self.u_dir[None] = 2 * half_width * right
        self.v_dir[None] = 2 * half_width * up

    @ti.func
    def get_camera_ray(self, u, v):
        offset = u * self.u_dir[None] + v * self.v_dir[None]
        direction = self.lower_left[None] + offset
        return Ray(self.pos, direction)
