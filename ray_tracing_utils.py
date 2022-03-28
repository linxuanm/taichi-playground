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
