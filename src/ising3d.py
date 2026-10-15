import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class Ising3d:
  def __init__(self, window_size: tuple, dims: tuple, T: float):
    self.window_size = window_size
    N, M, K = dims

    self.acc = ti.field(ti.f32, shape=7)
    self.spins = ti.field(ti.i8, shape=dims)
    self.positions = ti.Vector.field(3, ti.f32, shape=N * M * K)
    self.colors = ti.Vector.field(4, ti.f32, shape=N * M * K)

    self.rank_x = ti.field(ti.i32, shape=N)
    self.rank_y = ti.field(ti.i32, shape=M)
    self.rank_z = ti.field(ti.i32, shape=K)

    self.set_temperature(1 / T)

  def run(self):
    N, M, K = self.spins.shape
    window = ti.ui.Window("Ising model", res=self.window_size)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 3, 1.5)
    camera.lookat(0.5, 0.5, 0.0)
    camera.up(0, 1, 0)

    self._init_spins()

    while window.running:
      camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
      scene.set_camera(camera)
      scene.ambient_light((0.8, 0.8, 0.8))
      scene.point_light(pos=(0.5, 10, 1.0), color=(1, 1, 1))

      self._simulate_half(0)
      self._simulate_half(1)

      self._update_ranks(camera.curr_position)
      self._collect()

      scene.particles(self.positions, radius=0.5 / N, per_vertex_color=self.colors, index_count=N * M * K)
      canvas.scene(scene)
      window.show()

  def _update_ranks(self, c):
    N = self.spins.shape[0]
    for f, ca in ((self.rank_x, c[0]), (self.rank_y, c[1]), (self.rank_z, c[2])):
      n = f.shape[0]
      d = np.abs(np.arange(n) / N - ca)
      order = np.argsort(-d, kind="stable")
      r = np.empty(n, np.int32)
      r[order] = np.arange(n, dtype=np.int32)
      f.from_numpy(r)

  @ti.kernel
  def set_temperature(self, beta: float):
    for i in self.acc:
      self.acc[i] = ti.exp(-4.0 * beta * (i - 3))

  @ti.func
  def _dE(self, x: int, y: int, z: int) -> int:
    N, M, K = self.spins.shape
    nn = 0
    for dx, dy, dz in ti.static(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
      nn += self.spins[(x + dx) % N, (y + dy) % M, (z + dz) % K]
    return 2 * self.spins[x, y, z] * nn

  @ti.func
  def _flip(self, x: int, y: int, z: int) -> bool:
    return ti.random() < self.acc[self._dE(x, y, z) // 4 + 3]

  @ti.kernel
  def _init_spins(self):
    for x, y, z in self.spins:
      self.spins[x, y, z] = 1 if ti.random(float) <= 0.5 else -1

  @ti.kernel
  def _simulate_half(self, parity: int):
    N, M, K = self.spins.shape
    for x, y, h in ti.ndrange(N, M, K // 2):
      z = 2 * h + (x + y + parity) % 2
      if self._flip(x, y, z):
        self.spins[x, y, z] = -self.spins[x, y, z]

  @ti.kernel
  def _collect(self):
    N, M, K = self.spins.shape
    for x, y, z in self.spins:
      i = (self.rank_x[x] * M + self.rank_y[y]) * K + self.rank_z[z]
      self.positions[i] = tm.vec3(x, y, z) / N
      if self.spins[x, y, z] == 1:
        self.colors[i] = tm.vec4(0.7, 0.6, 0.4, 0.05)
      else:
        self.colors[i] = tm.vec4(0.3, 0.6, 0.7, 0.0)