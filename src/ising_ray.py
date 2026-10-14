import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class IsingRay:
  def __init__(self, window_size: tuple, dims: tuple, T: float):
    self.window_size = window_size
    N, M, K = dims

    self.acc = ti.field(ti.f32, shape=7)
    self.spins = ti.field(ti.i8, shape=dims)
    self.image = ti.Vector.field(3, ti.f32, shape=window_size)

    self.rank_x = ti.field(ti.i32, shape=N)
    self.rank_y = ti.field(ti.i32, shape=M)
    self.rank_z = ti.field(ti.i32, shape=K)
    
    self.set_temperature(1 / T)

  def run(self):
    N, M, K = self.spins.shape
    window = ti.ui.Window("Ising model", res=self.window_size)
    canvas = window.get_canvas()
    camera = ti.ui.Camera()
    camera.position(0.5, 3, 1.5)
    camera.lookat(0.5, 0.5, 0.0)
    camera.up(0, 1, 0)

    self._init_spins()

    while window.running:
      camera.track_user_inputs(window, movement_speed=0.03, yaw_speed=10, pitch_speed=10, hold_key=ti.ui.LMB)
      ro, target, up = camera.curr_position, camera.curr_lookat, camera.curr_up
      self._render(ro, target, up)
      canvas.set_image(self.image)
      window.show()
      
  @ti.kernel
  def _render(self, ro: ti.math.vec3, target: ti.math.vec3, up: ti.math.vec3):
    for u, v in self.image:
      rd = self._ray_dir(u, v, ro, target, up)
      tn, tf = self._hit_box(ro, rd, (0, 0, 0), (1, 1, 1))
      self.image[u, v] = ti.select(tf >= tn, ti.math.vec3(1.0, 1.0, 1.0), ti.math.vec3(0.0)) / tn
    
  @ti.func
  def _ray_dir(self, u: int, v: int, ro, target, up):
    W, H = self.window_size
    f = (target-ro).normalized()
    r = f.cross(up).normalized()
    w = r.cross(f)
    px = (2*(u + 0.5)/W - 1.0)
    py = (2*(v + 0.5)/H - 1.0)
    rd = f + px*r + py*w
    for d in ti.static(range(3)):
      if ti.abs(rd[d]) < 1e-6:
        rd[d] = 1e-6
    return rd
    
  @ti.func
  def _hit_box(self, ro, rd, lo, hi) -> tuple:
    inv = 1.0 / rd
    t0, t1 = (lo - ro) * inv, (hi - ro) * inv
    return ti.max(ti.min(t0, t1).max(), 0.0), ti.max(t0, t1).min()
   

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