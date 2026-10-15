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

    self.set_temperature(1 / T)

  def run(self):
    window = ti.ui.Window("Ising model", res=self.window_size)
    canvas = window.get_canvas()
    camera = ti.ui.Camera()
    camera.position(0.5, 3, 1.5)
    camera.lookat(0.5, 0.5, 0.0)
    camera.up(0, 1, 0)

    self._init_spins()

    while window.running:
      camera.track_user_inputs(window, movement_speed=0.03, yaw_speed=10, pitch_speed=10, hold_key=ti.ui.LMB)
      
      self._simulate_half(0) 
      self._simulate_half(1)
      
      ro, target, up = camera.curr_position, camera.curr_lookat, camera.curr_up
      self._render(ro, target, up)
      canvas.set_image(self.image)
      window.show()
      
  @ti.kernel
  def _render(self, ro: ti.math.vec3, target: ti.math.vec3, up: ti.math.vec3):
    for u, v in self.image:
      self.image[u, v] = self._ray_trace(u, v, ro, target, up)
      
  @ti.func 
  def _ray_trace(self, u: int, v: int, ro, target, up):
    GRID_SIDE = self.spins.shape[0]
    MAX_STEPS = GRID_SIDE*3
    
    rd = self._ray_dir(u, v, ro, target, up)
    grid_min, grid_max = ti.math.vec3(0, 0, 0), ti.math.vec3(1, 1, 1)
    box = self._hit_box(ro, rd, grid_min, grid_max)
    
    T = 1.0
    ALPHA = 0.01
    color = tm.vec3(0.0)
    VOXEL_COLOR = tm.vec3(0.5, 0.8, 0.9)
    rd_len = rd.norm()
    sigma = -ti.log(1.0 - ALPHA)
    s_prev = 0.0
    
    if box[0] <= box[1]:
      entry_t = box[0]
    
      v_per_u = GRID_SIDE/(grid_max - grid_min)
      entry_pos = ((ro + rd * (entry_t + 0.0001)) - grid_min) * v_per_u
      step = ti.cast(ti.math.sign(rd), ti.i32)
      delta = ti.abs(1/rd)
      pos = ti.cast(ti.math.clamp(entry_pos, 0, GRID_SIDE-1), ti.i32)
     
      next_entry_t = (pos - entry_pos + max(step, 0)) / rd
      for _ in range(MAX_STEPS):
        s_next = next_entry_t.min()
        if self.spins[pos.x, pos.y, pos.z] == 1:
          a = 1.0 - ti.exp(-sigma * (s_next - s_prev) * rd_len)
          color += T * a * VOXEL_COLOR
          T *= 1 - a
          if T < 1e-3:
            break
          s_prev = s_next

        if next_entry_t.x < next_entry_t.y:
          if next_entry_t.x < next_entry_t.z:
            pos.x += step.x
            if pos.x < 0 or pos.x >= GRID_SIDE:
              break
            next_entry_t.x += delta.x
          else:
            pos.z += step.z
            if pos.z < 0 or pos.z >= GRID_SIDE:
              break
            next_entry_t.z += delta.z
        else:
          if next_entry_t.y < next_entry_t.z:
            pos.y += step.y
            if pos.y < 0 or pos.y >= GRID_SIDE:
              break
            next_entry_t.y += delta.y
          else:
            pos.z += step.z
            if pos.z < 0 or pos.z >= GRID_SIDE:
              break
            next_entry_t.z += delta.z
    return color
    
  @ti.func
  def _ray_dir(self, u: int, v: int, ro, target, up):
    W, H = self.window_size
    f = (target-ro).normalized()
    r = f.cross(up).normalized()
    w = r.cross(f)
    px = (2*(u + 0.5)/W - 1.0) * W/H
    py = (2*(v + 0.5)/H - 1.0) * W/H
    rd = f + px*r + py*w
    for d in ti.static(range(3)):
      if ti.abs(rd[d]) < 1e-6:
        rd[d] = 1e-6
    return rd
    
  @ti.func
  def _hit_box(self, ro, rd, lo, hi):
    inv = 1.0 / rd
    t0, t1 = (lo - ro) * inv, (hi - ro) * inv
    t_near = ti.min(t0, t1).max()
    t_far  = ti.max(t0, t1).min()
    return tm.vec2(ti.max(t_near, 0.0), t_far)

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