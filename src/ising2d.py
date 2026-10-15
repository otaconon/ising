import taichi as ti
import numpy as np

@ti.data_oriented
class Ising2d:
  def __init__(self, window_size: tuple, dims: tuple, T: float):
    self.window_size = window_size
    
    self.acc = ti.field(ti.f32, shape=5) 
    self.spins = ti.field(dtype=ti.i8, shape=dims)
    self.img = ti.field(ti.f32, shape=dims)
    self.set_temperature(1/T)

  def run(self):
    window = ti.ui.Window("Ising model", res=self.window_size)
    canvas = window.get_canvas()

    self._init_voxels()

    while window.running:
      self._simulate_half(0)
      self._simulate_half(1)
      
      self._render()
      canvas.set_image(self.img)
      
      window.show()
    
  @ti.kernel
  def _render(self):
    for x, y in self.spins:
      self.img[x, y] = (self.spins[x, y] + 1) * 0.5 
      
  @ti.kernel
  def set_temperature(self, beta: float):
    for i in self.acc:
      self.acc[i] = ti.exp(-4.0 * beta * (i - 2)) 

  @ti.func
  def _dE(self, x: int, y: int) -> int:
    N, M = self.spins.shape
    nn = 0
    for dx, dy in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
      nn += self.spins[(x + dx) % N, (y + dy) % M]
    return 2 * self.spins[x, y] * nn

  @ti.func
  def _flip(self, x: int, y: int) -> bool:
    return ti.random() < self.acc[self._dE(x, y) // 4 + 2]

  @ti.kernel
  def _init_voxels(self):
    for x, y in self.spins:
      self.spins[x, y] = 1 if ti.random(float) <= 0.5 else -1

  @ti.kernel
  def _simulate_half(self, parity: int):
    N, M = self.spins.shape
    for x, h in ti.ndrange(N, M//2):
      y = 2 * h + (x + parity) % 2
      if self._flip(x, y):
        self.spins[x, y] = -self.spins[x, y]
