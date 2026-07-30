"""
Self-contained pyglet rendering utilities for SlimeVolley.

This is a trimmed re-implementation of the ``rendering`` helpers that used to
ship inside ``gym.envs.classic_control.rendering`` (removed in Gym >= 0.26 and
absent from Gymnasium). Only the pieces used by the slime environment are kept:
``Viewer``, ``SimpleImageViewer``, ``Transform``, ``Color`` and a small set of
geometry primitives (``FilledPolygon``, ``PolyLine``, ``make_polygon``,
``make_circle``).

This module is imported lazily by ``slimevolley.py`` (only when rendering is
actually requested), so importing the environment itself does not require
pyglet or a display.
"""

import math

import numpy as np
import pyglet
from pyglet import gl


class Attr:
    def enable(self):
        pass

    def disable(self):
        pass


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        gl.glColor4f(*self.vec4)


class Transform(Attr):
    """A 2D translate/rotate/scale applied via the OpenGL modelview stack."""

    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1.0, 1.0)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def set_translation(self, x, y):
        self.translation = (float(x), float(y))

    def set_rotation(self, rot):
        self.rotation = float(rot)

    def set_scale(self, sx, sy):
        self.scale = (float(sx), float(sy))

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.translation[0], self.translation[1], 0.0)
        gl.glRotatef(self.rotation, 0.0, 0.0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1.0)

    def disable(self):
        gl.glPopMatrix()


class Geom:
    def __init__(self):
        self._color = Color((0.0, 0.0, 0.0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in self.attrs:
            attr.enable()
        self.render1()
        for attr in reversed(self.attrs):
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, *args):
        if len(args) == 3:
            self._color.vec4 = (float(args[0]), float(args[1]), float(args[2]), 1.0)
        else:
            self._color.vec4 = tuple(float(c) for c in args)


def _flatten(v):
    flat = []
    for (x, y) in v:
        flat.append(float(x))
        flat.append(float(y))
    return flat


class FilledPolygon(Geom):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def render1(self):
        n = len(self.v)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        pyglet.graphics.draw(n, gl.GL_POLYGON, ('v2f', _flatten(self.v)))
        gl.glDisable(gl.GL_BLEND)


class PolyLine(Geom):
    def __init__(self, v, close):
        super().__init__()
        self.v = v
        self.close = close

    def render1(self):
        n = len(self.v)
        mode = gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP
        pyglet.graphics.draw(n, mode, ('v2f', _flatten(self.v)))


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    return PolyLine(v, True)


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    return PolyLine(points, True)


class Viewer:
    """Minimal pyglet viewer mimicking the old gym ``rendering.Viewer``."""

    def __init__(self, width, height, display=None, visible=True):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height,
                                           display=display, visible=visible)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def close(self):
        try:
            self.window.close()
        except Exception:
            pass

    def window_closed_by_user(self):
        self.close()

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        self.window.switch_to()
        self.window.dispatch_events()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data('RGBA', buffer.width * 4),
                                dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        self.onetime_geoms = []
        return arr


class SimpleImageViewer:
    """Shows a numpy RGB array in a pyglet window (replaces gym's version)."""

    def __init__(self, display=None, maxwidth=500):
        self.display = display
        self.maxwidth = maxwidth
        self.window = None
        self.width = None
        self.height = None
        self.isopen = False

    def imshow(self, arr):
        arr = np.ascontiguousarray(arr)
        if self.window is None:
            h, w = arr.shape[:2]
            self.width = w
            self.height = h
            self.window = pyglet.window.Window(width=w, height=h,
                                               display=self.display)
            self.window.on_close = self.close
            self.isopen = True

        self.window.switch_to()
        self.window.dispatch_events()
        image = pyglet.image.ImageData(self.width, self.height, 'RGB',
                                       arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen and self.window is not None:
            self.window.close()
            self.isopen = False
            self.window = None

    def __bool__(self):
        return self.isopen
