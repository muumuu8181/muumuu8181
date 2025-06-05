import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_pygame_stub():
    pygame = types.ModuleType('pygame')
    pygame.init = lambda: None
    pygame.quit = lambda: None
    pygame.QUIT = 'QUIT'
    pygame.K_LEFT = 0
    pygame.K_RIGHT = 1
    pygame.K_UP = 2
    pygame.K_DOWN = 3
    pygame.display = types.SimpleNamespace(
        set_mode=lambda size: types.SimpleNamespace(
            fill=lambda color: None,
            blit=lambda surf, rect: None,
            get_rect=lambda: types.SimpleNamespace(left=0, right=0, top=0, bottom=0),
        ),
        set_caption=lambda caption: None,
        flip=lambda: None,
    )
    pygame.time = types.SimpleNamespace(Clock=lambda: types.SimpleNamespace(tick=lambda fps: None))
    pygame.font = types.SimpleNamespace(
        Font=lambda *a, **k: types.SimpleNamespace(
            render=lambda *a, **k: types.SimpleNamespace(
                get_rect=lambda **k: types.SimpleNamespace(
                    center=(0, 0),
                    x=0,
                    y=0,
                    left=0,
                    right=0,
                    top=0,
                    bottom=0,
                )
            )
        )
    )
    pygame.event = types.SimpleNamespace(get=lambda: [types.SimpleNamespace(type=pygame.QUIT)])
    pygame.key = types.SimpleNamespace(get_pressed=lambda: {pygame.K_LEFT: 0, pygame.K_RIGHT: 0, pygame.K_UP: 0, pygame.K_DOWN: 0})
    pygame.draw = types.SimpleNamespace(rect=lambda *a, **k: None)

    def rect(x, y, w, h):
        r = types.SimpleNamespace(
            x=x,
            y=y,
            width=w,
            height=h,
            left=x,
            right=x + w,
            top=y,
            bottom=y + h,
        )
        r.clamp_ip = lambda other: None
        r.colliderect = lambda other: False
        return r

    pygame.Rect = rect
    return pygame


def test_main_runs(monkeypatch):
    pygame_stub = make_pygame_stub()
    monkeypatch.setitem(sys.modules, 'pygame', pygame_stub)
    hello_codex = importlib.import_module('hello_codex')
    import pytest
    with pytest.raises(SystemExit):
        hello_codex.main()
