"""Realistic synthetic Braille page generator."""

from .layout import Geometry, PageLayout, paginate, wrap_braille
from .render import RenderResult, render_page

__all__ = ["Geometry", "PageLayout", "paginate", "wrap_braille", "RenderResult", "render_page"]
