"""Interactive scatter plots linked to their source images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import io

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image
import plotly.graph_objects as go


def _css_color(value: str | Sequence[float]) -> str:
    if isinstance(value, str):
        return value
    channels = np.asarray(value, dtype=float)[:3]
    if channels.max() <= 1:
        channels *= 255
    red, green, blue = np.clip(channels.round(), 0, 255).astype(int)
    return f"rgb({red},{green},{blue})"


def _image_widget(image_bytes: bytes, width: int) -> widgets.Image:
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
    image.thumbnail((width, width))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return widgets.Image(
        value=output.getvalue(),
        format="jpeg",
        layout=widgets.Layout(max_width=f"{width}px", width="100%"),
    )


class EmbeddingScatterViewer:
    """Link points in one or more embedding plots to source images."""

    def __init__(
        self,
        *,
        coordinates_by_panel: Mapping[str, np.ndarray],
        images: Sequence[bytes],
        labels: Sequence[str],
        uuids: Sequence[str],
        colors: Mapping[str, str | Sequence[float]],
        title: str,
        link_panels: bool = False,
        panel_width: int = 430,
    ) -> None:
        if not coordinates_by_panel:
            raise ValueError("At least one coordinate panel is required")
        if len(images) == 0:
            raise ValueError("At least one image is required")
        if len(images) != len(labels) or len(images) != len(uuids):
            raise ValueError("Images, labels, and UUIDs must have equal lengths")

        self.images = list(images)
        self.labels = np.asarray(labels, dtype=np.str_)
        self.uuids = np.asarray(uuids, dtype=np.str_)
        self.colors = {
            str(label): _css_color(color) for label, color in colors.items()
        }
        self.title = title
        self.link_panels = link_panels
        self.panel_width = panel_width
        self.coordinates_by_panel = {}
        for panel_name, values in coordinates_by_panel.items():
            coordinates = np.asarray(values, dtype=float)
            if coordinates.shape != (len(images), 2):
                raise ValueError(
                    f"{panel_name} coordinates have shape {coordinates.shape}; "
                    f"expected {(len(images), 2)}"
                )
            if not np.isfinite(coordinates).all():
                raise ValueError(f"{panel_name} coordinates are not finite")
            self.coordinates_by_panel[str(panel_name)] = coordinates

        missing_colors = sorted(set(self.labels) - set(self.colors))
        if missing_colors:
            raise ValueError(f"Missing colors for labels: {missing_colors}")

        self.figures: dict[str, go.FigureWidget] = {}
        self.image_outputs: dict[str, widgets.Box] = {}
        self.detail_outputs: dict[str, widgets.HTML] = {}
        self.highlight_traces = {}
        panels = [
            self._build_panel(panel_name, coordinates)
            for panel_name, coordinates in self.coordinates_by_panel.items()
        ]
        self.widget = widgets.VBox(
            [
                widgets.HTML(f"<h4 style='margin:0'>{escape(title)}</h4>"),
                self._build_legend(),
                widgets.HBox(
                    panels,
                    layout=widgets.Layout(
                        width="100%",
                        gap="16px",
                        align_items="flex-start",
                        flex_flow="row wrap",
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

    def _build_legend(self) -> widgets.HTML:
        entries = "".join(
            "<span style='display:inline-flex;align-items:center;"
            "margin:0 12px 5px 0'>"
            f"<span style='width:10px;height:10px;background:{color};"
            "display:inline-block;margin-right:5px'></span>"
            f"<i>{escape(label)}</i></span>"
            for label, color in sorted(self.colors.items())
        )
        return widgets.HTML(
            f"<div style='line-height:1.3'>{entries}</div>"
        )

    def _build_panel(
        self,
        panel_name: str,
        coordinates: np.ndarray,
    ) -> widgets.VBox:
        figure = go.FigureWidget()
        for species in sorted(np.unique(self.labels)):
            indices = np.flatnonzero(self.labels == species)
            figure.add_trace(
                go.Scatter(
                    x=coordinates[indices, 0],
                    y=coordinates[indices, 1],
                    mode="markers",
                    text=self.labels[indices],
                    customdata=self.uuids[indices],
                    marker={
                        "size": 7,
                        "color": self.colors[str(species)],
                        "opacity": 0.78,
                    },
                    hovertemplate=(
                        "<i>%{text}</i><br>"
                        "Image %{customdata}<extra></extra>"
                    ),
                    showlegend=False,
                    name=str(species),
                )
            )
            trace = figure.data[-1]

            def select_point(
                _trace,
                points,
                _state,
                *,
                source_indices=indices,
                source_panel=panel_name,
            ) -> None:
                if points.point_inds:
                    image_index = int(
                        source_indices[int(points.point_inds[0])]
                    )
                    self._select(source_panel, image_index)

            trace.on_click(select_point)

        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 16,
                    "color": "rgba(255,255,255,0)",
                    "line": {"color": "#111111", "width": 3},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
        self.highlight_traces[panel_name] = figure.data[-1]
        figure.update_layout(
            title={"text": panel_name, "x": 0.5, "xanchor": "center"},
            height=430,
            width=self.panel_width,
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
            clickmode="event+select",
            dragmode="pan",
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        figure.update_xaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
        )
        figure.update_yaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )

        image_output = widgets.Box(
            [
                widgets.HTML(
                    "<em>Select a marker to inspect its source image.</em>"
                )
            ],
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                justify_content="center",
            ),
        )
        detail_output = widgets.HTML()
        self.figures[panel_name] = figure
        self.image_outputs[panel_name] = image_output
        self.detail_outputs[panel_name] = detail_output
        return widgets.VBox(
            [figure, image_output, detail_output],
            layout=widgets.Layout(
                width=f"{self.panel_width}px",
                max_width="100%",
            ),
        )

    def _select(self, source_panel: str, image_index: int) -> None:
        target_panels = (
            list(self.coordinates_by_panel)
            if self.link_panels
            else [source_panel]
        )
        for panel_name in target_panels:
            coordinates = self.coordinates_by_panel[panel_name]
            with self.figures[panel_name].batch_update():
                self.highlight_traces[panel_name].x = [
                    coordinates[image_index, 0]
                ]
                self.highlight_traces[panel_name].y = [
                    coordinates[image_index, 1]
                ]
            self.image_outputs[panel_name].children = (
                _image_widget(self.images[image_index], self.panel_width),
            )
            self.detail_outputs[panel_name].value = (
                f"<b>Recorded species:</b> "
                f"<i>{escape(str(self.labels[image_index]))}</i><br>"
                f"<b>Image ID:</b> "
                f"{escape(str(self.uuids[image_index])[:8])}"
            )

    def display(self) -> None:
        display(self.widget)
