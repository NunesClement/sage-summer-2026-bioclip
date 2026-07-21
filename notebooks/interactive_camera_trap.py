"""Interactive inspection of genus-only camera-trap predictions."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import io
import json
from pathlib import Path
from typing import Callable, Sequence
from urllib.request import urlretrieve

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw


MEGADETECTOR_RESULTS_URL = (
    "https://raw.githubusercontent.com/Imageomics/"
    "sage-summer-2026-bioclip/main/data/"
    "lila_peromyscus/megadetector_results.json"
)


def format_details(fields: Sequence[tuple[str, object]]) -> str:
    """Format escaped label-value pairs for an image browser."""
    return "<br>".join(
        f"<b>{escape(str(label))}:</b> {escape(str(value))}"
        for label, value in fields
    )


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    """Format an escaped compact table for an image browser."""
    header_html = "".join(
        f"<th style='text-align:left;padding-right:12px'>"
        f"{escape(str(header))}</th>"
        for header in headers
    )
    row_html = "".join(
        "<tr>"
        + "".join(
            f"<td style='padding-right:12px'>{escape(str(value))}</td>"
            for value in row
        )
        + "</tr>"
        for row in rows
    )
    return (
        "<table style='border-collapse:collapse'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{row_html}</tbody></table>"
    )


@dataclass(frozen=True)
class Prediction:
    model: str
    precision: str
    method: str
    species: str
    margin: float


@dataclass(frozen=True)
class PredictionSource:
    model: str
    precision: str
    method: str
    predict: Callable[[int], tuple[str, float]]


def _top_two_margin(scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(scores).reshape(-1))
    return float(ordered[-1] - ordered[-2])


def build_prediction_sources(
    *,
    model_names: Sequence[str],
    class_names: Sequence[str],
    prototype_method: str,
    prototypes: dict[str, np.ndarray],
    variants: dict[str, tuple[dict, dict]],
) -> list[PredictionSource]:
    """Build prototype and SVM predictors for each model representation."""
    class_array = np.asarray(class_names)
    sources = []

    for precision, (bundles, fitted_svms) in variants.items():
        for model_name in model_names:
            features = bundles[model_name].features
            model_prototypes = prototypes[model_name]
            svm = fitted_svms[model_name]

            def predict_prototype(
                index: int,
                features=features,
                model_prototypes=model_prototypes,
            ) -> tuple[str, float]:
                scores = features[index] @ model_prototypes.T
                return (
                    str(class_array[int(np.argmax(scores))]),
                    _top_two_margin(scores),
                )

            def predict_svm(
                index: int,
                features=features,
                svm=svm,
            ) -> tuple[str, float]:
                row = features[index : index + 1]
                scores = svm.decision_function(row)[0]
                return str(svm.predict(row)[0]), _top_two_margin(scores)

            sources.extend([
                PredictionSource(
                    model=model_name,
                    precision=precision,
                    method=prototype_method,
                    predict=predict_prototype,
                ),
                PredictionSource(
                    model=model_name,
                    precision=precision,
                    method="40-shot linear SVM",
                    predict=predict_svm,
                ),
            ])

    return sources


def load_megadetector_boxes(
    expected_uuids: Sequence[str],
    *,
    confidence_threshold: float = 0.30,
    results_path: Path | None = None,
) -> list[list[list[float]]]:
    """Load stored animal boxes in the same order as the camera images."""
    if results_path is None:
        artifact_relative_path = Path(
            "data/lila_peromyscus/megadetector_results.json"
        )
        module_path = Path(__file__).resolve()
        candidates = [
            Path.cwd() / artifact_relative_path,
            module_path.parents[1] / artifact_relative_path,
            module_path.with_name("megadetector_results.json"),
        ]
        results_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            module_path.with_name("megadetector_results.json"),
        )
        if not results_path.exists():
            urlretrieve(MEGADETECTOR_RESULTS_URL, results_path)

    with results_path.open() as source:
        results = json.load(source)

    detector = results.get("info", {}).get("detector")
    if detector != "MDV1000-redwood":
        raise ValueError(f"Unexpected MegaDetector model: {detector}")
    by_uuid = {
        Path(row["file"]).stem: row
        for row in results.get("images", [])
    }

    ordered_boxes = []
    for uuid in expected_uuids:
        row = by_uuid.get(str(uuid))
        if row is None:
            raise ValueError(f"MegaDetector results are missing UUID {uuid}")
        boxes = [
            [float(value) for value in detection["bbox"]]
            for detection in row.get("detections", [])
            if detection["category"] == "1"
            and float(detection["conf"]) >= confidence_threshold
        ]
        if not boxes:
            raise ValueError(
                f"UUID {uuid} has no animal detection at "
                f"{confidence_threshold:.2f}"
            )
        if any(
            len(box) != 4
            or min(box) < 0
            or box[0] + box[2] > 1
            or box[1] + box[3] > 1
            for box in boxes
        ):
            raise ValueError(f"UUID {uuid} has an invalid normalized box")
        ordered_boxes.append(boxes)
    return ordered_boxes


def _image_widget(
    image_bytes: bytes,
    width: int,
    boxes: Sequence[Sequence[float]] = (),
) -> widgets.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((width, width))
    if boxes:
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size
        for x, y, box_width, box_height in boxes:
            draw.rectangle(
                (
                    round(x * image_width),
                    round(y * image_height),
                    round((x + box_width) * image_width),
                    round((y + box_height) * image_height),
                ),
                outline=(255, 0, 0),
                width=2,
            )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return widgets.Image(
        value=output.getvalue(),
        format="jpeg",
        layout=widgets.Layout(max_width=f"{width}px", width="100%"),
    )


class ImageBrowser:
    """Display one image at a time with sequential or random navigation."""

    def __init__(
        self,
        *,
        images: Sequence[bytes | Path | str],
        details: Sequence[str],
        title: str,
        boxes: Sequence[Sequence[Sequence[float]]] | None = None,
        width: int = 620,
        random_seed: int = 42,
    ) -> None:
        if len(images) == 0:
            raise ValueError("At least one image is required")
        if len(images) != len(details):
            raise ValueError("Images and details must have equal lengths")
        if boxes is not None and len(images) != len(boxes):
            raise ValueError("Images and boxes must have equal lengths")

        self.images = list(images)
        self.details = list(details)
        self.boxes = (
            list(boxes) if boxes is not None else [[] for _ in images]
        )
        self.width = width
        self.index = 0
        self.rng = np.random.default_rng(random_seed)
        self.change_callbacks: list[Callable[[int], None]] = []

        self.previous_button = widgets.Button(
            icon="arrow-left",
            tooltip="Previous image",
            layout=widgets.Layout(width="42px"),
        )
        self.next_button = widgets.Button(
            icon="arrow-right",
            tooltip="Next image",
            layout=widgets.Layout(width="42px"),
        )
        self.navigation_mode = widgets.ToggleButtons(
            options=[
                ("Sequential", "sequential"),
                ("Random", "random"),
            ],
            value="sequential",
            tooltips=[
                "Move to the next image in order",
                "Draw another image at random",
            ],
        )
        self.counter = widgets.HTML()
        self.image = widgets.Box()
        self.detail = widgets.HTML(
            layout=widgets.Layout(
                max_height="360px",
                overflow_x="auto",
                overflow_y="auto",
            )
        )

        self.previous_button.on_click(lambda _: self._change(-1))
        self.next_button.on_click(lambda _: self._change(1))
        self.widget = widgets.VBox(
            [
                widgets.HTML(f"<h4 style='margin:0'>{escape(title)}</h4>"),
                widgets.HBox(
                    [
                        self.previous_button,
                        self.next_button,
                        self.navigation_mode,
                        self.counter,
                    ],
                    layout=widgets.Layout(
                        align_items="center",
                        flex_flow="row wrap",
                    ),
                ),
                self.image,
                self.detail,
            ],
            layout=widgets.Layout(width="100%", max_width=f"{width}px"),
        )
        self._render()

    def _read_image(self) -> bytes:
        source = self.images[self.index]
        if isinstance(source, bytes):
            return source
        return Path(source).read_bytes()

    def _change(self, step: int) -> None:
        if (
            step > 0
            and self.navigation_mode.value == "random"
            and len(self.images) > 1
        ):
            offset = int(self.rng.integers(1, len(self.images)))
            self.index = (self.index + offset) % len(self.images)
        else:
            self.index = (self.index + step) % len(self.images)
        self._render()

    def _render(self) -> None:
        self.image.children = (
            _image_widget(
                self._read_image(),
                self.width,
                self.boxes[self.index],
            ),
        )
        self.counter.value = (
            f"<b>{self.index + 1} / {len(self.images)}</b>"
        )
        self.detail.value = self.details[self.index]
        for callback in self.change_callbacks:
            callback(self.index)

    def on_change(
        self,
        callback: Callable[[int], None],
        *,
        call_immediately: bool = False,
    ) -> None:
        self.change_callbacks.append(callback)
        if call_immediately:
            callback(self.index)

    def display(self) -> None:
        display(self.widget)


class ClassificationErrorViewer:
    """Compare a classifier error with labeled predicted-class examples."""

    def __init__(
        self,
        *,
        error_images: Sequence[bytes],
        error_details: Sequence[str],
        predicted_labels: Sequence[str],
        reference_images: Sequence[bytes],
        reference_labels: Sequence[str],
        title: str,
        random_seed: int = 42,
    ) -> None:
        if len(error_images) != len(predicted_labels):
            raise ValueError(
                "Error images and predicted labels must have equal lengths"
            )
        if len(reference_images) != len(reference_labels):
            raise ValueError(
                "Reference images and labels must have equal lengths"
            )

        self.predicted_labels = np.asarray(
            predicted_labels, dtype=np.str_
        )
        self.reference_images = list(reference_images)
        self.reference_labels = np.asarray(
            reference_labels, dtype=np.str_
        )
        self.random_seed = random_seed
        self.reference_browser = None

        self.error_browser = ImageBrowser(
            images=error_images,
            details=error_details,
            title=title,
            random_seed=random_seed,
        )
        self.prediction_button = widgets.Button(
            icon="search",
            tooltip="Show labeled examples of the predicted species",
            layout=widgets.Layout(width="auto"),
        )
        self.prediction_button.on_click(
            lambda _: self._show_references()
        )
        prediction_row = widgets.HBox(
            [
                widgets.HTML("<b>SVM prediction:</b>"),
                self.prediction_button,
            ],
            layout=widgets.Layout(
                align_items="center",
                flex_flow="row wrap",
            ),
        )
        self.reference_panel = widgets.VBox(
            [
                widgets.HTML(
                    "<h4 style='margin:0'>Predicted-species references</h4>"
                ),
                widgets.HTML(
                    "Select the SVM prediction to inspect labeled examples."
                ),
            ],
            layout=widgets.Layout(
                width="calc(50% - 9px)",
                min_width="360px",
            ),
        )
        left_panel = widgets.VBox(
            [self.error_browser.widget, prediction_row],
            layout=widgets.Layout(
                width="calc(50% - 9px)",
                min_width="360px",
            ),
        )
        self.widget = widgets.HBox(
            [left_panel, self.reference_panel],
            layout=widgets.Layout(
                width="100%",
                gap="18px",
                align_items="flex-start",
                flex_flow="row wrap",
            ),
        )
        self.error_browser.on_change(
            self._error_changed,
            call_immediately=True,
        )

    def _error_changed(self, index: int) -> None:
        species = str(self.predicted_labels[index])
        self.prediction_button.description = species
        self.prediction_button.tooltip = (
            f"Show labeled {species} examples"
        )
        self.reference_browser = None
        self.reference_panel.children = (
            widgets.HTML(
                "<h4 style='margin:0'>Predicted-species references</h4>"
            ),
            widgets.HTML(
                "Select the SVM prediction to inspect labeled examples."
            ),
        )

    def _show_references(self) -> None:
        species = str(
            self.predicted_labels[self.error_browser.index]
        )
        indices = np.flatnonzero(self.reference_labels == species)
        if len(indices) == 0:
            self.reference_panel.children = (
                widgets.HTML(
                    f"No labeled <i>{escape(species)}</i> examples "
                    "are available."
                ),
            )
            return

        self.reference_browser = ImageBrowser(
            images=[self.reference_images[index] for index in indices],
            details=[
                format_details([
                    ("Dataset label", species),
                    ("Reference", f"{position + 1} of {len(indices)}"),
                ])
                for position in range(len(indices))
            ],
            title=f"Labeled {species} examples",
            random_seed=self.random_seed,
        )
        self.reference_panel.children = (
            self.reference_browser.widget,
        )

    def display(self) -> None:
        display(self.widget)


class CameraTrapViewer:
    """Two-panel viewer for camera images and labeled references."""

    def __init__(
        self,
        *,
        camera_images: Sequence[bytes],
        camera_metadata: Sequence[dict],
        camera_boxes: Sequence[Sequence[Sequence[float]]],
        prediction_sources: Sequence[PredictionSource],
        reference_images: Sequence[bytes],
        reference_labels: Sequence[str],
        random_seed: int = 42,
    ) -> None:
        if len(camera_images) != len(camera_metadata):
            raise ValueError("Camera images and metadata must have equal lengths")
        if len(camera_images) != len(camera_boxes):
            raise ValueError("Camera images and boxes must have equal lengths")
        if not camera_images:
            raise ValueError("At least one camera image is required")
        if len(reference_images) != len(reference_labels):
            raise ValueError(
                "Reference images and labels must have equal lengths"
            )

        self.camera_images = list(camera_images)
        self.camera_metadata = list(camera_metadata)
        self.camera_boxes = list(camera_boxes)
        self.prediction_sources = list(prediction_sources)
        self.reference_images = list(reference_images)
        self.reference_labels = np.asarray(reference_labels)
        self.camera_index = 0
        self.reference_positions: dict[str, int] = {}

        self.camera_rng = np.random.default_rng(random_seed)
        reference_rng = np.random.default_rng(random_seed)
        self.reference_indices = {}
        for species in np.unique(self.reference_labels):
            indices = np.flatnonzero(self.reference_labels == species)
            self.reference_indices[str(species)] = (
                reference_rng.permutation(indices)
            )

        self.previous_button = widgets.Button(
            icon="arrow-left",
            tooltip="Previous camera image",
            layout=widgets.Layout(width="42px"),
        )
        self.next_button = widgets.Button(
            icon="arrow-right",
            tooltip="Next camera image",
            layout=widgets.Layout(width="42px"),
        )
        self.image_counter = widgets.HTML()
        self.navigation_mode = widgets.ToggleButtons(
            options=[
                ("Sequential", "sequential"),
                ("Random", "random"),
            ],
            value="sequential",
            tooltips=[
                "Move to the next camera image in order",
                "Draw another camera image at random",
            ],
        )
        self.camera_image = widgets.Box()
        self.camera_details = widgets.HTML()
        self.run_button = widgets.Button(
            description=f"Run {len(self.prediction_sources)} predictions",
            icon="play",
            button_style="primary",
        )
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=len(self.prediction_sources),
            description="Predictions",
            bar_style="",
            layout=widgets.Layout(width="100%"),
        )
        self.progress_text = widgets.HTML()
        self.results = widgets.HTML(
            "<em>Run the predictions for this image.</em>"
        )
        self.species_buttons = widgets.GridBox(
            layout=widgets.Layout(
                grid_template_columns="repeat(2, minmax(0, 1fr))",
                grid_gap="6px",
            )
        )

        self.reference_heading = widgets.HTML(
            "<h4 style='margin:0'>Labeled reference examples</h4>"
        )
        self.reference_status = widgets.HTML(
            "Select a predicted species to add an example."
        )
        self.clear_button = widgets.Button(
            description="Clear",
            icon="trash",
            tooltip="Clear reference examples",
            layout=widgets.Layout(width="90px"),
        )
        self.reference_examples = widgets.VBox()

        self.previous_button.on_click(
            lambda _: self._change_camera_image(-1)
        )
        self.next_button.on_click(lambda _: self._change_camera_image(1))
        self.run_button.on_click(self._run_predictions)
        self.clear_button.on_click(lambda _: self._clear_references())

        left_panel = widgets.VBox(
            [
                widgets.HTML(
                    "<h4 style='margin:0'>Genus-only camera-trap image</h4>"
                ),
                widgets.HBox(
                    [
                        self.previous_button,
                        self.next_button,
                        self.navigation_mode,
                        self.image_counter,
                    ],
                    layout=widgets.Layout(
                        align_items="center",
                        flex_flow="row wrap",
                    ),
                ),
                self.camera_image,
                self.camera_details,
                self.run_button,
                self.progress,
                self.progress_text,
                self.species_buttons,
                self.results,
            ],
            layout=widgets.Layout(
                width="calc(50% - 9px)",
                min_width="360px",
            ),
        )
        right_panel = widgets.VBox(
            [
                widgets.HBox(
                    [self.reference_heading, self.clear_button],
                    layout=widgets.Layout(
                        justify_content="space-between",
                        align_items="center",
                    ),
                ),
                self.reference_status,
                self.reference_examples,
            ],
            layout=widgets.Layout(
                width="calc(50% - 9px)",
                min_width="360px",
                max_height="900px",
                overflow_y="auto",
            ),
        )
        self.widget = widgets.HBox(
            [left_panel, right_panel],
            layout=widgets.Layout(
                width="100%",
                gap="18px",
                align_items="flex-start",
                flex_flow="row wrap",
            ),
        )
        self._render_camera_image()

    def _render_camera_image(self) -> None:
        metadata = self.camera_metadata[self.camera_index]
        self.camera_image.children = (
            _image_widget(
                self.camera_images[self.camera_index],
                620,
                self.camera_boxes[self.camera_index],
            ),
        )
        self.image_counter.value = (
            f"<b>{self.camera_index + 1} / {len(self.camera_images)}</b>"
        )
        self.camera_details.value = (
            f"<b>Source:</b> {escape(str(metadata['publisher']))}<br>"
            f"<b>Image ID:</b> "
            f"{escape(str(metadata['uuid'])[:8])}<br>"
            f"<a href='{escape(str(metadata['source_url']), quote=True)}' "
            "target='_blank'>Open source image</a>"
        )
        self.progress.value = 0
        self.progress.bar_style = ""
        self.progress_text.value = ""
        self.results.value = "<em>Run the predictions for this image.</em>"
        self.species_buttons.children = ()
        self._clear_references()

    def _change_camera_image(self, step: int) -> None:
        if (
            step > 0
            and self.navigation_mode.value == "random"
            and len(self.camera_images) > 1
        ):
            offset = int(
                self.camera_rng.integers(1, len(self.camera_images))
            )
            self.camera_index = (
                self.camera_index + offset
            ) % len(self.camera_images)
        else:
            self.camera_index = (
                self.camera_index + step
            ) % len(self.camera_images)
        self._render_camera_image()

    def _run_predictions(self, _: widgets.Button) -> None:
        self.run_button.disabled = True
        self.progress.value = 0
        self.progress.bar_style = "info"
        predictions = []
        try:
            for completed, source in enumerate(
                self.prediction_sources, start=1
            ):
                self.progress_text.value = (
                    f"{escape(source.model)} | "
                    f"{escape(source.precision)} | "
                    f"{escape(source.method)}"
                )
                species, margin = source.predict(self.camera_index)
                predictions.append(
                    Prediction(
                        model=source.model,
                        precision=source.precision,
                        method=source.method,
                        species=species,
                        margin=margin,
                    )
                )
                self.progress.value = completed
        except Exception:
            self.progress.bar_style = "danger"
            raise
        else:
            self.progress.bar_style = "success"
            self.progress_text.value = (
                f"Completed {len(predictions)} predictions."
            )
            self._render_predictions(predictions)
        finally:
            self.run_button.disabled = False

    def _render_predictions(
        self, predictions: Sequence[Prediction]
    ) -> None:
        species, counts = np.unique(
            [prediction.species for prediction in predictions],
            return_counts=True,
        )
        ordered_species = sorted(
            zip(species, counts),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        buttons = []
        for species_name, count in ordered_species:
            button = widgets.Button(
                description=f"{species_name} ({count})",
                tooltip=f"Add a labeled {species_name} reference image",
                layout=widgets.Layout(width="auto"),
            )
            button.on_click(
                lambda _, selected=str(species_name): (
                    self._add_reference(selected)
                )
            )
            buttons.append(button)
        self.species_buttons.children = tuple(buttons)

        rows = []
        for prediction in predictions:
            rows.append(
                "<tr>"
                f"<td>{escape(prediction.model)}</td>"
                f"<td>{escape(prediction.precision)}</td>"
                f"<td>{escape(prediction.method)}</td>"
                f"<td><i>{escape(prediction.species)}</i></td>"
                f"<td>{prediction.margin:.3f}</td>"
                "</tr>"
            )
        self.results.value = (
            "<table style='width:100%; border-collapse:collapse'>"
            "<thead><tr>"
            "<th>Model</th><th>Precision</th><th>Method</th>"
            "<th>Prediction</th><th>Margin</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    def _add_reference(self, species: str) -> None:
        indices = self.reference_indices.get(species)
        if indices is None or len(indices) == 0:
            self.reference_status.value = (
                f"No labeled references are available for "
                f"<i>{escape(species)}</i>."
            )
            return

        position = self.reference_positions.get(species, 0)
        image_index = int(indices[position % len(indices)])
        self.reference_positions[species] = position + 1
        example = widgets.VBox(
            [
                widgets.HTML(
                    f"<b><i>{escape(species)}</i></b> | "
                    f"example {position % len(indices) + 1} "
                    f"of {len(indices)}"
                ),
                _image_widget(self.reference_images[image_index], 520),
            ]
        )
        self.reference_examples.children = (example,)
        self.reference_status.value = (
            "Click a species button again to add another labeled example."
        )

    def _clear_references(self) -> None:
        self.reference_examples.children = ()
        self.reference_status.value = (
            "Select a predicted species to add an example."
        )

    def display(self) -> None:
        display(self.widget)
