import base64
import collections
import csv
import ctypes
import json
import pkgutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from saprot.data.sequence_to_prosst import (
    ESMFOLD_MAX_RESIDUES,
    normalize_protein_sequence,
)
from saprot.model.prosst.specs import (
    DEFAULT_PROSST_MODEL,
    PROSST_MODEL_SPECS,
    PROSST_HUB_NAMESPACE,
    PROSST_HUB_URL,
    get_prosst_model_spec,
)


COLAB_ENVIRONMENT_GENERATION = "2026-07-12-clean-kernel-v2"
COLAB_ENVIRONMENT_MARKER = Path(
    "/content/.cache/colabprosst/environment_generation"
)


def _validate_colab_environment():
    try:
        import google.colab  # noqa: F401
    except Exception:
        return

    actual_generation = (
        COLAB_ENVIRONMENT_MARKER.read_text(encoding="utf-8").strip()
        if COLAB_ENVIRONMENT_MARKER.is_file()
        else ""
    )
    if actual_generation != COLAB_ENVIRONMENT_GENERATION:
        raise RuntimeError(
            "This Colab tab is running an outdated ColabProSST bootstrap. "
            "Open a new notebook tab from the current GitHub Colab URL, "
            "run its code cell, wait for the one-time Python kernel restart, "
            "and then run that cell once more."
        )


class _UploadField:
    def __init__(self, ui, description, placeholder):
        self.ui = ui
        widgets = ui.widgets
        upload_id = uuid.uuid4().hex
        self.input_id = f"colabprosst-files-{upload_id}"
        self.output_id = f"colabprosst-result-{upload_id}"
        self.path = widgets.Text(
            value="",
            placeholder=placeholder,
            description=description,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.upload_button = widgets.Button(
            description="Upload your file",
            button_style="info",
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.inline_upload = widgets.HTML(
            value=self._inline_upload_html(),
            layout=widgets.Layout(width=ui.WIDTH, display="none"),
        )
        self.status = widgets.HTML(layout=widgets.Layout(width=ui.WIDTH))
        self.upload_button.on_click(self._upload)
        self.items = [
            self.path,
            self.upload_button,
            self.inline_upload,
            self.status,
        ]

    def _inline_upload_html(self):
        try:
            files_js = pkgutil.get_data(
                "google.colab.files", "resources/files.js"
            )
        except (ImportError, ModuleNotFoundError):
            return ""
        if files_js is None:
            return ""

        return """
            <style>
              #{container_id} {{
                align-items: center;
                display: flex;
                gap: 10px;
                min-height: 30px;
              }}
              #{input_id} {{
                border: 0;
                clip: rect(0 0 0 0);
                clip-path: inset(50%);
                height: 1px;
                margin: -1px;
                overflow: hidden;
                padding: 0;
                position: absolute;
                white-space: nowrap;
                width: 1px;
              }}
              #{label_id} {{
                background: #1a73e8;
                border-radius: 2px;
                color: #fff;
                cursor: pointer;
                display: inline-flex;
                font: 500 13px Arial, sans-serif;
                padding: 6px 12px;
              }}
              #{input_id}:disabled + #{label_id} {{
                cursor: wait;
                opacity: 0.65;
              }}
              #{filename_id} {{
                color: inherit;
                font: 13px Arial, sans-serif;
              }}
            </style>
            <div id="{container_id}">
              <input type="file" id="{input_id}" name="files[]" disabled
                     onchange="document.getElementById('{filename_id}').textContent = this.files.length ? this.files[0].name : 'No file selected'" />
              <label id="{label_id}" for="{input_id}">Choose file</label>
              <span id="{filename_id}">No file selected</span>
            </div>
            <output id="{output_id}"></output>
            <script>{files_js}</script>
        """.format(
            input_id=self.input_id,
            output_id=self.output_id,
            container_id=f"{self.input_id}-container",
            label_id=f"{self.input_id}-label",
            filename_id=f"{self.input_id}-filename",
            files_js=files_js.decode("utf-8"),
        )

    def _upload_inline(self):
        from google.colab import output

        output.eval_js(
            'document.getElementById("{input_id}").value = ""'.format(
                input_id=self.input_id
            )
        )
        result = output.eval_js(
            'google.colab._files._uploadFiles("{input_id}", "{output_id}")'.format(
                input_id=self.input_id,
                output_id=self.output_id,
            )
        )
        uploaded_files = collections.defaultdict(bytes)
        while result["action"] != "complete":
            result = output.eval_js(
                'google.colab._files._uploadFilesContinue("{output_id}")'.format(
                    output_id=self.output_id
                )
            )
            if result["action"] == "append":
                uploaded_files[result["file"]] += base64.b64decode(result["data"])

        if not uploaded_files:
            return None
        filename, content = next(iter(uploaded_files.items()))
        save_uploaded_content = getattr(
            self.ui.workflow, "save_uploaded_content", None
        )
        if callable(save_uploaded_content):
            return save_uploaded_content(filename, content)

        safe_name = Path(str(filename).replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Uploaded file must have a valid filename.")
        save_path = Path(self.ui.workflow.upload_dir) / safe_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(bytes(content))
        return str(save_path)

    @property
    def value(self):
        return self.path.value.strip()

    @value.setter
    def value(self, path):
        self.path.value = str(path or "")

    def set_visible(self, visible):
        display = None if visible else "none"
        self.path.layout.display = display
        self.upload_button.layout.display = display
        self.status.layout.display = display
        if not visible:
            self.inline_upload.layout.display = "none"

    def _upload(self, _button):
        self.upload_button.disabled = True
        self.status.value = "Choose one file below, or cancel the upload."
        try:
            if self.inline_upload.value:
                self.inline_upload.layout.display = None
                uploaded_path = self._upload_inline()
            else:
                uploaded_path = self.ui.workflow.maybe_upload_path("", True)
            if uploaded_path is None:
                self.status.value = "Upload canceled."
                return
            self.value = uploaded_path
            self.status.value = f"Uploaded: {Path(uploaded_path).name}"
        except Exception as exc:
            self.status.value = f"<font color='red'>{type(exc).__name__}: {exc}</font>"
        finally:
            self.inline_upload.layout.display = "none"
            self.upload_button.disabled = False


class _ModelArtifactField:
    LOCAL = "local"
    HUGGING_FACE = "huggingface"

    def __init__(self, ui, description, placeholder):
        self.ui = ui
        self.visible = True
        self.downloaded_path = ""
        self.metadata = None
        self.loaded_callback = None
        widgets = ui.widgets
        self.source = widgets.ToggleButtons(
            options=[
                ("Upload / Colab path", self.LOCAL),
                ("ProSSTHub / Hugging Face", self.HUGGING_FACE),
            ],
            value=self.LOCAL,
            description="Model source:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.local = _UploadField(ui, description, placeholder)
        self.repo_id = widgets.Text(
            value="",
            placeholder="ProSSTHub/model-name or compatible owner/model-name",
            description="Repository ID:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.revision = widgets.Text(
            value="",
            placeholder="Optional branch, tag, or commit; default: main",
            description="Revision:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.refresh_button = widgets.Button(
            description="Refresh ProSSTHub models",
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.hub_models = widgets.Dropdown(
            options=[("No model list loaded", "")],
            value="",
            description="ProSSTHub models:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.load_button = widgets.Button(
            description="Load Hub model",
            button_style="info",
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.community_link = widgets.HTML(
            value=(
                f"Browse community models on <a href='{PROSST_HUB_URL}' "
                f"target='_blank'>{PROSST_HUB_NAMESPACE}</a>. For models in "
                f"{PROSST_HUB_NAMESPACE}, the short repository name is enough; "
                "compatible ColabProSST repositories from another namespace "
                "can be entered as owner/model."
            ),
            layout=widgets.Layout(width="100%", max_width=ui.GUIDE_WIDTH),
        )
        self.load_output = widgets.Output(
            layout=widgets.Layout(width="100%", max_width=ui.GUIDE_WIDTH)
        )
        self.items = [
            self.source,
            *self.local.items,
            self.refresh_button,
            self.hub_models,
            self.repo_id,
            self.revision,
            self.load_button,
            self.community_link,
            self.load_output,
        ]
        self.source.observe(self._update, names="value")
        self.repo_id.observe(self._invalidate, names="value")
        self.revision.observe(self._invalidate, names="value")
        self.hub_models.observe(self._select_hub_model, names="value")
        self.refresh_button.on_click(
            lambda _button: ui._start_task(
                self.refresh_button,
                self.load_output,
                self._refresh_hub_models,
            )
        )
        self.load_button.on_click(
            lambda _button: ui._start_task(
                self.load_button,
                self.load_output,
                self._load_community_model,
            )
        )
        self._update()

    @property
    def value(self):
        if self.source.value == self.HUGGING_FACE:
            return self.downloaded_path
        return self.local.value

    @value.setter
    def value(self, path):
        self.local.value = path

    def set_visible(self, visible):
        self.visible = bool(visible)
        self._update()

    def set_local_copy(self, description, placeholder):
        self.local.path.description = description
        self.local.path.placeholder = placeholder

    def on_loaded(self, callback):
        self.loaded_callback = callback

    def _invalidate(self, _change=None):
        self.downloaded_path = ""
        self.metadata = None

    def _update(self, _change=None):
        self.source.layout.display = None if self.visible else "none"
        show_local = self.visible and self.source.value == self.LOCAL
        show_hf = self.visible and self.source.value == self.HUGGING_FACE
        self.local.set_visible(show_local)
        for item in [
            self.refresh_button,
            self.hub_models,
            self.repo_id,
            self.revision,
            self.load_button,
            self.community_link,
            self.load_output,
        ]:
            item.layout.display = None if show_hf else "none"

    def _select_hub_model(self, change):
        if change["new"]:
            self.repo_id.value = change["new"]

    def _refresh_hub_models(self):
        print("Refreshing compatible ProSSTHub models...")
        repo_ids = self.ui.workflow.list_prossthub_models()
        self.hub_models.options = [
            ("Select a ProSSTHub model", ""),
            *[(repo_id.split("/", 1)[1], repo_id) for repo_id in repo_ids],
        ]
        self.hub_models.value = ""
        if repo_ids:
            print(f"Found {len(repo_ids)} compatible model(s).")
        else:
            print(
                "No public ColabProSST-compatible models are currently "
                f"published on {PROSST_HUB_NAMESPACE}. You can still enter a "
                "compatible owner/model repository ID manually."
            )

    def _load_community_model(self):
        if not self.repo_id.value.strip():
            raise ValueError("Enter a Hugging Face repository ID.")
        print("Downloading and inspecting the Hub model...")
        result = self.ui.workflow.download_community_checkpoint(
            repo_id=self.repo_id.value,
            revision=self.revision.value,
        )
        if self.loaded_callback is not None:
            self.loaded_callback(result)
        self.downloaded_path = result["artifact_path"]
        self.metadata = result
        print("loaded artifact:", result["artifact_path"])
        print("repository:", result["repo_id"])
        print("artifact type:", result["artifact_type"])
        print("task:", result["task_type"])
        print("base model:", result["model_path"])


class _StructureInput:
    SEQUENCE = "sequence"
    TOKENS = "tokens"

    def __init__(self, ui):
        self.ui = ui
        self.pair_mode = False
        widgets = ui.widgets
        self.mode = widgets.RadioButtons(
            options=self._mode_options(),
            value=self.SEQUENCE,
            description="Input method:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%", max_width=ui.GUIDE_WIDTH),
        )
        self.hint = widgets.HTML(
            layout=widgets.Layout(
                width="100%",
                max_width=ui.GUIDE_WIDTH,
                overflow="visible",
            )
        )
        self.home_help = widgets.HTML(
            value=(
                "<b>Need more guidance or an example?</b> Return to the "
                "ColabProSST home page for the complete input instructions. "
                "Choose your ProSST model there and click "
                "<b>Download CSV templates</b> for ready-to-use examples."
            ),
            layout=widgets.Layout(
                width="100%",
                max_width=ui.GUIDE_WIDTH,
                overflow="visible",
                margin="6px 0 0 0",
            ),
        )
        self.items = [self.mode, self.hint, self.home_help]
        self.display_items = [
            widgets.VBox(
                self.items,
                layout=widgets.Layout(
                    width="100%",
                    max_width=ui.GUIDE_WIDTH,
                    grid_gap="4px",
                    margin="0 0 18px 0",
                    overflow="visible",
                ),
            )
        ]
        self.mode.observe(self._update, names="value")
        self._update({"new": self.mode.value})

    def _mode_options(self):
        return [
            ("Sequence only - prepare structure automatically", self.SEQUENCE),
            ("Prepared CSV with structure_tokens", self.TOKENS),
        ]

    def set_pair_mode(self, enabled):
        enabled = bool(enabled)
        if enabled == self.pair_mode:
            self._update({"new": self.mode.value})
            return

        self.pair_mode = enabled
        self.mode.options = self._mode_options()
        self._update({"new": self.mode.value})

    @property
    def input_mode(self):
        return self.mode.value

    def validate(self, csv_path, structure_vocab_size=None):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = {
                column.strip().lower() for column in (reader.fieldnames or [])
            }
            rows = [
                {
                    str(column).strip().lower(): value
                    for column, value in row.items()
                }
                for row in reader
            ]
        if not columns or not rows:
            raise ValueError("The uploaded CSV is empty.")

        sequence_columns = (
            ("sequence_1", "sequence_2") if self.pair_mode else ("sequence",)
        )
        missing_sequences = sorted(set(sequence_columns) - columns)
        if missing_sequences:
            raise ValueError(
                "The uploaded CSV is missing required sequence column(s): "
                f"{missing_sequences}."
            )

        if self.mode.value == self.SEQUENCE:
            for row_number, row in enumerate(rows, start=2):
                for column in sequence_columns:
                    normalize_protein_sequence(
                        row.get(column, ""),
                        context=f"row {row_number} {column}",
                    )

        if self.pair_mode and self.mode.value == self.TOKENS:
            required = {"structure_tokens_1", "structure_tokens_2"}
            missing = sorted(required - columns)
            if missing:
                raise ValueError(
                    "Protein-pair token input requires both "
                    "structure_tokens_1 and structure_tokens_2. Missing: "
                    f"{missing}."
                )
        elif self.mode.value == self.TOKENS and "structure_tokens" not in columns:
            raise ValueError(
                "You selected `Prepared CSV with structure_tokens`, but the "
                "uploaded CSV has no structure_tokens column. Upload a prepared "
                "CSV or choose the sequence-only input method."
            )
        if self.mode.value == self.TOKENS:
            if "structure_vocab_size" not in columns:
                raise ValueError(
                    "A prepared token CSV must include structure_vocab_size. "
                    "Use the file downloaded from ColabProSST preparation."
                )
            try:
                vocab_sizes = {
                    int(str(row.get("structure_vocab_size", "")).strip())
                    for row in rows
                }
            except ValueError as exc:
                raise ValueError(
                    "structure_vocab_size must be an integer in every CSV row."
                ) from exc
            if structure_vocab_size is not None and vocab_sizes != {
                int(structure_vocab_size)
            }:
                raise ValueError(
                    "The prepared CSV uses structure_vocab_size="
                    f"{sorted(vocab_sizes)}, but the selected ProSST model uses "
                    f"{structure_vocab_size}. Select the model used to prepare "
                    "this CSV."
                )

    def _update(self, change):
        mode = change["new"]
        if self.pair_mode and mode == self.SEQUENCE:
            self.hint.value = (
                "Upload a CSV with <code>sequence_1</code> and "
                "<code>sequence_2</code>. ColabProSST predicts both structures "
                "and generates matching tokens automatically. Each sequence "
                f"must be at most {ESMFOLD_MAX_RESIDUES} residues."
            )
        elif self.pair_mode and mode == self.TOKENS:
            self.hint.value = (
                "Upload a prepared CSV containing <code>structure_tokens_1</code> "
                "and <code>structure_tokens_2</code>, aligned with "
                "<code>sequence_1</code> and <code>sequence_2</code>. Select the "
                "same ProSST model used to prepare the tokens."
            )
        elif mode == self.SEQUENCE:
            self.hint.value = (
                "Upload a CSV with <code>sequence</code>. ColabProSST predicts "
                "each structure and generates tokens for the selected model "
                f"automatically. Sequences must be at most {ESMFOLD_MAX_RESIDUES} "
                "residues. The prepared CSV can be downloaded after the task."
            )
        else:
            self.hint.value = (
                "Upload a prepared CSV containing <code>sequence</code>, "
                "<code>structure_tokens</code>, and "
                "<code>structure_vocab_size</code>. Select the same ProSST "
                "model used to prepare the tokens."
            )


class ColabProSSTUI:
    """ColabSaprot-style interactive interface backed by ColabProSSTWorkflow."""

    WIDTH = "500px"
    HEIGHT = "30px"
    GUIDE_WIDTH = "720px"

    def __init__(self, workflow):
        _validate_colab_environment()
        try:
            import ipywidgets
            from IPython.display import Image, clear_output, display
        except Exception as exc:
            raise RuntimeError(
                "ColabProSSTUI requires ipywidgets and an IPython notebook runtime."
            ) from exc

        self.widgets = ipywidgets
        self.display = display
        self.Image = Image
        self.clear_output = clear_output
        self.workflow = workflow
        self.current_page = None
        self.navigation_history = []
        self.active_thread = None
        self.latest_checkpoint = ""
        self.latest_model_path = DEFAULT_PROSST_MODEL.model_path
        self.latest_task_type = "classification"
        self.latest_num_labels = 2
        self._polling = False
        self._task_lock = threading.Lock()
        self._build_system_widgets()

    def _html(self, value, **layout_kwargs):
        return self.widgets.HTML(
            value=value,
            layout=self.widgets.Layout(**layout_kwargs),
        )

    def _heading(self, text, level=2):
        margin = "0 0 14px 0" if level == 2 else "18px 0 8px 0"
        return self._html(
            f"<h{level} style='margin:0'>{text}</h{level}>",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            margin=margin,
            overflow="visible",
        )

    def _separator(self):
        return self._html(
            "<hr style='border:0;border-top:1px solid #dadce0;margin:0'>",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            margin="18px 0 12px 0",
        )

    def _widget_stack(self, *items):
        return self.widgets.VBox(
            list(items),
            layout=self.widgets.Layout(
                width="100%",
                max_width=self.GUIDE_WIDTH,
                overflow="visible",
            ),
        )

    def _display_page(self, *items):
        self.display(self._widget_stack(*items))

    def _button(self, description, width=None, style=""):
        return self.widgets.Button(
            description=description,
            button_style=style,
            layout=self.widgets.Layout(
                width=width or self.WIDTH,
                height=self.HEIGHT,
            ),
        )

    def _result_downloads(self, files):
        buttons = []
        for label, path in files:
            if not path or not Path(path).is_file():
                continue
            button = self._button(
                f"Download {label}",
                width="320px",
                style="success",
            )
            button.tooltip = str(path)

            def queue_result(_button, download_path=str(path)):
                self.system_status.clear_output(wait=True)
                try:
                    self.workflow.queue_download(download_path)
                    with self.system_status:
                        print(f"Preparing download: {Path(download_path).name}")
                except Exception as exc:
                    with self.system_status:
                        print(
                            f"Download failed for {download_path}: "
                            f"{type(exc).__name__}: {exc}"
                        )

            button.on_click(queue_result)
            buttons.append(button)

        if not buttons:
            return None
        return self._widget_stack(
            self._heading("Download results", level=3),
            *buttons,
        )

    def _display_result_downloads(self, *files):
        downloads = self._result_downloads(files)
        if downloads is not None:
            self.display(downloads)

    def _model_dropdown(self, value=None):
        selected = value or self.latest_model_path
        if selected not in {spec.model_path for spec in PROSST_MODEL_SPECS}:
            selected = DEFAULT_PROSST_MODEL.model_path
        return self.widgets.Dropdown(
            options=[
                (spec.display_name, spec.model_path)
                for spec in PROSST_MODEL_SPECS
            ],
            value=selected,
            description="Base model:",
            layout=self.widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )

    def _apply_artifact_metadata(
        self,
        metadata,
        task_widget,
        model_widget,
        num_labels_widget,
        training_method_widget=None,
    ):
        task_values = {value for _label, value in task_widget.options}
        model_values = {value for _label, value in model_widget.options}
        if metadata["task_type"] not in task_values:
            raise ValueError(
                f"Unsupported artifact task: {metadata['task_type']}."
            )
        if metadata["model_path"] not in model_values:
            raise ValueError(
                "This ColabProSST interface supports community artifacts based "
                "on the official ProSST family only; unsupported base model: "
                f"{metadata['model_path']}."
            )
        task_widget.value = metadata["task_type"]
        model_widget.value = metadata["model_path"]
        if metadata.get("num_labels") is not None:
            num_labels_widget.value = int(metadata["num_labels"])
        if training_method_widget is not None:
            training_method_widget.value = (
                "lora" if metadata["artifact_type"] == "lora" else "full"
            )
        self.latest_checkpoint = metadata["artifact_path"]
        self.latest_model_path = metadata["model_path"]
        self.latest_task_type = metadata["task_type"]
        if metadata.get("num_labels") is not None:
            self.latest_num_labels = int(metadata["num_labels"])

    def _task_dropdown(self, value="classification"):
        return self.widgets.Dropdown(
            options=[
                ("Protein-level Classification", "classification"),
                ("Protein-level Regression", "regression"),
                ("Residue-level Classification", "token_classification"),
                ("Protein-pair Classification", "pair_classification"),
                ("Protein-pair Regression", "pair_regression"),
            ],
            value=value,
            description="Task type:",
            layout=self.widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )

    @staticmethod
    def _task_intro(task_type):
        if task_type == "classification":
            return (
                "<font color='red'>What is <b>Protein-level Classification:</b> "
                "Given a protein, you have some categories and you want to "
                "predict which category the protein belongs to.</font>"
            )
        if task_type == "token_classification":
            return (
                "<font color='red'>What is <b>Residue-level "
                "Classification:</b> Given a protein, predict one category for "
                "each amino-acid residue, such as a binding-site or secondary-"
                "structure label.</font>"
            )
        if task_type == "pair_classification":
            return (
                "<font color='red'>What is <b>Protein-pair "
                "Classification:</b> Given two proteins, predict a category "
                "for their relationship, such as whether they interact.</font>"
            )
        if task_type == "pair_regression":
            return (
                "<font color='red'>What is <b>Protein-pair Regression:</b> "
                "Given two proteins, predict a continuous score for their "
                "relationship, such as interaction strength or affinity.</font>"
            )
        return (
            "<font color='red'>What is <b>Protein-level Regression:</b> Given a "
            "protein, you want to predict a score about its property such as "
            "stability or enzyme activity.</font>"
        )

    @staticmethod
    def _uses_category_count(task_type):
        return task_type in {
            "classification",
            "token_classification",
            "pair_classification",
        }

    @staticmethod
    def _is_pair_task(task_type):
        return task_type in {"pair_classification", "pair_regression"}

    @staticmethod
    def _training_dataset_help(task_type):
        if task_type == "token_classification":
            return (
                "The CSV must contain <code>sequence</code>, "
                "<code>residue_labels</code>, and <code>stage</code>. "
                "<code>residue_labels</code> must contain one integer category "
                "per residue; use <code>-100</code> only to ignore an unlabeled "
                "residue. Then choose one of the two input methods below."
            )
        if ColabProSSTUI._is_pair_task(task_type):
            return (
                "The CSV must contain <code>sequence_1</code>, "
                "<code>sequence_2</code>, <code>label</code>, and "
                "<code>stage</code>. ColabProSST can prepare both structures "
                "automatically, or use tokens already stored in the CSV."
            )
        return (
            "The CSV must contain <code>sequence</code>, <code>label</code>, "
            "and <code>stage</code>. Then choose one of the two input methods below."
        )

    @staticmethod
    def _prediction_output_help(task_type):
        if task_type == "token_classification":
            return (
                "The prediction CSV contains <code>predicted_labels</code>, "
                "<code>confidence</code>, and one <code>prob_*</code> column per "
                "category. Each cell is a space-separated list aligned "
                "one-to-one with the input residues."
            )
        if task_type in {"classification", "pair_classification"}:
            return (
                "The prediction CSV contains the predicted category in "
                "<code>pred</code> and one probability column per category."
            )
        return "The prediction CSV contains the predicted value in <code>pred</code>."

    def _num_labels(self):
        return self.widgets.BoundedIntText(
            value=2,
            min=2,
            max=100000000,
            step=1,
            description="Number of categories:",
            style={"description_width": "initial"},
            layout=self.widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )

    def _output(self):
        return self.widgets.Output(
            layout=self.widgets.Layout(width="100%", border="0")
        )

    def _input_guide(self):
        return self._html(
            "<h3>Important: Choose one input method</h3>"
            "<p>ProSST requires both an amino-acid sequence and matching "
            "ProSST structure tokens as input. Therefore, every protein must "
            "be prepared using one of the two methods below.</p>"
            "<ol>"
            "<li><b>Sequence only:</b> upload a CSV containing amino-acid "
            "sequences. ColabProSST predicts structures and generates tokens "
            "automatically. This uses the public ESMFold service, sends each "
            f"sequence to that service, and supports up to {ESMFOLD_MAX_RESIDUES} "
            "residues per sequence. You can download the prepared token CSV "
            "after the task.</li>"
            "<li><b>Prepared token CSV:</b> first open <b>Prediction &gt; Prepare "
            "reusable structure-token CSV</b>, generate and download the CSV, "
            "or download the prepared input CSV after any sequence-only task. "
            "Upload that file to later tasks. Select the same ProSST model "
            "used during preparation. This avoids repeating structure "
            "prediction and is recommended for repeated work.</li>"
            "</ol>"
            "<p>Protein-pair tasks use the same two methods with "
            "<code>sequence_1</code> and <code>sequence_2</code>.</p>"
            "<hr style='border:0;border-top:1px solid #dadce0;margin:18px 0 12px'>"
            "<h3 style='margin:0 0 8px'>CSV templates</h3>"
            "<p><b>Start here:</b> choose your intended ProSST model below, "
            "then download ready-to-use examples for every supported task. "
            "Prepared-token templates contain the matching "
            "<code>structure_vocab_size</code>; sequence-only templates can "
            "be used directly.</p>",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )

    def _build_system_widgets(self):
        self.back_button = self._button("Go back", width="120px", style="success")
        self.refresh_button = self._button("Refresh", width="120px", style="success")
        self.stop_button = self._button("Stop", width="120px", style="danger")
        self.system_status = self.widgets.Output(
            layout=self.widgets.Layout(width=self.WIDTH)
        )

        self.back_button.on_click(lambda _button: self._go_back())
        self.refresh_button.on_click(lambda _button: self._refresh_page())
        self.stop_button.on_click(lambda _button: self.stop_task())

        self.system_widgets = [
            self._html(
                "<b><font color='red'>Note: At any time you can use the buttons "
                "below to stop and restart.</font></b>"
            ),
            self.widgets.HBox(
                [self.back_button, self.refresh_button, self.stop_button]
            ),
            self._html(
                "<b>Go back:</b> stop the running task and return to the previous "
                "interface.<br><b>Refresh:</b> stop the running task and reset the "
                "current interface.<br><b>Stop:</b> stop the running task."
            ),
            self.system_status,
        ]

    @staticmethod
    def _download_javascript(comm_id, filename, size):
        request_key = f"colabprosst-download:{comm_id}"
        return f"""
        (async () => {{
          const requestKey = {json.dumps(request_key)};
          window.__colabProSSTDownloadRequests =
              window.__colabProSSTDownloadRequests || new Set();
          let alreadyRequested =
              window.__colabProSSTDownloadRequests.has(requestKey);
          try {{
            alreadyRequested = alreadyRequested ||
                sessionStorage.getItem(requestKey) === '1';
          }} catch (error) {{}}
          if (alreadyRequested || !google.colab.kernel.accessAllowed) return;

          window.__colabProSSTDownloadRequests.add(requestKey);
          try {{ sessionStorage.setItem(requestKey, '1'); }} catch (error) {{}}

          const progressBox = document.createElement('div');
          const label = document.createElement('label');
          label.textContent = `Downloading ${{{json.dumps(filename)}}}: `;
          progressBox.appendChild(label);
          const progress = document.createElement('progress');
          progress.max = {int(size)};
          progressBox.appendChild(progress);
          document.body.appendChild(progressBox);

          try {{
            const buffers = [];
            let downloaded = 0;
            const channel = await google.colab.kernel.comms.open(
                {json.dumps(comm_id)});
            channel.send({{}});
            for await (const message of channel.messages) {{
              channel.send({{}});
              if (!message.buffers) continue;
              for (const buffer of message.buffers) {{
                buffers.push(buffer);
                downloaded += buffer.byteLength;
                progress.value = downloaded;
              }}
            }}
            const blob = new Blob(buffers, {{type: 'application/binary'}});
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = {json.dumps(filename)};
            progressBox.appendChild(link);
            link.click();
            setTimeout(() => URL.revokeObjectURL(link.href), 1000);
          }} catch (error) {{
            window.__colabProSSTDownloadRequests.delete(requestKey);
            try {{ sessionStorage.removeItem(requestKey); }} catch (_) {{}}
            throw error;
          }} finally {{
            progressBox.remove();
          }}
        }})();
        """

    def _download_file_once(self, path):
        from google.colab import output as colab_output
        from IPython import get_ipython

        download_path = Path(path)
        if not download_path.is_file():
            raise FileNotFoundError(f"Download file does not exist: {download_path}")

        shell = get_ipython()
        if shell is None or getattr(shell, "kernel", None) is None:
            raise RuntimeError("A live IPython kernel is required for downloads.")
        comm_manager = shell.kernel.comm_manager
        comm_id = f"colabprosst_download_{uuid.uuid4()}"

        def send_file(comm, _open_message):
            handle = download_path.open("rb")

            def send_chunk(_message):
                chunk = handle.read(1024 * 1024)
                if chunk:
                    comm.send({}, None, [chunk])
                    return
                comm.close()
                handle.close()
                comm_manager.unregister_target(comm_id, send_file)

            comm.on_msg(send_chunk)

        comm_manager.register_target(comm_id, send_file)
        script = self._download_javascript(
            comm_id,
            download_path.name,
            download_path.stat().st_size,
        )
        try:
            # The UI cell keeps running to poll widget events, so Javascript
            # display records may be delayed or ignored by Colab. Evaluate the
            # request directly in the active cell output frame instead.
            colab_output.eval_js(script, ignore_result=True)
        except Exception:
            comm_manager.unregister_target(comm_id, send_file)
            raise

    def _update_navigation_controls(self):
        self.back_button.disabled = not self.navigation_history

    @staticmethod
    def _enable_adaptive_colab_height():
        try:
            from google.colab import output as colab_output

            colab_output.no_vertical_scroll()
        except (ImportError, AttributeError):
            # Local Jupyter runtimes do not provide Colab's output-frame API.
            pass

    def _navigate(self, page, remember=True):
        if page is None:
            return
        previous_page = self.current_page
        self.stop_task(silent=True)
        if remember and previous_page is not None and previous_page != page:
            self.navigation_history.append(previous_page)
        self.current_page = page
        self.clear_output(wait=True)
        page()
        self._update_navigation_controls()
        self.display(self._widget_stack(*self.system_widgets))
        self._enable_adaptive_colab_height()

    def _go_back(self):
        if not self.navigation_history:
            self._update_navigation_controls()
            return
        previous_page = self.navigation_history.pop()
        self._navigate(previous_page, remember=False)

    def _refresh_page(self):
        self._navigate(self.current_page, remember=False)

    def _start_task(self, button, output, action):
        def runner():
            output.clear_output(wait=True)
            try:
                with output:
                    try:
                        action()
                    except SystemExit:
                        print("Task interrupted by user.")
                    except Exception as exc:
                        print(f"{type(exc).__name__}: {exc}")
                        traceback.print_exc()
            finally:
                button.disabled = False
                with self._task_lock:
                    if self.active_thread is threading.current_thread():
                        self.active_thread = None

        with self._task_lock:
            if self.active_thread is not None:
                thread = None
            else:
                button.disabled = True
                thread = threading.Thread(target=runner, daemon=True)
                self.active_thread = thread

        if thread is None:
            with output:
                print("A task is already running. Stop it before starting another one.")
            return

        try:
            thread.start()
        except Exception:
            with self._task_lock:
                if self.active_thread is thread:
                    self.active_thread = None
            button.disabled = False
            raise

    def _process_pending_download(self):
        with self._task_lock:
            if self.active_thread is not None:
                return

        pop_download = getattr(self.workflow, "pop_pending_download", None)
        if pop_download is None:
            return

        path = pop_download()
        if path is None:
            return

        try:
            import google.colab  # noqa: F401

            self._download_file_once(path)
            self.system_status.clear_output(wait=True)
            with self.system_status:
                print(
                    f'Download started: {Path(path).name}. '
                    'Check your browser downloads.'
                )
        except Exception as exc:
            self.system_status.clear_output(wait=True)
            with self.system_status:
                print(f"Download failed for {path}: {type(exc).__name__}: {exc}")

    def stop_task(self, silent=False):
        thread = self.active_thread
        if thread is None or not thread.is_alive():
            self.active_thread = None
            if not silent:
                self.system_status.clear_output(wait=True)
                with self.system_status:
                    print("No running task to be stopped.")
            return

        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread.ident), ctypes.py_object(SystemExit)
        )
        if result == 0:
            raise RuntimeError("The running task thread no longer exists.")
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), None
            )
            raise RuntimeError("Could not stop the running task safely.")
        thread.join(timeout=2)
        if thread.is_alive():
            self.active_thread = thread
            if not silent:
                self.system_status.clear_output(wait=True)
                with self.system_status:
                    print(
                        "Stop requested. The task is finishing its current native "
                        "operation; wait before starting another task."
                    )
            return
        self.active_thread = None
        if not silent:
            self.system_status.clear_output(wait=True)
            with self.system_status:
                print("Task interrupted by user.")

    def _download_templates(self, button, model_path=None):
        output = self.system_status

        def action():
            spec = get_prosst_model_spec(
                model_path or self.latest_model_path
            )
            self.workflow.create_csv_templates(
                download=True,
                structure_vocab_size=spec.structure_vocab_size,
            )

        self._start_task(button, output, action)

    def _home_page(self):
        self.current_page = self._home_page
        title = self._heading(
            "Please choose what you want to do with ColabProSST"
        )
        input_guide = self._input_guide()
        template_model = self._model_dropdown()
        template_model.description = "Template model:"
        template_model.style = {"description_width": "initial"}
        template_button = self._button(
            "Download CSV templates", width="280px", style="info"
        )
        train_button = self._button(
            "I want to train my own model", width="400px"
        )
        predict_button = self._button(
            "I want to use existing models to make prediction", width="400px"
        )
        share_button = self._button(
            "I want to share my model publicly", width="400px"
        )

        train_button.on_click(lambda _button: self._navigate(self._training_page))
        predict_button.on_click(
            lambda _button: self._navigate(self._prediction_menu_page)
        )
        share_button.on_click(lambda _button: self._navigate(self._share_page))
        template_button.on_click(
            lambda button: self._download_templates(button, template_model.value)
        )

        self._display_page(
            title,
            train_button,
            predict_button,
            share_button,
            self._separator(),
            input_guide,
            template_model,
            template_button,
        )

    def _training_page(self):
        self.current_page = self._training_page
        widgets = self.widgets
        task_name = widgets.Text(
            value="ProSSTUserTask",
            description="Name your task:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        task_type = self._task_dropdown()
        num_labels = self._num_labels()
        task_intro = self._html(
            self._task_intro("classification"), width=self.WIDTH
        )
        model = self._model_dropdown()
        training_method = widgets.ToggleButtons(
            options=[
                ("Standard / full checkpoint", "full"),
                ("LoRA / PEFT adapter", "lora"),
            ],
            value="full",
            description="Training method:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        lora_rank = widgets.BoundedIntText(
            value=8,
            min=1,
            max=256,
            step=1,
            description="LoRA rank:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, display="none"),
        )
        lora_alpha = widgets.BoundedIntText(
            value=16,
            min=1,
            max=1024,
            step=1,
            description="LoRA alpha:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, display="none"),
        )
        lora_dropout = widgets.FloatSlider(
            value=0.05,
            min=0.0,
            max=0.5,
            step=0.01,
            readout_format=".2f",
            description="LoRA dropout:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, display="none"),
        )
        lora_help = self._html(
            "LoRA trains low-rank adapters in ProSST attention/feed-forward "
            "layers together with the task head. The output is a compact PEFT "
            "adapter ZIP. When continuing an adapter, its saved r/alpha/dropout "
            "configuration is reused.",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )
        training_start = widgets.ToggleButtons(
            options=[
                ("Fresh official model", "fresh"),
                ("Continue from checkpoint", "checkpoint"),
            ],
            value="fresh",
            description="Training start:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        initial_checkpoint = _ModelArtifactField(
            self,
            "Initial checkpoint:",
            "Path to a compatible ColabProSST .pt checkpoint",
        )
        initial_checkpoint.value = self.latest_checkpoint
        resume_optimizer_state = widgets.Checkbox(
            value=False,
            description="Restore optimizer and scheduler (exact resume)",
            style={"description_width": "initial"},
            layout=widgets.Layout(display="none"),
        )
        full_checkpoint_help = (
            "<b>Weight-only fine-tuning:</b> leave exact resume unchecked to "
            "load model weights and start a new optimizer. The checkpoint must "
            "use the same task, base model, structure vocabulary, and category "
            "count.<br><b>Exact resume:</b> also restores optimizer, scheduler, "
            "epoch, and best validation value; it requires a checkpoint saved "
            "with training state. The Epoch setting specifies how many "
            "additional epochs to run."
        )
        lora_checkpoint_help = (
            "Upload a ColabProSST LoRA adapter ZIP or enter an extracted adapter "
            "directory. Adapter weights are continued with a new optimizer; "
            "the task, base model, structure vocabulary, and category count "
            "must match."
        )
        checkpoint_help = self._html(
            full_checkpoint_help,
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )
        csv_input = _UploadField(
            self,
            "Training CSV:",
            "CSV with sequence, label, and stage; tokens are optional",
        )
        dataset_help = self._html(
            self._training_dataset_help("classification"),
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )
        structure_input = _StructureInput(self)
        batch_size = widgets.Dropdown(
            options=[1, 2, 4, 8, 16, 32],
            value=1,
            description="Batch size:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        epochs = widgets.BoundedIntText(
            value=2,
            min=1,
            max=100,
            description="Epoch:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        learning_rate = widgets.FloatText(
            value=2.0e-5,
            description="Learning rate:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        advanced_button = self._button("Show advanced settings")
        freeze_backbone = widgets.Checkbox(
            value=True,
            description="Freeze ProSST backbone",
            style={"description_width": "initial"},
            layout=widgets.Layout(display="none"),
        )
        gradient_checkpointing = widgets.Checkbox(
            value=True,
            description="Use gradient checkpointing",
            style={"description_width": "initial"},
            layout=widgets.Layout(display="none"),
        )
        save_training_state = widgets.Checkbox(
            value=False,
            description="Save optimizer state for future exact resume (larger file)",
            style={"description_width": "initial"},
            layout=widgets.Layout(display="none"),
        )
        start_button = self._button("Start training", style="info")
        output = self._output()
        finish_hint = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )

        def update_task(change):
            selected_task = change["new"]
            num_labels.layout.display = (
                None if self._uses_category_count(selected_task) else "none"
            )
            task_intro.value = self._task_intro(selected_task)
            dataset_help.value = self._training_dataset_help(selected_task)
            structure_input.set_pair_mode(self._is_pair_task(selected_task))
            if selected_task == "token_classification":
                placeholder = (
                    "CSV with sequence, residue_labels, and stage"
                )
            elif self._is_pair_task(selected_task):
                placeholder = (
                    "CSV with sequence_1, sequence_2, label, and stage"
                )
            else:
                placeholder = "CSV with sequence, label, and stage"
            csv_input.path.placeholder = placeholder

        def toggle_advanced(_button):
            show = advanced_button.description == "Show advanced settings"
            mode = None if show else "none"
            use_lora = training_method.value == "lora"
            freeze_backbone.layout.display = "none" if use_lora else mode
            gradient_checkpointing.layout.display = mode
            save_training_state.layout.display = "none" if use_lora else mode
            advanced_button.description = (
                "Hide advanced settings" if show else "Show advanced settings"
            )

        def update_training_controls(_change=None):
            use_lora = training_method.value == "lora"
            use_checkpoint = training_start.value == "checkpoint"
            initial_checkpoint.set_visible(use_checkpoint)
            resume_optimizer_state.layout.display = (
                None if use_checkpoint and not use_lora else "none"
            )
            checkpoint_help.layout.display = None if use_checkpoint else "none"
            checkpoint_help.value = (
                lora_checkpoint_help if use_lora else full_checkpoint_help
            )
            initial_checkpoint.set_local_copy(
                "Initial adapter:" if use_lora else "Initial checkpoint:",
                "Path to a LoRA adapter directory or ZIP"
                if use_lora
                else "Path to a compatible ColabProSST .pt checkpoint",
            )
            show_new_lora_config = use_lora and not use_checkpoint
            for item in [lora_rank, lora_alpha, lora_dropout]:
                item.layout.display = None if show_new_lora_config else "none"
            lora_help.layout.display = None if use_lora else "none"
            if use_lora:
                freeze_backbone.value = False
                save_training_state.value = False
                resume_optimizer_state.value = False
            advanced_visible = advanced_button.description == "Hide advanced settings"
            freeze_backbone.layout.display = (
                None if advanced_visible and not use_lora else "none"
            )
            save_training_state.layout.display = (
                None if advanced_visible and not use_lora else "none"
            )
            if not use_checkpoint:
                resume_optimizer_state.value = False

        def apply_initial_artifact(metadata):
            self._apply_artifact_metadata(
                metadata,
                task_type,
                model,
                num_labels,
                training_method,
            )
            training_start.value = "checkpoint"

        def train():
            if not csv_input.value:
                raise ValueError("Upload a training CSV or enter its path.")
            use_checkpoint = training_start.value == "checkpoint"
            if use_checkpoint and not initial_checkpoint.value:
                raise ValueError(
                    "Upload an initial checkpoint or enter its path."
                )
            clean_task_name = task_name.value.strip()
            if not clean_task_name or Path(clean_task_name).name != clean_task_name:
                raise ValueError(
                    "Task name must be a non-empty file-name-safe value."
                )
            model_spec = get_prosst_model_spec(model.value)
            structure_input.validate(
                csv_input.value,
                structure_vocab_size=model_spec.structure_vocab_size,
            )
            print("Start training...")
            if use_checkpoint:
                print("Selected checkpoint:", initial_checkpoint.value)
                print(
                    "Continuation mode:",
                    "exact resume"
                    if resume_optimizer_state.value
                    else "weight-only fine-tuning",
                )
            result = self.workflow.train_downstream(
                task_type=task_type.value,
                input_csv=csv_input.value,
                input_mode=structure_input.input_mode,
                task_name=clean_task_name,
                num_labels=num_labels.value,
                max_epochs=epochs.value,
                batch_size=batch_size.value,
                learning_rate=learning_rate.value,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                freeze_backbone=freeze_backbone.value,
                gradient_checkpointing=gradient_checkpointing.value,
                initial_checkpoint=(
                    initial_checkpoint.value if use_checkpoint else ""
                ),
                resume_optimizer_state=(
                    resume_optimizer_state.value if use_checkpoint else False
                ),
                save_training_state=save_training_state.value,
                training_method=training_method.value,
                lora_rank=lora_rank.value,
                lora_alpha=lora_alpha.value,
                lora_dropout=lora_dropout.value,
                download=False,
            )
            self.latest_checkpoint = result["checkpoint_path"]
            self.latest_model_path = result["model_path"]
            self.latest_task_type = result["task_type"]
            self.latest_num_labels = num_labels.value
            print("Training finished.")
            print("Model artifact:", result["checkpoint_path"])
            print("Download artifact:", result["checkpoint_download_path"])
            print("Test predictions:", result["test_result_csv"])
            if result["training_method"] == "full":
                print(
                    "Checkpoint training state:",
                    "included"
                    if result["save_training_state"]
                    else "weights only",
                )
            artifact_label = (
                "LoRA adapter ZIP"
                if result["training_method"] == "lora"
                else "model checkpoint"
            )
            self._display_result_downloads(
                (artifact_label, result["checkpoint_download_path"]),
                ("test predictions CSV", result["test_result_csv"]),
                ("prepared input CSV", result.get("prepared_input_csv")),
            )
            finish_hint.value = (
                "<h3>The training is completed. You can then:</h3>"
                "<ul>"
                "<li><b>Train again:</b> click <b>Refresh</b>, adjust the "
                "settings, and start a new task.</li>"
                "<li><b>Use this model for prediction:</b> click <b>Go back</b>, "
                "choose <b>I want to use existing models to make prediction</b>, "
                "then choose <b>Protein property prediction</b>. The checkpoint "
                "is selected automatically in this session.</li>"
                "<li><b>Share this model:</b> click <b>Go back</b> and choose "
                "<b>I want to share my model publicly</b>.</li>"
                "</ul>"
            )
            finish_hint.layout.display = None

        update_task({"new": task_type.value})
        task_type.observe(update_task, names="value")
        initial_checkpoint.on_loaded(apply_initial_artifact)
        update_training_controls()
        training_start.observe(update_training_controls, names="value")
        training_method.observe(update_training_controls, names="value")
        advanced_button.on_click(toggle_advanced)
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, train)
        )

        self._display_page(
            self._heading("Please finish the setting of your training task"),
            self._heading("Task setting:", level=3),
            task_name,
            task_type,
            num_labels,
            task_intro,
            self._heading("Model setting:", level=3),
            model,
            training_method,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_help,
            training_start,
            *initial_checkpoint.items,
            resume_optimizer_state,
            checkpoint_help,
            self._heading("Dataset setting:", level=3),
            dataset_help,
            *csv_input.items,
            *structure_input.display_items,
            self._heading("Training hyper-parameters:", level=3),
            batch_size,
            epochs,
            learning_rate,
            advanced_button,
            freeze_backbone,
            gradient_checkpointing,
            save_training_state,
            self._separator(),
            start_button,
            output,
            finish_hint,
        )

    def _prediction_menu_page(self):
        self.current_page = self._prediction_menu_page
        property_button = self._button(
            "Protein property prediction", style="info"
        )
        mutation_button = self._button(
            "Mutational effect prediction", style="info"
        )
        saturation_button = self._button(
            "Single-site saturation mutagenesis", style="info"
        )
        embedding_button = self._button(
            "Extract protein embeddings", style="info"
        )
        structure_button = self._button(
            "Prepare reusable structure-token CSV", style="info"
        )
        property_button.on_click(
            lambda _button: self._navigate(self._property_prediction_page)
        )
        mutation_button.on_click(
            lambda _button: self._navigate(self._mutation_page)
        )
        saturation_button.on_click(
            lambda _button: self._navigate(self._saturation_page)
        )
        embedding_button.on_click(
            lambda _button: self._navigate(self._embedding_page)
        )
        structure_button.on_click(
            lambda _button: self._navigate(self._structure_page)
        )
        self._display_page(
            self._heading(
                "ColabProSST supports multiple prediction tasks, which one "
                "would you like to choose?"
            ),
            self._separator(),
            property_button,
            self._html(
                "Use a trained ProSST checkpoint for protein-level "
                "classification/regression, residue-level classification, or "
                "protein-pair classification/regression."
            ),
            self._separator(),
            embedding_button,
            self._html(
                "Extract final-layer protein-level vectors, residue-level "
                "vectors, or both from an official ProSST model."
            ),
            self._separator(),
            mutation_button,
            self._html(
                "Use official ProSST masked-language-model scores for zero-shot "
                "single-site or multi-site mutation effects."
            ),
            self._separator(),
            saturation_button,
            self._html(
                "Score all 20 amino acids at every position and generate a "
                "20-by-length matrix plus a zero-centered heatmap."
            ),
            self._separator(),
            structure_button,
            self._html(
                "Convert one PDB/mmCIF structure into a durable CSV containing "
                "its sequence and ProSST structure tokens."
            ),
        )

    def _property_prediction_page(self):
        self.current_page = self._property_prediction_page
        widgets = self.widgets
        task_type = self._task_dropdown(self.latest_task_type)
        num_labels = self._num_labels()
        num_labels.value = self.latest_num_labels
        num_labels.layout.display = (
            None
            if self._uses_category_count(self.latest_task_type)
            else "none"
        )
        task_intro = self._html(
            self._task_intro(self.latest_task_type), width=self.WIDTH
        )
        prediction_help = self._html(
            self._prediction_output_help(self.latest_task_type),
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )
        model = self._model_dropdown()
        checkpoint = _ModelArtifactField(
            self,
            "Model or adapter:",
            "Path to a .pt checkpoint, LoRA adapter directory, or LoRA ZIP",
        )
        checkpoint.value = self.latest_checkpoint
        csv_input = _UploadField(
            self,
            "Prediction CSV:",
            "CSV with sequence; prepared structure tokens are optional",
        )
        structure_input = _StructureInput(self)
        structure_input.set_pair_mode(
            self._is_pair_task(self.latest_task_type)
        )
        batch_size = widgets.Dropdown(
            options=[1, 2, 4, 8, 16, 32],
            value=1,
            description="Batch size:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        start_button = self._button("Start prediction", style="info")
        output = self._output()

        def update_task(change):
            selected_task = change["new"]
            num_labels.layout.display = (
                None if self._uses_category_count(selected_task) else "none"
            )
            task_intro.value = self._task_intro(selected_task)
            prediction_help.value = self._prediction_output_help(selected_task)
            structure_input.set_pair_mode(self._is_pair_task(selected_task))
            csv_input.path.placeholder = (
                "CSV with sequence_1 and sequence_2"
                if self._is_pair_task(selected_task)
                else "CSV with sequence"
            )

        def predict():
            if not checkpoint.value:
                raise ValueError(
                    "Upload a model checkpoint/adapter or enter its path."
                )
            if not csv_input.value:
                raise ValueError("Upload a prediction CSV or enter its path.")
            model_spec = get_prosst_model_spec(model.value)
            structure_input.validate(
                csv_input.value,
                structure_vocab_size=model_spec.structure_vocab_size,
            )
            print("Start prediction...")
            result = self.workflow.predict_downstream(
                task_type=task_type.value,
                input_csv=csv_input.value,
                checkpoint_path=checkpoint.value,
                input_mode=structure_input.input_mode,
                num_labels=num_labels.value,
                batch_size=batch_size.value,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                download=False,
            )
            self.latest_model_path = model.value
            self.latest_task_type = task_type.value
            self.latest_num_labels = num_labels.value
            self.display(result.head())
            self._display_result_downloads(
                ("predictions CSV", result.attrs.get("output_csv")),
                ("prepared input CSV", result.attrs.get("prepared_input_csv")),
            )

        checkpoint.on_loaded(
            lambda metadata: self._apply_artifact_metadata(
                metadata,
                task_type,
                model,
                num_labels,
            )
        )
        update_task({"new": task_type.value})
        task_type.observe(update_task, names="value")
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, predict)
        )
        self._display_page(
            self._heading("Protein and protein-pair property prediction"),
            self._heading("Choose the prediction task:", level=3),
            task_type,
            num_labels,
            task_intro,
            prediction_help,
            self._heading("Choose the model for prediction:", level=3),
            model,
            *checkpoint.items,
            self._html(
                "Use a full ColabProSST <code>.pt</code> checkpoint, an "
                "extracted LoRA adapter directory, or a downloaded LoRA "
                "adapter ZIP. Select the checkpoint's original task and base "
                "model."
            ),
            self._heading("Input proteins:", level=3),
            *csv_input.items,
            *structure_input.display_items,
            batch_size,
            self._separator(),
            start_button,
            output,
        )

    def _embedding_page(self):
        self.current_page = self._embedding_page
        widgets = self.widgets
        level = widgets.ToggleButtons(
            options=[
                ("Protein-level", "protein"),
                ("Residue-level", "residue"),
                ("Both", "both"),
            ],
            value="protein",
            description="Embedding level:",
            style={"description_width": "initial"},
            layout=widgets.Layout(
                width="100%",
                max_width=self.GUIDE_WIDTH,
                min_height=self.HEIGHT,
                height="auto",
                overflow="visible",
            ),
        )
        model = self._model_dropdown()
        embedding_model_source = widgets.ToggleButtons(
            options=[
                ("Official ProSST", "official"),
                ("Fine-tuned / community artifact", "artifact"),
            ],
            value="official",
            description="Embedding model:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        embedding_artifact = _ModelArtifactField(
            self,
            "Model or adapter:",
            "Path to a .pt checkpoint, LoRA adapter directory, or LoRA ZIP",
        )
        embedding_artifact.value = self.latest_checkpoint
        embedding_artifact.set_visible(False)
        csv_input = _UploadField(
            self,
            "Protein CSV:",
            "CSV with sequence; prepared structure tokens are optional",
        )
        structure_input = _StructureInput(self)
        batch_size = widgets.Dropdown(
            options=[1, 2, 4, 8, 16, 32],
            value=1,
            description="Batch size:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        max_length = widgets.BoundedIntText(
            value=2046,
            min=1,
            max=2046,
            step=1,
            description="Maximum residues:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        start_button = self._button("Start embedding extraction", style="info")
        output = self._output()
        completion = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )

        def update_embedding_source(change):
            embedding_artifact.set_visible(change["new"] == "artifact")

        def apply_embedding_artifact(metadata):
            model_values = {value for _label, value in model.options}
            if metadata["model_path"] not in model_values:
                raise ValueError(
                    "Embedding artifacts must use an official ProSST family "
                    f"base model; found {metadata['model_path']}."
                )
            model.value = metadata["model_path"]
            self.latest_checkpoint = metadata["artifact_path"]
            self.latest_model_path = metadata["model_path"]

        def extract():
            completion.value = ""
            completion.layout.display = "none"
            if not csv_input.value:
                raise ValueError("Upload a protein CSV or enter its path.")
            use_artifact = embedding_model_source.value == "artifact"
            if use_artifact and not embedding_artifact.value:
                raise ValueError(
                    "Upload a fine-tuned model/adapter, enter its path, or load "
                    "a Hugging Face repository."
                )
            model_spec = get_prosst_model_spec(model.value)
            structure_input.validate(
                csv_input.value,
                structure_vocab_size=model_spec.structure_vocab_size,
            )
            print("Extracting ProSST embeddings...")
            result = self.workflow.extract_embeddings(
                input_csv=csv_input.value,
                input_mode=structure_input.input_mode,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                level=level.value,
                batch_size=batch_size.value,
                max_length=max_length.value,
                checkpoint_path=(
                    embedding_artifact.value if use_artifact else ""
                ),
                download=False,
            )
            self.latest_model_path = model.value
            if "protein_embeddings" in result:
                print(
                    "protein embedding shape:",
                    tuple(result["protein_embeddings"].shape),
                )
            if "residue_embeddings" in result:
                residue_shapes = [
                    tuple(embedding.shape)
                    for embedding in result["residue_embeddings"]
                ]
                print("residue embedding shapes:", residue_shapes)
            print("embedding package:", result["archive_path"])
            self._display_result_downloads(
                ("embedding ZIP", result["archive_path"]),
                ("embeddings PT", result["output_pt"]),
                ("embedding index CSV", result["output_index_csv"]),
                ("prepared input CSV", result.get("prepared_input_csv")),
            )
            completion.value = (
                "<b>Embedding extraction completed.</b><br>"
                f"Package: <code>{result['archive_path']}</code><br>"
                "Use the result buttons to download the package or either "
                "individual file."
            )
            completion.layout.display = None

        embedding_artifact.on_loaded(apply_embedding_artifact)
        embedding_model_source.observe(update_embedding_source, names="value")
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, extract)
        )
        self._display_page(
            self._heading("Extract ProSST embeddings"),
            self._html(
                "The final ProSST hidden layer is used. Protein-level output "
                "has shape <code>[N, D]</code>. Residue-level output is a list "
                "of <code>[L, D]</code> tensors, one per protein. CLS, EOS, and "
                "padding tokens are excluded."
            ),
            self._heading("Embedding setting:", level=3),
            level,
            model,
            embedding_model_source,
            *embedding_artifact.items,
            self._heading("Input proteins:", level=3),
            *csv_input.items,
            *structure_input.display_items,
            batch_size,
            max_length,
            self._separator(),
            start_button,
            output,
            completion,
        )

    def _saturation_page(self):
        self.current_page = self._saturation_page
        model = self._model_dropdown()
        csv_input = _UploadField(
            self,
            "Protein CSV:",
            "One-row CSV with sequence; prepared structure tokens are optional",
        )
        structure_input = _StructureInput(self)
        start_button = self._button(
            "Start saturation mutagenesis",
            style="info",
        )
        output = self._output()

        def predict():
            if not csv_input.value:
                raise ValueError("Upload a protein CSV or enter its path.")
            model_spec = get_prosst_model_spec(model.value)
            structure_input.validate(
                csv_input.value,
                structure_vocab_size=model_spec.structure_vocab_size,
            )
            print("Running single-site saturation mutagenesis...")
            result = self.workflow.run_saturation_mutagenesis(
                input_csv=csv_input.value,
                input_mode=structure_input.input_mode,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                download=False,
            )
            self.latest_model_path = model.value
            print("scored mutations:", len(result["score_table"]))
            print("saturation package:", result["archive_path"])
            self.display(self.Image(filename=result["output_heatmap_png"]))
            self.display(result["score_table"].head())
            self._display_result_downloads(
                ("saturation ZIP", result["archive_path"]),
                ("mutation scores CSV", result["output_csv"]),
                ("score matrix CSV", result["output_matrix_csv"]),
                ("heatmap PNG", result["output_heatmap_png"]),
                ("prepared input CSV", result.get("prepared_input_csv")),
            )

        start_button.on_click(
            lambda _button: self._start_task(start_button, output, predict)
        )
        self._display_page(
            self._heading("Single-site saturation mutagenesis"),
            self._html(
                "Upload a CSV containing <b>exactly one protein row</b>. "
                "ColabProSST scores all 20 amino acids at every position with "
                "the official <code>log P(mutant) - log P(wild type)</code> "
                "formula. The ZIP contains a long score table, a "
                "<code>20 x L</code> matrix, and a PNG heatmap. WT-to-WT "
                "entries are zero."
            ),
            self._heading("Model setting:", level=3),
            model,
            self._heading("Protein input:", level=3),
            *csv_input.items,
            *structure_input.display_items,
            self._separator(),
            start_button,
            output,
        )

    def _mutation_page(self):
        self.current_page = self._mutation_page
        widgets = self.widgets
        model = self._model_dropdown()
        csv_input = _UploadField(
            self,
            "Mutation CSV:",
            "CSV with sequence and mutant; prepared structure tokens are optional",
        )
        structure_input = _StructureInput(self)
        start_button = self._button("Start prediction", style="info")
        output = self._output()

        def predict():
            if not csv_input.value:
                raise ValueError("Upload a mutation CSV or enter its path.")
            model_spec = get_prosst_model_spec(model.value)
            structure_input.validate(
                csv_input.value,
                structure_vocab_size=model_spec.structure_vocab_size,
            )
            print("Start mutational effect prediction...")
            result = self.workflow.run_zero_shot(
                input_csv=csv_input.value,
                input_mode=structure_input.input_mode,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                download=False,
            )
            self.latest_model_path = model.value
            self.display(result.head())
            self._display_result_downloads(
                ("mutation scores CSV", result.attrs.get("output_csv")),
                ("prepared input CSV", result.attrs.get("prepared_input_csv")),
            )

        start_button.on_click(
            lambda _button: self._start_task(start_button, output, predict)
        )
        self._display_page(
            self._heading("Mutational effect prediction"),
            self._heading("Model setting:", level=3),
            model,
            self._html(
                "<b>Zero-shot model note:</b> This task calculates "
                "<code>log P(mutant) - log P(wild type)</code> from an official "
                "pretrained ProSST masked-language model. Protein-level "
                "classification and regression checkpoints do not provide this "
                "mutation score; use them under <b>Protein property "
                "prediction</b> instead.",
                width="100%",
                max_width=self.GUIDE_WIDTH,
                overflow="visible",
            ),
            self._heading("Mutation data:", level=3),
            self._html(
                "The CSV must contain <code>sequence</code> and "
                "<code>mutant</code>. Choose automatic preparation or upload a "
                "prepared token CSV. Zero-shot mutation scoring uses official "
                "ProSST family models, not downstream fine-tuned checkpoints."
            ),
            *csv_input.items,
            *structure_input.display_items,
            self._separator(),
            start_button,
            output,
        )

    def _structure_page(self):
        self.current_page = self._structure_page
        widgets = self.widgets
        model = self._model_dropdown()
        source = widgets.ToggleButtons(
            options=[
                ("Sequence CSV - automatic", "sequence"),
                ("One PDB/mmCIF file", "structure"),
            ],
            value="sequence",
            description="Preparation source:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        source_help = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )
        sequence_csv = _UploadField(
            self,
            "Sequence CSV:",
            "CSV with sequence, or sequence_1 and sequence_2",
        )
        structure = _UploadField(
            self,
            "Structure file:",
            "Path to one PDB or mmCIF file",
        )
        chain = widgets.Text(
            value="",
            placeholder="Leave empty to use all chains",
            description="Chain:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        vocab = widgets.IntText(
            value=get_prosst_model_spec(model.value).structure_vocab_size,
            description="Structure vocab:",
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        start_button = self._button("Convert structure", style="info")
        output = self._output()
        next_steps = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )

        def update_model(change):
            vocab.value = get_prosst_model_spec(
                change["new"]
            ).structure_vocab_size

        def update_source(change):
            use_sequence = change["new"] == "sequence"
            sequence_csv.set_visible(use_sequence)
            structure.set_visible(not use_sequence)
            chain.layout.display = "none" if use_sequence else None
            start_button.description = (
                "Prepare token CSV" if use_sequence else "Convert structure"
            )
            source_help.value = (
                "ColabProSST sends each sequence to the public ESMFold service, "
                f"which accepts at most {ESMFOLD_MAX_RESIDUES} residues per "
                "sequence. Results are cached in this runtime."
                if use_sequence
                else "The uploaded coordinates are quantized directly and are "
                "not sent to the ESMFold service."
            )

        def convert():
            if source.value == "sequence":
                if not sequence_csv.value:
                    raise ValueError("Upload a sequence CSV or enter its path.")
                print("Preparing structures and ProSST tokens from sequences...")
                result = self.workflow.prepare_sequence_input_csv(
                    input_csv=sequence_csv.value,
                    structure_vocab_size=vocab.value,
                    download=False,
                )
            else:
                if not structure.value:
                    raise ValueError("Upload a PDB/mmCIF file or enter its path.")
                print("Converting structure to ProSST tokens...")
                result = self.workflow.convert_structure(
                    structure_path=structure.value,
                    chain_id=chain.value,
                    structure_vocab_size=vocab.value,
                    download=False,
                )
            self.latest_model_path = model.value
            self.display(result.head())
            self._display_result_downloads(
                ("prepared token CSV", result.attrs.get("output_csv")),
            )
            next_steps.value = (
                "<h3>Your reusable input CSV is ready</h3>"
                "<ol>"
                "<li>Download the <b>prepared token CSV</b> above.</li>"
                "<li>Open training or the prediction task you need and upload "
                "that downloaded CSV.</li>"
                f"<li>Keep <b>{get_prosst_model_spec(model.value).display_name}</b> "
                "selected, because tokens are specific to its structure "
                "vocabulary.</li>"
                "<li>Select <b>Prepared CSV with structure_tokens</b>.</li>"
                "</ol>"
                "The downloaded file already contains <code>sequence</code>, "
                "<code>structure_tokens</code>, and "
                "<code>structure_vocab_size</code>, so it can also be reused in "
                "future Colab sessions. Add task-specific columns such as "
                "<code>label</code>, <code>stage</code>, or <code>mutant</code> "
                "when needed."
            )
            next_steps.layout.display = None

        model.observe(update_model, names="value")
        source.observe(update_source, names="value")
        update_source({"new": source.value})
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, convert)
        )
        self._display_page(
            self._heading("Prepare reusable structure-token CSV"),
            self._html(
                "Use a sequence CSV for automatic batch preparation, or use an "
                "existing experimental/predicted PDB or mmCIF structure for one "
                "protein. All original CSV columns are kept in the output."
            ),
            self._heading("Model setting:", level=3),
            model,
            self._heading("Preparation input:", level=3),
            source,
            source_help,
            *sequence_csv.items,
            *structure.items,
            chain,
            vocab,
            self._separator(),
            start_button,
            output,
            next_steps,
        )

    def _share_page(self):
        self.current_page = self._share_page
        widgets = self.widgets
        login_button = widgets.Button(
            description="Log in to Hugging Face",
            button_style="info",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        token_input = widgets.Password(
            value="",
            placeholder="Paste a Hugging Face write token (hf_...)",
            description="Token:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        login_help = self._html(
            "Log in first with a Hugging Face user access token that has write "
            "permission. The model will first be uploaded to your personal "
            "Hugging Face account, so ProSSTHub organization access is not "
            "required. Wait for the login success message before continuing. "
            "The token is handled by Hugging Face and is not stored in this "
            "notebook.",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )
        login_status = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )
        repo_name = widgets.Text(
            value="",
            placeholder="ProSST-2048-Task-Method",
            description="Repository name:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        hub_help = self._html(
            "The repository will be created under the account used in step 1, "
            "for example <code>your-name/ProSST-2048-Stability</code>. Use a "
            "new repository name unless you explicitly intend to update an "
            f"existing model. After upload, you can contribute it to "
            f"<a href='{PROSST_HUB_URL}' target='_blank'>"
            f"{PROSST_HUB_NAMESPACE}</a>.",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )
        checkpoint = _UploadField(
            self,
            "Model or adapter:",
            "Path to a ColabProSST .pt, LoRA directory, or LoRA ZIP",
        )
        checkpoint.value = self.latest_checkpoint
        model = self._model_dropdown()
        task_type = self._task_dropdown(self.latest_task_type)
        num_labels = self._num_labels()
        num_labels.value = self.latest_num_labels
        num_labels.layout.display = (
            None
            if self._uses_category_count(self.latest_task_type)
            else "none"
        )
        allow_update = widgets.Checkbox(
            value=False,
            description="Update my repository if it already exists",
            style={"description_width": "initial"},
        )
        title = widgets.Text(
            value="ColabProSST model",
            description="Model title:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        description = widgets.Textarea(
            value="A ProSST checkpoint trained with ColabProSST.",
            description="Description:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height="90px"),
        )
        start_button = self._button("Upload model", style="info")
        output = self._output()
        contribution_hint = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )

        def update_task(change):
            num_labels.layout.display = (
                None if self._uses_category_count(change["new"]) else "none"
            )

        def log_in(_button):
            from contextlib import redirect_stdout
            from io import StringIO

            from huggingface_hub import HfApi, login

            token = token_input.value.strip()
            token_input.value = ""
            if not token:
                login_status.value = (
                    "<span style='color:#b3261e'>Paste a Hugging Face write "
                    "token before logging in.</span>"
                )
                return

            login_button.disabled = True
            login_status.value = "Checking the token..."
            try:
                with redirect_stdout(StringIO()):
                    login(
                        token=token,
                        add_to_git_credential=False,
                        write_permission=True,
                    )
                account = HfApi().whoami()
                username = str(account.get("name", "")).strip()
                account_text = (
                    f" Logged in as <b>{username}</b>." if username else ""
                )
                login_status.value = (
                    "<span style='color:#137333'><b>Login successful.</b>"
                    f"{account_text} The write token is saved in the Hugging "
                    "Face cache for this Colab runtime.</span>"
                )
            except Exception as exc:
                login_status.value = (
                    "<span style='color:#b3261e'><b>Login failed:</b> "
                    f"{type(exc).__name__}: {exc}</span>"
                )
            finally:
                token = ""
                login_button.disabled = False

        def upload():
            if not checkpoint.value:
                raise ValueError(
                    "Upload a model checkpoint/adapter or enter its path."
                )
            clean_repo_name = repo_name.value.strip()
            if not clean_repo_name:
                raise ValueError("Enter a Hugging Face repository name.")
            repo_id = self.workflow.personal_hf_model_repo_id(clean_repo_name)
            print("Uploading model to your Hugging Face account...")
            print("Target repository:", repo_id)
            model_spec = get_prosst_model_spec(model.value)
            package = self.workflow.upload_checkpoint_to_hf(
                repo_id=repo_id,
                checkpoint_path=checkpoint.value,
                task_type=task_type.value,
                num_labels=num_labels.value,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                private=False,
                run_login=False,
                title=title.value,
                description=description.value,
                allow_update=allow_update.value,
            )
            print("Local model package:", package)
            model_url = f"https://huggingface.co/{repo_id}"
            contribution_hint.value = (
                f"<b>Upload complete:</b> <a href='{model_url}' "
                f"target='_blank'>{repo_id}</a><br>"
                "The model is now in your personal Hugging Face account. To "
                f"contribute it to <a href='{PROSST_HUB_URL}' target='_blank'>"
                f"{PROSST_HUB_NAMESPACE}</a>, send this repository link to the "
                "ProSSTHub maintainers for review and transfer."
            )
            contribution_hint.layout.display = None

        task_type.observe(update_task, names="value")
        login_button.on_click(log_in)
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, upload)
        )
        self._display_page(
            self._heading(
                "Share your ColabProSST model with the ProSSTHub community"
            ),
            self._heading("1. Log in to Hugging Face", level=3),
            login_help,
            token_input,
            login_button,
            login_status,
            self._separator(),
            self._heading("2. Choose the model to share", level=3),
            *checkpoint.items,
            model,
            task_type,
            num_labels,
            self._separator(),
            self._heading("3. Name your Hugging Face repository", level=3),
            hub_help,
            repo_name,
            allow_update,
            self._separator(),
            self._heading("4. Describe your model", level=3),
            title,
            description,
            self._separator(),
            self._heading("5. Upload to Hugging Face", level=3),
            start_button,
            output,
            contribution_hint,
        )

    def launch(self, poll=True):
        """Display the first page and optionally keep Colab widget events alive."""
        self.navigation_history.clear()
        self.current_page = self._home_page
        self._home_page()
        self._update_navigation_controls()
        self.display(self._widget_stack(*self.system_widgets))
        self._enable_adaptive_colab_height()

        if not poll:
            return self

        try:
            from jupyter_ui_poll import ui_events
        except Exception as exc:
            raise RuntimeError(
                "jupyter_ui_poll is required for the live ColabProSST interface."
            ) from exc

        self._polling = True
        try:
            with ui_events() as poll_events:
                while self._polling:
                    poll_events(10)
                    self._process_pending_download()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.system_status.clear_output(wait=True)
            with self.system_status:
                print(
                    "Interface polling stopped. Run this cell again to resume "
                    "interactive controls."
                )
        finally:
            self._polling = False
            self.stop_task(silent=True)


def launch_colabprosst(workflow, poll=True):
    return ColabProSSTUI(workflow).launch(poll=poll)
