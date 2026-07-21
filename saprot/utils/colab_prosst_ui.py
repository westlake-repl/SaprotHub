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
from saprot.utils.colab_prosst_templates import get_input_template_name


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
        result = self.ui.workflow.download_community_adapter(
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
    STRUCTURE = "structure"

    def __init__(self, ui):
        self.ui = ui
        self.pair_mode = False
        self.template_group = None
        self.template_task = None
        self.task_label = "this task"
        self.required_columns = set()
        self.exact_rows = None
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
        self.structure_archive = _UploadField(
            ui,
            "Structure ZIP:",
            "ZIP containing the PDB/mmCIF files named in the CSV",
        )
        self.home_help = widgets.HTML(
            value=(
                "<b>Need more guidance or an example?</b> Return to the "
                "ColabProSST home page for the complete input instructions. "
                "Click <b>Download input templates</b> there for ready-to-use "
                "examples covering every supported task and input method."
            ),
            layout=widgets.Layout(
                width="100%",
                max_width=ui.GUIDE_WIDTH,
                overflow="visible",
                margin="6px 0 0 0",
            ),
        )
        self.items = [
            self.mode,
            self.hint,
            *self.structure_archive.items,
            self.home_help,
        ]
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

    def set_context(
        self,
        template_group,
        template_task,
        task_label,
        required_columns=(),
        exact_rows=None,
    ):
        self.template_group = template_group
        self.template_task = template_task
        self.task_label = task_label
        self.required_columns = {
            str(column).strip().lower() for column in required_columns
        }
        self.exact_rows = exact_rows
        self._update({"new": self.mode.value})

    def _template_name(self, mode=None):
        if not self.template_group or not self.template_task:
            return ""
        return get_input_template_name(
            self.template_group,
            self.template_task,
            mode or self.mode.value,
        )

    def _expected_template_message(self):
        template_name = self._template_name()
        if not template_name:
            return ""
        return (
            f" Use `{template_name}` from the home-page Input templates for "
            "this task and input method."
        )

    def _mode_options(self):
        return [
            ("Sequence + structure files", self.STRUCTURE),
            ("Sequence only - prepare structure automatically", self.SEQUENCE),
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

    @property
    def structure_zip(self):
        return self.structure_archive.value

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
            uploaded_shape = ""
            if self.pair_mode and "sequence" in columns:
                uploaded_shape = (
                    " The uploaded file looks like a single-protein template, "
                    "but the selected task is a protein-pair task."
                )
            elif not self.pair_mode and {
                "sequence_1",
                "sequence_2",
            }.issubset(columns):
                uploaded_shape = (
                    " The uploaded file looks like a protein-pair template, "
                    "but the selected task is a single-protein task."
                )
            raise ValueError(
                f"{self.task_label} is missing required sequence column(s): "
                f"{missing_sequences}.{uploaded_shape}"
                f"{self._expected_template_message()}"
            )

        missing_task_columns = sorted(self.required_columns - columns)
        if missing_task_columns:
            raise ValueError(
                f"{self.task_label} requires CSV column(s) "
                f"{missing_task_columns}, but they are missing."
                f"{self._expected_template_message()}"
            )

        if self.exact_rows is not None and len(rows) != int(self.exact_rows):
            raise ValueError(
                f"{self.task_label} requires exactly {self.exact_rows} CSV "
                f"row(s), but the uploaded file contains {len(rows)}."
                f"{self._expected_template_message()}"
            )

        for row_number, row in enumerate(rows, start=2):
            for column in sequence_columns:
                normalize_protein_sequence(
                    row.get(column, ""),
                    context=f"row {row_number} {column}",
                    max_residues=(
                        ESMFOLD_MAX_RESIDUES
                        if self.mode.value == self.SEQUENCE
                        else None
                    ),
                )

        if self.pair_mode and self.mode.value == self.STRUCTURE:
            required = {"structure_file_1", "structure_file_2"}
            missing = sorted(required - columns)
            if missing:
                raise ValueError(
                    "Protein-pair structure input requires both "
                    "structure_file_1 and structure_file_2. Missing: "
                    f"{missing}.{self._expected_template_message()}"
                )
        elif self.mode.value == self.STRUCTURE and "structure_file" not in columns:
            raise ValueError(
                "You selected `Sequence + structure files`, but the uploaded "
                "CSV has no structure_file column. Add the structure filenames "
                "to the CSV or choose the sequence-only input method."
                f"{self._expected_template_message()}"
            )
        if self.mode.value == self.STRUCTURE:
            if not self.structure_zip:
                raise ValueError(
                    "Upload the Structure ZIP containing the PDB/mmCIF files "
                    "named in the CSV."
                )
            archive_path = Path(self.structure_zip)
            if not archive_path.is_file():
                raise FileNotFoundError(
                    f"Structure ZIP does not exist: {archive_path}"
                )

    def _update(self, change):
        mode = change["new"]
        self.structure_archive.set_visible(mode == self.STRUCTURE)
        if self.template_group and self.template_task:
            sequence_template = self._template_name(self.SEQUENCE)
            structure_template = self._template_name(self.STRUCTURE)
            selected_template = self._template_name(mode)
            self.home_help.value = (
                f"<b>Templates for {self.task_label}:</b><br>"
                f"Sequence only: <code>{sequence_template}</code><br>"
                "Sequence + structure files: "
                f"<code>{structure_template}</code><br>"
                f"Current selection: <b><code>{selected_template}</code></b>. "
                "Download these files from <b>Input templates</b> on the "
                "ColabProSST home page."
            )
        if self.pair_mode and mode == self.SEQUENCE:
            self.hint.value = (
                "Upload a CSV with <code>sequence_1</code> and "
                "<code>sequence_2</code>. Only these sequence columns are used; "
                "structure-token columns are not required. ColabProSST predicts "
                "both structures and generates matching tokens automatically. "
                f"Each sequence must be at most {ESMFOLD_MAX_RESIDUES} residues."
            )
        elif self.pair_mode and mode == self.STRUCTURE:
            self.hint.value = (
                "Upload a CSV with <code>sequence_1</code>, "
                "<code>sequence_2</code>, <code>structure_file_1</code>, and "
                "<code>structure_file_2</code>, plus one ZIP containing those "
                "PDB/mmCIF files. Optional <code>chain_1</code> and "
                "<code>chain_2</code> columns select a chain. ColabProSST "
                "generates both token sequences automatically. Each ProSST "
                "input supports up to 2046 residues."
            )
        elif mode == self.SEQUENCE:
            self.hint.value = (
                "Upload a CSV with <code>sequence</code>. Only this sequence "
                "column is used; <code>structure_tokens</code> is not required. "
                "ColabProSST predicts each structure and generates tokens for "
                "the selected model automatically. Sequences must be at most "
                f"{ESMFOLD_MAX_RESIDUES} residues."
            )
        else:
            self.hint.value = (
                "Upload a CSV containing <code>sequence</code> and "
                "<code>structure_file</code>, plus one ZIP containing the named "
                "PDB/mmCIF files. Add an optional <code>chain</code> column when "
                "a structure contains multiple chains. ColabProSST validates "
                "the sequence and generates tokens for the selected model. "
                "ProSST inputs support up to 2046 residues."
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
        self.latest_adapter = ""
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

            def download_result(_button, download_path=str(path)):
                self._download_with_status(download_path)

            button.on_click(download_result)
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
        self.latest_adapter = metadata["artifact_path"]
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
    def _task_label(task_type):
        return {
            "classification": "Protein-level Classification",
            "regression": "Protein-level Regression",
            "token_classification": "Residue-level Classification",
            "pair_classification": "Protein-pair Classification",
            "pair_regression": "Protein-pair Regression",
        }[task_type]

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
                "<code>stage</code>. Then choose one of the two input methods "
                "below."
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
            "<li><b>Sequence + structure files:</b> upload a CSV containing "
            "amino-acid sequences and the corresponding PDB/mmCIF filenames, "
            "then upload those files together as one Structure ZIP. Use "
            "<code>structure_file</code> for single-protein tasks and "
            "<code>structure_file_1</code>/<code>structure_file_2</code> for "
            "protein-pair tasks. ColabProSST validates each sequence and "
            "generates the model-specific structure tokens automatically. This "
            "method is suitable when you already have experimental or predicted "
            "structures, including proteins longer than the automatic service's "
            "limit, up to the current ProSST input maximum of 2046 residues.</li>"
            "<li><b>Sequence only:</b> upload a CSV containing amino-acid "
            "sequences. ColabProSST predicts structures and generates tokens "
            "automatically. This uses the public ESMFold service, sends each "
            f"sequence to that service, and supports up to {ESMFOLD_MAX_RESIDUES} "
            "residues per sequence.</li>"
            "</ol>"
            "<hr style='border:0;border-top:1px solid #dadce0;margin:18px 0 12px'>"
            "<h3 style='margin:0 0 8px'>Input templates</h3>"
            "<p><b>Start here:</b> download ready-to-use input examples for "
            "every supported task. The package contains separate sequence-only "
            "and sequence + structure-file examples. These input formats work "
            "with every official ProSST model; no model selection is required "
            "before downloading.</p>",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )

    def _build_system_widgets(self):
        self.back_button = self._button("Go back", width="120px", style="success")
        self.refresh_button = self._button("Refresh", width="120px", style="success")
        self.stop_button = self._button("Stop", width="120px", style="danger")
        self.system_status = self._new_system_status()

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
        self._system_status_index = len(self.system_widgets) - 1

    def _new_system_status(self):
        return self.widgets.Output(
            layout=self.widgets.Layout(width=self.WIDTH)
        )

    def _reset_system_status(self):
        self.system_status = self._new_system_status()
        self.system_widgets[self._system_status_index] = self.system_status

    @staticmethod
    def _download_file_once(path):
        from google.colab import files

        download_path = Path(path)
        if not download_path.is_file():
            raise FileNotFoundError(f"Download file does not exist: {download_path}")
        files.download(str(download_path))

    def _download_with_status(self, path):
        download_path = Path(path)
        self.system_status.clear_output(wait=True)
        try:
            with self.system_status:
                print(
                    f"Download started: {download_path.name}. "
                    "Check your browser downloads."
                )
                # Capture Colab's transient Javascript in this widget so it
                # cannot replace the cell output that owns the full interface.
                self._download_file_once(download_path)
        except Exception as exc:
            self.system_status.clear_output(wait=True)
            with self.system_status:
                print(
                    f"Download failed for {download_path}: "
                    f"{type(exc).__name__}: {exc}"
                )

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
        # Download Javascript captured by an Output widget can replay if that
        # widget is displayed again. Every page receives a fresh status slot.
        self._reset_system_status()
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

        self._download_with_status(path)

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

    def _download_input_templates(self, button):
        output = self.system_status

        def action():
            self.workflow.create_input_templates(download=True)

        self._start_task(button, output, action)

    def _home_page(self):
        self.current_page = self._home_page
        title = self._heading(
            "Please choose what you want to do with ColabProSST"
        )
        input_guide = self._input_guide()
        input_template_button = self._button(
            "Download input templates", width="280px", style="info"
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
        input_template_button.on_click(
            lambda button: self._download_input_templates(button)
        )

        self._display_page(
            title,
            train_button,
            predict_button,
            share_button,
            self._separator(),
            input_guide,
            input_template_button,
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
        training_method_help = self._html(
            "<b>Training method: LoRA fine-tuning.</b> ColabProSST trains the "
            "complete task head and LoRA parameters inside the ProSST backbone. "
            "The official backbone weights remain frozen. The result is a "
            "compact PEFT adapter ZIP that ColabProSST can load together with "
            "the selected official ProSST model.",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
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
            "These settings apply when starting from an official ProSST model. "
            "When continuing an adapter, its saved rank, alpha, and dropout are "
            "reused and training starts with a new optimizer.",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )
        training_start = widgets.ToggleButtons(
            options=[
                ("Start from official ProSST", "fresh"),
                ("Continue from a ColabProSST adapter", "adapter"),
            ],
            value="fresh",
            description="Training start:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        initial_adapter = _ModelArtifactField(
            self,
            "Initial adapter:",
            "Path to a ColabProSST adapter directory or ZIP",
        )
        initial_adapter.value = self.latest_adapter
        adapter_help = (
            "Upload a ColabProSST adapter ZIP, enter an extracted adapter "
            "directory, or load a compatible ProSSTHub repository. The adapter "
            "and its saved task head are continued with a new optimizer; "
            "the task, base model, structure vocabulary, and category count "
            "must match."
        )
        adapter_help_widget = self._html(
            adapter_help,
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )
        csv_input = _UploadField(
            self,
            "Training CSV:",
            "CSV with sequence, label, and stage",
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
        gradient_checkpointing = widgets.Checkbox(
            value=True,
            description="Use gradient checkpointing",
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
            required_columns = (
                {"residue_labels", "stage"}
                if selected_task == "token_classification"
                else {"label", "stage"}
            )
            structure_input.set_context(
                "training",
                selected_task,
                f"{self._task_label(selected_task)} training",
                required_columns=required_columns,
            )
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
            advanced_button.description = (
                "Hide advanced settings" if show else "Show advanced settings"
            )
            update_training_controls()

        def update_training_controls(_change=None):
            continue_adapter = training_start.value == "adapter"
            initial_adapter.set_visible(continue_adapter)
            adapter_help_widget.layout.display = (
                None if continue_adapter else "none"
            )
            advanced_visible = advanced_button.description == "Hide advanced settings"
            show_new_lora_config = advanced_visible and not continue_adapter
            for item in [lora_rank, lora_alpha, lora_dropout]:
                item.layout.display = None if show_new_lora_config else "none"
            gradient_checkpointing.layout.display = (
                None if advanced_visible else "none"
            )
            lora_help.layout.display = None if advanced_visible else "none"

        def apply_initial_artifact(metadata):
            self._apply_artifact_metadata(
                metadata,
                task_type,
                model,
                num_labels,
            )
            training_start.value = "adapter"

        def train():
            if not csv_input.value:
                raise ValueError("Upload a training CSV or enter its path.")
            continue_adapter = training_start.value == "adapter"
            if continue_adapter and not initial_adapter.value:
                raise ValueError(
                    "Upload an initial ColabProSST adapter or enter its path."
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
            if continue_adapter:
                print("Continuing adapter:", initial_adapter.value)
                print("Optimizer state: new optimizer")
            result = self.workflow.train_downstream(
                task_type=task_type.value,
                input_csv=csv_input.value,
                input_mode=structure_input.input_mode,
                structure_zip=structure_input.structure_zip,
                task_name=clean_task_name,
                num_labels=num_labels.value,
                max_epochs=epochs.value,
                batch_size=batch_size.value,
                learning_rate=learning_rate.value,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                gradient_checkpointing=gradient_checkpointing.value,
                initial_adapter=(
                    initial_adapter.value if continue_adapter else ""
                ),
                lora_rank=lora_rank.value,
                lora_alpha=lora_alpha.value,
                lora_dropout=lora_dropout.value,
                download=False,
            )
            self.latest_adapter = result["adapter_path"]
            self.latest_model_path = result["model_path"]
            self.latest_task_type = result["task_type"]
            self.latest_num_labels = num_labels.value
            print("Training finished.")
            print("Adapter:", result["adapter_path"])
            print("Adapter ZIP:", result["adapter_download_path"])
            print("Test predictions:", result["test_result_csv"])
            self._display_result_downloads(
                ("LoRA adapter ZIP", result["adapter_download_path"]),
                ("test predictions CSV", result["test_result_csv"]),
            )
            finish_hint.value = (
                "<h3>The training is completed. You can then:</h3>"
                "<ul>"
                "<li><b>Train again:</b> click <b>Refresh</b>, adjust the "
                "settings, and start a new task.</li>"
                "<li><b>Use this model for prediction:</b> click <b>Go back</b>, "
                "choose <b>I want to use existing models to make prediction</b>, "
                "then choose <b>Protein property prediction</b>. The adapter "
                "is selected automatically in this session.</li>"
                "<li><b>Share this model:</b> click <b>Go back</b> and choose "
                "<b>I want to share my model publicly</b>.</li>"
                "</ul>"
            )
            finish_hint.layout.display = None

        update_task({"new": task_type.value})
        task_type.observe(update_task, names="value")
        initial_adapter.on_loaded(apply_initial_artifact)
        update_training_controls()
        training_start.observe(update_training_controls, names="value")
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
            training_method_help,
            training_start,
            *initial_adapter.items,
            adapter_help_widget,
            self._heading("Dataset setting:", level=3),
            dataset_help,
            *csv_input.items,
            *structure_input.display_items,
            self._heading("Training hyper-parameters:", level=3),
            batch_size,
            epochs,
            learning_rate,
            advanced_button,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_help,
            gradient_checkpointing,
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
        self._display_page(
            self._heading(
                "ColabProSST supports multiple prediction tasks, which one "
                "would you like to choose?"
            ),
            self._separator(),
            property_button,
            self._html(
                "Use a trained ColabProSST adapter for protein-level "
                "classification/regression, residue-level classification, or "
                "protein-pair classification/regression."
            ),
            self._separator(),
            embedding_button,
            self._html(
                "Extract final-layer protein-level vectors, residue-level "
                "vectors, or both from an official ProSST model or a trained "
                "ColabProSST adapter."
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
        adapter = _ModelArtifactField(
            self,
            "Fine-tuned adapter:",
            "Path to a ColabProSST adapter directory or ZIP",
        )
        adapter.value = self.latest_adapter
        csv_input = _UploadField(
            self,
            "Prediction CSV:",
            "CSV with sequence or sequence pairs",
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
            structure_input.set_context(
                "prediction",
                "pair" if self._is_pair_task(selected_task) else "single",
                f"{self._task_label(selected_task)} prediction",
            )
            csv_input.path.placeholder = (
                "CSV with sequence_1 and sequence_2"
                if self._is_pair_task(selected_task)
                else "CSV with sequence"
            )

        def predict():
            if not adapter.value:
                raise ValueError(
                    "Upload a ColabProSST adapter or enter its path."
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
                adapter_path=adapter.value,
                input_mode=structure_input.input_mode,
                structure_zip=structure_input.structure_zip,
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
            )

        adapter.on_loaded(
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
            *adapter.items,
            self._html(
                "Use an extracted ColabProSST adapter directory, a downloaded "
                "adapter ZIP, or a compatible ProSSTHub repository. The task "
                "head is included in the adapter. Select the adapter's original "
                "task and official ProSST base model."
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
                ("Fine-tuned / community adapter", "artifact"),
            ],
            value="official",
            description="Embedding model:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        embedding_artifact = _ModelArtifactField(
            self,
            "Fine-tuned adapter:",
            "Path to a ColabProSST adapter directory or ZIP",
        )
        embedding_artifact.value = self.latest_adapter
        embedding_artifact.set_visible(False)
        csv_input = _UploadField(
            self,
            "Protein CSV:",
            "CSV with sequence",
        )
        structure_input = _StructureInput(self)
        structure_input.set_context(
            "embedding",
            "single",
            "Embedding extraction",
        )
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
            self.latest_adapter = metadata["artifact_path"]
            self.latest_model_path = metadata["model_path"]

        def extract():
            completion.value = ""
            completion.layout.display = "none"
            if not csv_input.value:
                raise ValueError("Upload a protein CSV or enter its path.")
            use_artifact = embedding_model_source.value == "artifact"
            if use_artifact and not embedding_artifact.value:
                raise ValueError(
                    "Upload a fine-tuned adapter, enter its path, or load "
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
                structure_zip=structure_input.structure_zip,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                level=level.value,
                batch_size=batch_size.value,
                max_length=max_length.value,
                adapter_path=(
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
            "One-row CSV with sequence",
        )
        structure_input = _StructureInput(self)
        structure_input.set_context(
            "saturation",
            "single",
            "Single-site saturation mutagenesis",
            exact_rows=1,
        )
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
                structure_zip=structure_input.structure_zip,
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
            "CSV with sequence and mutant",
        )
        structure_input = _StructureInput(self)
        structure_input.set_context(
            "zero_shot",
            "single",
            "Mutational effect prediction",
            required_columns={"mutant"},
        )
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
                structure_zip=structure_input.structure_zip,
                model_path=model.value,
                structure_vocab_size=model_spec.structure_vocab_size,
                download=False,
            )
            self.latest_model_path = model.value
            self.display(result.head())
            self._display_result_downloads(
                ("mutation scores CSV", result.attrs.get("output_csv")),
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
                "classification and regression adapters do not provide this "
                "mutation score; use them under <b>Protein property "
                "prediction</b> instead.",
                width="100%",
                max_width=self.GUIDE_WIDTH,
                overflow="visible",
            ),
            self._heading("Mutation data:", level=3),
            self._html(
                "The CSV must contain <code>sequence</code> and "
                "<code>mutant</code>. Then choose sequence + structure files or "
                "automatic sequence-only preparation. Zero-shot mutation "
                "scoring uses official "
                "ProSST family models, not downstream fine-tuned adapters."
            ),
            *csv_input.items,
            *structure_input.display_items,
            self._separator(),
            start_button,
            output,
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
        adapter = _ModelArtifactField(
            self,
            "ColabProSST adapter:",
            "Path to a ColabProSST adapter directory or ZIP",
        )
        adapter.value = self.latest_adapter
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
            value="A ProSST adapter trained with ColabProSST.",
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
            if not adapter.value:
                raise ValueError(
                    "Upload a ColabProSST adapter or enter its path."
                )
            clean_repo_name = repo_name.value.strip()
            if not clean_repo_name:
                raise ValueError("Enter a Hugging Face repository name.")
            repo_id = self.workflow.personal_hf_model_repo_id(clean_repo_name)
            print("Uploading model to your Hugging Face account...")
            print("Target repository:", repo_id)
            model_spec = get_prosst_model_spec(model.value)
            package = self.workflow.upload_adapter_to_hf(
                repo_id=repo_id,
                adapter_path=adapter.value,
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
        adapter.on_loaded(
            lambda metadata: self._apply_artifact_metadata(
                metadata,
                task_type,
                model,
                num_labels,
            )
        )
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
            *adapter.items,
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
