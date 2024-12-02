import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
import json
from pathlib import Path
from tml_library.utils import initialize_train_pipeline
from tml_library.trainer import Trainer
from tml_library.evaluator import Evaluator

class MLPipelineGUI(Gtk.Window):
    def __init__(self):
        super().__init__(title="ML Pipeline GUI")
        self.set_border_width(10)
        self.set_default_size(600, 400)

        # Main layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Data File Selection
        self.data_path = None
        file_chooser_label = Gtk.Label(label="Select Data File:")
        vbox.pack_start(file_chooser_label, False, False, 0)

        self.file_chooser = Gtk.FileChooserButton(title="Choose Data File", action=Gtk.FileChooserAction.OPEN)
        self.file_chooser.set_filter(self.create_file_filter("Text Files", "*.txt"))
        self.file_chooser.connect("file-set", self.on_file_set)
        vbox.pack_start(self.file_chooser, False, False, 0)

        # Target Column Drop-down
        target_label = Gtk.Label(label="Select Target Column:")
        vbox.pack_start(target_label, False, False, 0)

        self.target_combo = Gtk.ComboBoxText()
        self.target_combo.append_text("0")  # Add options dynamically if needed
        self.target_combo.append_text("1")
        self.target_combo.set_active(0)
        vbox.pack_start(self.target_combo, False, False, 0)

        # Model Selection Drop-down
        model_label = Gtk.Label(label="Select Model:")
        vbox.pack_start(model_label, False, False, 0)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.append_text("RandomForest")
        self.model_combo.append_text("SVC")
        self.model_combo.set_active(0)
        vbox.pack_start(self.model_combo, False, False, 0)

        # Log File Input
        log_label = Gtk.Label(label="Log File Name:")
        vbox.pack_start(log_label, False, False, 0)

        self.log_entry = Gtk.Entry()
        self.log_entry.set_text("train.log")
        vbox.pack_start(self.log_entry, False, False, 0)

        # Start Button
        self.start_button = Gtk.Button(label="Start Pipeline")
        self.start_button.connect("clicked", self.on_start_pipeline_clicked)
        self.start_button.set_sensitive(False)
        vbox.pack_start(self.start_button, False, False, 0)

        # Output Text View
        self.output_view = Gtk.TextView()
        self.output_view.set_editable(False)
        self.output_view.set_wrap_mode(Gtk.WrapMode.WORD)
        vbox.pack_start(Gtk.ScrolledWindow(child=self.output_view), True, True, 0)

    def create_file_filter(self, name, pattern):
        file_filter = Gtk.FileFilter()
        file_filter.set_name(name)
        file_filter.add_pattern(pattern)
        return file_filter

    def on_file_set(self, widget):
        self.data_path = widget.get_filename()
        self.append_output(f"Data file selected: {self.data_path}")
        self.start_button.set_sensitive(True)

    def on_start_pipeline_clicked(self, widget):
        if not self.data_path:
            self.append_output("No data file selected.")
            return

        config = {
            "data_path": self.data_path,
            "target_column": self.target_combo.get_active_text(),
            "log_file": self.log_entry.get_text(),
            "model_name": self.model_combo.get_active_text(),
            "model_params": {}  # Extend this to include dynamic model parameter entry if required
        }

        try:
            # Save temporary config file
            config_path = Path("temp_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            self.append_output(f"Configuration saved: {config}")

            # Run pipeline
            self.append_output("Initializing pipeline...")
            X_train, y_train, model, param_grid, logger = initialize_train_pipeline(config_path)

            # Train model
            self.append_output("Training model...")
            trainer = Trainer(model, param_grid)
            best_model = trainer.train(X_train, y_train)
            trainer.save_model("best_model.joblib")
            self.append_output("Model training complete. Model saved as 'best_model.joblib'.")

            # # Evaluate model
            # self.append_output("Evaluating model...")
            # evaluator = Evaluator(best_model)
            # metrics = evaluator.evaluate(X_test, y_test)
            # self.append_output(f"Evaluation Metrics:\n{metrics}")

        except Exception as e:
            self.append_output(f"Error: {str(e)}")

    def append_output(self, message):
        buffer = self.output_view.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, message + "\n")


if __name__ == "__main__":
    win = MLPipelineGUI()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
