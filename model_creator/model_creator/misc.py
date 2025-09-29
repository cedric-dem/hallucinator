import keras
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from config import *

def save_models(encoder_model, decoder_model, output_dir, epoch_number):
	output_dir = Path(output_dir)
	epoch_dir = output_dir / f"epoch_{epoch_number:04d}"
	epoch_dir.mkdir(parents = True, exist_ok = True)
	encoder_model.save(epoch_dir / "model_encoder.keras")
	decoder_model.save(epoch_dir / "model_decoder.keras")

def save_loss_plot(history, output_path):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents = True, exist_ok = True)

	epochs = range(1, len(history.history.get("loss", [])) + 1)
	plt.figure()
	if "loss" in history.history:
		plt.plot(epochs, history.history["loss"], label = "Training Loss")
	if "val_loss" in history.history:
		plt.plot(epochs, history.history["val_loss"], label = "Validation Loss")
	plt.title("Training and Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_path, format = "jpg")
	plt.close()

def save_epoch_time_plot(epoch_times, output_path):
	if not epoch_times:
		return

	output_path = Path(output_path)
	output_path.parent.mkdir(parents = True, exist_ok = True)

	epochs = range(1, len(epoch_times) + 1)

	plt.figure()
	plt.plot(epochs, epoch_times, marker = "o", label = "Epoch Duration (s)")
	plt.title("Time per Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Seconds")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_path, format = "jpg")
	plt.close()

def load_comparison_images(directory, target_size):
	directory = Path(directory)
	if not directory.is_dir():
		print(f"Comparison images directory '{directory}' does not exist.")
		return np.empty((0, target_size[0], target_size[1], 3), dtype = "float32")

	images = []
	if hasattr(Image, "Resampling"):
		resample_method = Image.Resampling.LANCZOS
	else:
		resample_method = Image.LANCZOS


	for file_path in sorted(directory.iterdir()):
		if not file_path.is_file():
			continue

		try:
			image = Image.open(file_path).convert("RGB")
			image = image.resize(target_size, resample = resample_method)
			image_array = np.asarray(image, dtype = "float32") / 255.0
			images.append(image_array)
		except (OSError, ValueError) as error:
			print(f"Skipping comparison image '{file_path}': {error}")

	if not images:
		print(f"No valid comparison images were found in '{directory}'.")
		return np.empty((0, target_size[0], target_size[1], 3), dtype = "float32")

	return np.stack(images, axis = 0)


def save_comparisons(model, output_dir, epoch_number, batch_x, batch_y, num_examples, save_images = True):
	output_dir = Path(output_dir)
	if len(batch_x) == 0 or num_examples <= 0:
		return None

	predicted = model.predict(batch_x, verbose = 0)

	num_examples = min(num_examples, len(batch_x), len(batch_y))

	if save_images:
		epoch_dir = output_dir / f"epoch_{epoch_number:04d}"
		epoch_dir.mkdir(parents = True, exist_ok = True)

	total_difference = 0.0

	for index in range(num_examples):
		target_array = batch_y[index]
		predicted_array = predicted[index]
		total_difference += np.sum(np.abs(target_array - predicted_array))

		if save_images:
			target_image = np.clip(target_array * 255, 0, 255).astype("uint8")
			output_image = np.clip(predicted_array * 255, 0, 255).astype("uint8")

			comparison = np.hstack((target_image, output_image))
			comparison_image = Image.fromarray(comparison)

			comparison_path = epoch_dir / f"comparison_{index + 1:02d}.jpg"

			comparison_image.save(comparison_path, format = "JPEG")

	return calculate_average_difference_percentage(
		total_difference,
		num_examples,
		IMG_DIM,
		IMG_DIM,
	)

def calculate_average_difference_percentage(total_difference, num_samples, height, width):
	if num_samples <= 0 or height <= 0 or width <= 0:
		return None

	denominator = float(num_samples * 3 * height * width)
	return 100 * float(total_difference) / denominator


class EpochTimeTracker(keras.callbacks.Callback):
	def __init__(self, output_path):
		super().__init__()
		self.output_path = Path(output_path)
		self.epoch_times = []
		self._epoch_start_time = None

	def on_epoch_begin(self, epoch, logs = None):
		self._epoch_start_time = time.perf_counter()

	def on_epoch_end(self, epoch, logs = None):
		if self._epoch_start_time is None:
			return

		duration = time.perf_counter() - self._epoch_start_time
		self.epoch_times.append(duration)
		self._epoch_start_time = None

	def on_train_end(self, logs = None):
		save_epoch_time_plot(self.epoch_times, self.output_path)

class AverageDifferenceTracker(keras.callbacks.Callback):
	def __init__(self, output_dir, batch_x, batch_y, num_examples):
		super().__init__()
		self.output_dir = Path(output_dir)
		self.batch_x = batch_x
		self.batch_y = batch_y
		self.num_examples = num_examples
		self.differences = []
		self.comparisons_dir = self.output_dir / "comparisons"
		self.output_path = self.output_dir / "plots" / "avg_difference_curve.jpg"

	def on_train_end(self, logs = None):
		if not self.differences:
			return

		self.output_path.parent.mkdir(parents = True, exist_ok = True)
		epochs = range(1, len(self.differences) + 1)
		plt.figure()
		plt.plot(epochs, self.differences, marker = "o", label = "Avg Difference (%)")
		plt.title("Average Difference Percentage per Epoch")
		plt.xlabel("Epoch")
		plt.ylabel("Average Difference (%)")
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(self.output_path, format = "jpg")
		plt.close()

	def on_epoch_end(self, epoch, logs = None):
		if len(self.batch_x) == 0 or self.num_examples <= 0:
			return

		avg_difference = save_comparisons(
			self.model,
			self.comparisons_dir,
			epoch + 1,
			self.batch_x,
			self.batch_y,
			self.num_examples,
			save_images = False,
		)

		if avg_difference is not None:
			self.differences.append(avg_difference)

class PeriodicModelSaver(keras.callbacks.Callback):
	def __init__(self, frequency, output_dir, encoder_model, decoder_model):
		super().__init__()
		self.frequency = frequency
		self.output_dir = Path(output_dir)
		self.encoder_model = encoder_model
		self.decoder_model = decoder_model

	def on_train_begin(self, logs = None):
		if self.frequency <= 0:
			return
		initial_encoder_path = self.output_dir / "epoch_0000" / "model_encoder.keras"
		if not initial_encoder_path.exists():
			save_models(self.encoder_model, self.decoder_model, self.output_dir, 0)

	def on_epoch_end(self, epoch, logs = None):
		if self.frequency <= 0:
				return
		epoch_number = epoch + 1
		if epoch_number % self.frequency == 0:
			save_models(self.encoder_model, self.decoder_model, self.output_dir, epoch_number)


class PeriodicComparisonSaver(keras.callbacks.Callback):
	def __init__(self, frequency, output_dir, batch_x, batch_y, num_examples):
		super().__init__()
		self.frequency = frequency
		self.output_dir = Path(output_dir)
		self.batch_x = batch_x
		self.batch_y = batch_y
		self.num_examples = num_examples
		self.comparisons_dir = self.output_dir / "comparisons"

	def on_train_begin(self, logs = None):
		if self.frequency <= 0:
			return
		if len(self.batch_x) == 0:
			print("Skipping comparison image saving because no comparison images are available.")
			return
		save_comparisons(self.model, self.comparisons_dir, 0, self.batch_x, self.batch_y, self.num_examples)

	def on_epoch_end(self, epoch, logs = None):
		if self.frequency <= 0:
			return
		if len(self.batch_x) == 0:
			return
		epoch_number = epoch + 1
		if epoch_number % self.frequency == 0:
			save_comparisons(
				self.model,
				self.comparisons_dir,
				epoch_number,
				self.batch_x,
				self.batch_y,
				self.num_examples,
			)

def resize_all_images(old_directory, new_directory, new_size):
	old_dir = Path(old_directory)
	new_dir = Path(new_directory)

	new_dir.mkdir(parents = True, exist_ok = True)

	if isinstance(new_size, int):
		target_size = (new_size, new_size)
	elif isinstance(new_size, (tuple, list)) and len(new_size) == 2:
		target_size = tuple(new_size)
	else:
		raise ValueError("new_size not ok")

	valid_extensions = {".png", ".jpg", ".jpeg"}

	for file in old_dir.iterdir():
		if not file.is_file() or file.suffix.lower() not in valid_extensions:
			continue

		try:
			img = Image.open(file)
			img = img.resize(target_size, Image.LANCZOS)
			out_path = new_dir / file.name
			ext = file.suffix.lower()

			if ext in {".jpg", ".jpeg"}:
				if img.mode in {"RGBA", "P"}:
					img = img.convert("RGB")
				img.save(out_path, quality = 95, optimize = True)
			elif ext == ".png":
				if img.mode == "P":
					img = img.convert("RGBA")
				img.save(out_path, optimize = True)
			else:
				img.save(out_path)
		except Exception as e:
			print(f"Error {file.name} : {e}")

def compute_image_differences(original_directory, processed_directory):
	original_dir = Path(original_directory)
	processed_dir = Path(processed_directory)

	differences = {}
	for processed_file in processed_dir.iterdir():
		if not processed_file.is_file():
			continue

		original_file = original_dir / processed_file.name
		if not original_file.exists():
			continue

		try:
			with Image.open(original_file) as original_img, Image.open(
					processed_file
			) as processed_img:
				original_rgb = original_img.convert("RGB")
				processed_rgb = processed_img.convert("RGB")

				if original_rgb.size != processed_rgb.size:
					original_rgb = original_rgb.resize(processed_rgb.size, Image.LANCZOS)

				diff_total = 0
				for orig_pixel, proc_pixel in zip(
					original_rgb.getdata(), processed_rgb.getdata()
				):
					diff_total += sum(abs(o - p) for o, p in zip(orig_pixel, proc_pixel))

				differences[processed_file.name] = diff_total
		except Exception as e:
			print(f"Error comparing {processed_file.name}: {e}")

	return differences