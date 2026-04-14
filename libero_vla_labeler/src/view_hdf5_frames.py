from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import List, Optional

import h5py
import numpy as np
from PIL import Image, ImageTk

from src.gripper_segmenter import list_demo_keys
from src.pipeline import load_config


def _find_hdf5_files(directory: str) -> List[str]:
    """Recursively find all .hdf5 files under directory, skipping _bddl folders."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        str(p) for p in root.rglob("*.hdf5")
        if "_bddl" not in p.parts
    )


def _load_frames_from_hdf5(hdf5_path: str, demo_key: str, camera_key: str) -> np.ndarray:
    """Load RGB frames for a demo from HDF5. Returns (T, H, W, C) uint8 array."""
    obs_path = f"data/{demo_key}/obs/{camera_key}"
    with h5py.File(hdf5_path, "r") as f:
        if obs_path in f:
            return f[obs_path][:].astype(np.uint8)
        # Fallback: try any available image observation key
        obs_group = f.get(f"data/{demo_key}/obs")
        if obs_group is not None:
            for key in obs_group.keys():
                data = obs_group[key][:]
                if data.ndim == 4 and data.shape[-1] == 3:
                    print(f"      camera '{camera_key}' not found, using '{key}' instead.")
                    return data.astype(np.uint8)
    raise ValueError(f"No RGB observation found for {demo_key} in {hdf5_path}")


class HDF5FrameViewer:
    def __init__(
        self,
        root: tk.Tk,
        hdf5_path: Optional[str] = None,
        hdf5_dir: Optional[str] = None,
        initial_episode: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.root = root
        self.camera_key = (config or {}).get("camera", "agentview_rgb")
        self.hdf5_files = _find_hdf5_files(hdf5_dir) if hdf5_dir else []
        if hdf5_path and hdf5_path not in self.hdf5_files:
            self.hdf5_files.insert(0, hdf5_path)
        self.hdf5_path = hdf5_path or (self.hdf5_files[0] if self.hdf5_files else None)
        self.initial_episode = initial_episode

        self.episodes: List[str] = []
        self.current_episode = ""
        self.frames: Optional[np.ndarray] = None
        self.current_frame = 0
        self.rotation = 0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._updating_slider = False

        self.root.title("HDF5 Agent View Viewer")
        self.root.geometry("1600x1100")

        self._build_ui()
        if self.hdf5_path:
            self._load_hdf5_file(self.hdf5_path)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill="x")

        ttk.Label(controls, text="HDF5 File").grid(row=0, column=0, sticky="w")
        self.file_var = tk.StringVar(value=self.hdf5_path or "")
        self.file_box = ttk.Combobox(
            controls,
            textvariable=self.file_var,
            values=self.hdf5_files,
            state="readonly" if self.hdf5_files else "normal",
            width=90,
        )
        self.file_box.grid(row=0, column=1, columnspan=4, sticky="ew", padx=(8, 8))
        self.file_box.bind("<<ComboboxSelected>>", self._on_file_change)
        ttk.Button(controls, text="Browse", command=self._browse_file).grid(row=0, column=5, sticky="w")

        ttk.Label(controls, text="Episode").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.episode_var = tk.StringVar(value="")
        self.episode_box = ttk.Combobox(
            controls,
            textvariable=self.episode_var,
            values=[],
            state="readonly",
            width=50,
        )
        self.episode_box.grid(row=1, column=1, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.episode_box.bind("<<ComboboxSelected>>", self._on_episode_change)

        self.prev_button = ttk.Button(controls, text="Prev", command=self._prev_frame)
        self.prev_button.grid(row=1, column=2, padx=4, pady=(10, 0))
        self.next_button = ttk.Button(controls, text="Next", command=self._next_frame)
        self.next_button.grid(row=1, column=3, padx=4, pady=(10, 0))

        ttk.Button(controls, text="Rotate Left", command=lambda: self._rotate(-90)).grid(row=1, column=4, padx=4, pady=(10, 0))
        ttk.Button(controls, text="Rotate Right", command=lambda: self._rotate(90)).grid(row=1, column=5, padx=4, pady=(10, 0))

        ttk.Label(controls, text="Frame").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.frame_slider = ttk.Scale(controls, from_=0, to=0, orient="horizontal", command=self._on_slider_change)
        self.frame_slider.grid(row=2, column=1, columnspan=5, sticky="ew", pady=(10, 0))

        ttk.Label(controls, text="Jump To").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.jump_var = tk.StringVar(value="0")
        self.jump_entry = ttk.Entry(controls, textvariable=self.jump_var, width=12)
        self.jump_entry.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        self.jump_entry.bind("<Return>", self._on_jump)
        ttk.Button(controls, text="Go", command=self._on_jump).grid(row=3, column=2, sticky="w", padx=4, pady=(10, 0))

        ttk.Label(controls, text="Zoom").grid(row=3, column=3, sticky="e", pady=(10, 0))
        self.zoom_var = tk.DoubleVar(value=2.0)
        self.zoom_slider = ttk.Scale(controls, from_=0.5, to=4.0, orient="horizontal", command=self._on_zoom_change)
        self.zoom_slider.set(self.zoom_var.get())
        self.zoom_slider.grid(row=3, column=4, columnspan=2, sticky="ew", pady=(10, 0))

        for column in (1, 4):
            controls.columnconfigure(column, weight=1)

        self.info_var = tk.StringVar(value="Select an HDF5 file to begin.")
        ttk.Label(self.root, textvariable=self.info_var, padding=(12, 0)).pack(anchor="w")

        canvas_frame = ttk.Frame(self.root, padding=12)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, background="#202020")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

    def _browse_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select HDF5 File",
            filetypes=[("HDF5 files", "*.hdf5 *.h5"), ("All files", "*.*")],
        )
        if not selected:
            return
        if selected not in self.hdf5_files:
            self.hdf5_files.append(selected)
            self.hdf5_files = sorted(set(self.hdf5_files))
            self.file_box.configure(values=self.hdf5_files)
        self.file_var.set(selected)
        self._load_hdf5_file(selected)

    def _load_hdf5_file(self, hdf5_path: str) -> None:
        self.hdf5_path = hdf5_path
        self.episodes = list_demo_keys(hdf5_path)
        if not self.episodes:
            raise ValueError(f"No episodes found in {hdf5_path}")

        self.episode_box.configure(values=self.episodes)
        if self.initial_episode in self.episodes:
            self.current_episode = self.initial_episode or self.episodes[0]
        else:
            self.current_episode = self.episodes[0]
        self.episode_var.set(self.current_episode)
        self.rotation = 0
        self._load_episode(self.current_episode)

    def _load_episode(self, episode_name: str) -> None:
        if not self.hdf5_path:
            return
        frames = _load_frames_from_hdf5(self.hdf5_path, episode_name, self.camera_key)
        if len(frames) == 0:
            raise ValueError(f"No RGB frames found for episode {episode_name}")

        self.current_episode = episode_name
        self.frames = frames
        self.current_frame = 0
        self.frame_slider.configure(to=max(len(self.frames) - 1, 0))
        self._render_frame()

    def _render_frame(self) -> None:
        if self.frames is None or len(self.frames) == 0 or not self.hdf5_path:
            return

        self.current_frame = max(0, min(self.current_frame, len(self.frames) - 1))
        frame = np.asarray(self.frames[self.current_frame]).astype(np.uint8)
        image = Image.fromarray(frame)
        if self.rotation:
            image = image.rotate(self.rotation, expand=True)
        zoom = float(self.zoom_slider.get())
        resized = image.resize(
            (
                max(1, int(image.width * zoom)),
                max(1, int(image.height * zoom)),
            ),
            Image.Resampling.NEAREST,
        )
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, resized.width, resized.height))

        self._updating_slider = True
        try:
            self.frame_slider.set(self.current_frame)
        finally:
            self._updating_slider = False

        self.jump_var.set(str(self.current_frame))
        self.info_var.set(
            f"File: {Path(self.hdf5_path).name} | Episode: {self.current_episode} | "
            f"Frame: {self.current_frame}/{len(self.frames) - 1} | Shape: {tuple(frame.shape)} | "
            f"Zoom: {zoom:.1f}x | Rotation: {self.rotation % 360} deg"
        )

    def _on_file_change(self, _event: object) -> None:
        selected = self.file_var.get().strip()
        if selected:
            self._load_hdf5_file(selected)

    def _on_episode_change(self, _event: object) -> None:
        self._load_episode(self.episode_var.get())

    def _on_slider_change(self, value: str) -> None:
        if self._updating_slider:
            return
        self.current_frame = int(float(value))
        self._render_frame()

    def _on_jump(self, _event: object | None = None) -> None:
        if self.frames is None:
            return
        try:
            target = int(self.jump_var.get())
        except ValueError:
            return
        self.current_frame = max(0, min(target, len(self.frames) - 1))
        self._render_frame()

    def _on_zoom_change(self, _value: str) -> None:
        self._render_frame()

    def _rotate(self, delta: int) -> None:
        self.rotation = (self.rotation + delta) % 360
        self._render_frame()

    def _prev_frame(self) -> None:
        self.current_frame -= 1
        self._render_frame()

    def _next_frame(self) -> None:
        self.current_frame += 1
        self._render_frame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View agent-view RGB frames from LIBERO HDF5 files.")
    parser.add_argument("--hdf5", default=None, help="Path to a specific HDF5 file to open first.")
    parser.add_argument("--folder", default=None, help="Dataset folder to browse (e.g. /home/tiandy/libero_100).")
    parser.add_argument("--episode", default=None, help="Episode/demo key to open first (e.g. demo_0).")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = tk.Tk()
    HDF5FrameViewer(
        root,
        hdf5_path=args.hdf5,
        hdf5_dir=args.folder,
        initial_episode=args.episode,
        config=config,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
