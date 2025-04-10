"""Record3D visualizer

Batch process Record3D captures to generate recordings for multiple data folders.
"""

import time
import sys
import os
import argparse
from pathlib import Path

import numpy as onp
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf
import matplotlib.cm as cm  # For colormap

def process_folder(
    data_path: Path,
    downsample_factor: int,
    max_frames: int,
    conf_threshold: float,
    foreground_conf_threshold: float,
    point_size: float,
    camera_frustum_scale: float,
    no_mask: bool,
    xyzw: bool,
    axes_scale: float,
    bg_downsample_factor: int,
    output_dir: Path,
) -> None:
    print(f"Processing folder: {data_path}")

    server = viser.ViserServer()
    server.scene.set_up_direction('-z')

    loader = viser.extras.Record3dLoader_Customized(
        data_path,
        conf_threshold=conf_threshold,
        foreground_conf_threshold=foreground_conf_threshold,
        no_mask=no_mask,
        xyzw=xyzw,
        init_conf=True,
    )
    num_frames = min(max_frames, loader.num_frames())

    # Load frames
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    bg_positions = []
    bg_colors = []
    for i in tqdm(range(num_frames)):
        frame = loader.get_frame(i)
        position, color, bg_position, bg_color = frame.get_point_cloud(downsample_factor, bg_downsample_factor)

        bg_positions.append(bg_position)
        bg_colors.append(bg_color)

        # Add base frame
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position,
            colors=color,
            point_size=point_size,
            point_shape="rounded",
        )

        # Compute color for frustum based on frame index
        norm_i = i / (num_frames - 1) if num_frames > 1 else 0  # Normalize index to [0, 1]
        color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components

        # Place the frustum with the computed color
        fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
        aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=camera_frustum_scale,
            image=frame.rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
            color=color_rgb,  # Set the color for the frustum
        )

        # Add axes
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 10,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    # Add background frame
    bg_positions = onp.concatenate(bg_positions, axis=0)
    bg_colors = onp.concatenate(bg_colors, axis=0)
    server.scene.add_point_cloud(
        name=f"/frames/background",
        points=bg_positions,
        colors=bg_colors,
        point_size=point_size,
        point_shape="rounded",
    )

    # Automatically play through frames and record the scene
    rec = server._start_scene_recording()
    rec.set_loop_start()

    sleep_duration = 1.0 / loader.fps if loader.fps > 0 else 0.033  # Default to ~30 FPS

    for t in range(num_frames):
        # Update the scene to show frame t
        with server.atomic():
            for i, frame_node in enumerate(frame_nodes):
                frame_node.visible = (i == t)
        server.flush()
        rec.insert_sleep(sleep_duration)

    # Set all frames invisible
    with server.atomic():
        for frame_node in frame_nodes:
            frame_node.visible = False
    server.flush()

    # Finish recording
    bs = rec.end_and_serialize()

    # Save the recording to a file
    output_path = output_dir / f"recording_{data_path.name}.viser"
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bs)
    print(f"Recording saved to {output_path.resolve()}")

def main(
    data_paths: list[Path],
    output_dir: Path = Path("./viser_result"),
    downsample_factor: int = 1,
    max_frames: int = 100,
    conf_threshold: float = 1.0,
    foreground_conf_threshold: float = 0.1,
    point_size: float = 0.001,
    camera_frustum_scale: float = 0.02,
    no_mask: bool = False,
    xyzw: bool = True,
    axes_scale: float = 0.25,
    bg_downsample_factor: int = 1,
) -> None:
    # if data_path[0] has subfolders, process each subfolder
    if data_paths[0].is_dir():
        new_data_paths = sorted([subfolder for subfolder in data_paths[0].iterdir() if subfolder.is_dir()])
    
    if len(new_data_paths) > 0:
        data_paths = new_data_paths

    for data_path in data_paths:
        process_folder(
            data_path,
            downsample_factor,
            max_frames,
            conf_threshold,
            foreground_conf_threshold,
            point_size,
            camera_frustum_scale,
            no_mask,
            xyzw,
            axes_scale,
            bg_downsample_factor,
            output_dir,
        )

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process input arguments.")

    # Define arguments
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to the data folders (can specify multiple paths)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./viser_result"),
        help="Output directory for recordings",
    )
    parser.add_argument(
        "--conf_thre",
        type=float,
        default=0.1,
        help="Confidence threshold, default is 0.1",
    )
    parser.add_argument(
        "--fg_conf_thre",
        type=float,
        default=0.5,
        help="Foreground confidence threshold, default is 0.0",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.001,
        help="Point size, default is 0.001",
    )
    parser.add_argument(
        "--camera_size",
        type=float,
        default=0.015,
        help="Camera frustum scale, default is 0.015",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Don't use mask to filter out points",
    )
    parser.add_argument(
        "--wxyz",
        action="store_true",
        help="Use wxyz for SO3 representation",
    )
    parser.add_argument(
        "--axes_scale",
        type=float,
        default=0.1,
        help="Scale of axes",
    )
    parser.add_argument(
        "--bg_downsample",
        type=int,
        default=1,
        help="Background downsample factor",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsample factor",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to process",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        data_paths=args.data,
        output_dir=args.output_dir,
        conf_threshold=args.conf_thre,
        foreground_conf_threshold=args.fg_conf_thre,
        point_size=args.point_size,
        camera_frustum_scale=args.camera_size,
        no_mask=args.no_mask,
        xyzw=not args.wxyz,
        axes_scale=args.axes_scale,
        bg_downsample_factor=args.bg_downsample,
        downsample_factor=args.downsample,
        max_frames=args.max_frames,
    )
