import time
import sys
import argparse
from pathlib import Path

import numpy as onp
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf
import matplotlib.cm as cm  # For colormap

def main(
    data_path: Path = Path("./demo_tmp/NULL"),
    downsample_factor: int = 1,
    max_frames: int = 100,
    share: bool = False,
    conf_threshold: float = 1.0,
    foreground_conf_threshold: float = 0.1,
    point_size: float = 0.001,
    camera_frustum_scale: float = 0.02,
    no_mask: bool = False,
    xyzw: bool = True,
    axes_scale: float = 0.25,
    bg_downsample_factor: int = 1,
    init_conf: bool = False,
    cam_thickness: float = 1.5,
) -> None:
    from pathlib import Path  # <-- Import Path here if not already imported
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction('-z')
    if no_mask:             # not using dynamic / static mask
        init_conf = True    # must use init_conf map, to avoid depth cleaning
        fg_conf_thre = conf_threshold # now fg_conf_thre is the same as conf_thre
    print("Loading frames!")
    loader = viser.extras.Record3dLoader_Customized(
        data_path,
        conf_threshold=conf_threshold,
        foreground_conf_threshold=foreground_conf_threshold,
        no_mask=no_mask,
        xyzw=xyzw,
        init_conf=init_conf,
    )
    num_frames = min(max_frames, loader.num_frames())

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,  # Initially disabled
        )

    # Add recording UI.
    with server.gui.add_folder("Recording"):
        gui_record_scene = server.gui.add_button("Record Scene")

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider
        if gui_show_all_frames.value:
            # Show frames with stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)
            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            # Show only the current frame
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == current_timestep
            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Update frame visibility based on new stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)

    # Recording handler
    @gui_record_scene.on_click
    def _(_):
        gui_record_scene.disabled = True

        # Save the original frame visibility state
        original_visibility = [frame_node.visible for frame_node in frame_nodes]

        rec = server._start_scene_recording()
        rec.set_loop_start()
        
        # Determine sleep duration based on current FPS
        sleep_duration = 1.0 / gui_framerate.value if gui_framerate.value > 0 else 0.033  # Default to ~30 FPS
        
        if gui_show_all_frames.value:
            # Record all frames according to the stride
            stride = gui_stride.value
            frames_to_record = [i for i in range(num_frames) if i % stride == 0]
        else:
            # Record the frames in sequence
            frames_to_record = range(num_frames)
        
        for t in frames_to_record:
            # Update the scene to show frame t
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i == t) if not gui_show_all_frames.value else (i % gui_stride.value == 0)
            server.flush()
            rec.insert_sleep(sleep_duration)

        # set all invisible
        with server.atomic():
            for frame_node in frame_nodes:
                frame_node.visible = False
        
        # Finish recording
        bs = rec.end_and_serialize()
        
        # Save the recording to a file
        output_path = Path(f"./viser_result/recording_{str(data_path).split('/')[-1]}.viser")
        # make sure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bs)
        print(f"Recording saved to {output_path.resolve()}")
        
        # Restore the original frame visibility state
        with server.atomic():
            for frame_node, visibility in zip(frame_nodes, original_visibility):
                frame_node.visible = visibility
        server.flush()
        
        gui_record_scene.disabled = False

    # Load in frames.
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

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position,
            colors=color,
            point_size=point_size,
            point_shape="rounded",
        )

        # Compute color for frustum based on frame index.
        norm_i = i / (num_frames - 1) if num_frames > 1 else 0  # Normalize index to [0, 1]
        color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components

        # Place the frustum with the computed color.
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
            thickness=cam_thickness,
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 10,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    # Initialize frame visibility.
    for i, frame_node in enumerate(frame_nodes):
        if gui_show_all_frames.value:
            frame_node.visible = (i % gui_stride.value == 0)
        else:
            frame_node.visible = i == gui_timestep.value

    # Add background frame.
    bg_positions = onp.concatenate(bg_positions, axis=0)
    bg_colors = onp.concatenate(bg_colors, axis=0)
    server.scene.add_point_cloud(
        name=f"/frames/background",
        points=bg_positions,
        colors=bg_colors,
        point_size=point_size,
        point_shape="rounded",
    )

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value and not gui_show_all_frames.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process input arguments.")

    # Define arguments
    parser.add_argument(
        "--data",
        type=Path,
        nargs="?",
        default=Path("./demo_tmp/NULL"),
        help="Path to the data"
    )
    parser.add_argument(
        "--conf_thre",
        type=float,
        default=0.1,
        help="Confidence threshold, default is 1.0"
    )
    parser.add_argument(
        "--fg_conf_thre",
        type=float,
        default=0.0,
        help="Foreground confidence threshold, default is 0.1"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.001,
        help="Point size, default is 0.001"
    )
    parser.add_argument(
        "--camera_size",
        type=float,
        default=0.015,
        help="Camera frustum scale, default is 0.02"
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
        default=1,
        help="Downsample factor",
    )
    parser.add_argument(
        "--init_conf",
        action="store_true",
        help="Share the scene",
    )
    parser.add_argument(
        "--cam_thickness",
        type=float,
        default=1.5,
        help="Camera frustum thickness",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    tyro.cli(main(
        data_path=args.data,
        conf_threshold=args.conf_thre,
        foreground_conf_threshold=args.fg_conf_thre,
        point_size=args.point_size,
        camera_frustum_scale=args.camera_size,
        no_mask=args.no_mask,
        xyzw=not args.wxyz,
        axes_scale=args.axes_scale,
        bg_downsample_factor=args.bg_downsample,
        downsample_factor=args.downsample,
        init_conf=args.init_conf,
        cam_thickness=args.cam_thickness,
    ))
