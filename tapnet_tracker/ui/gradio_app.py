#!/usr/bin/env python3
"""
Gradio web interface for TAPNet Tracker.

Contains the complete web UI implementation with all features including
video processing, interactive editing, and visualization.
"""

import gradio as gr
import os
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw

# Import core components
from ..core.tracker import TAPNextTracker
from ..core.config import config
from ..interactive.semantic_info import SemanticInfoManager
from ..interactive.edit_mode import TrackEditManager
from ..utils.file_utils import get_video_choices, format_video_list_display


def create_tapnext_app():
    """Create TAPNext trajectory generator Gradio application"""
    
    # Create global instances
    tracker = TAPNextTracker()
    semantic_manager = SemanticInfoManager()
    edit_manager = TrackEditManager()
    
    # Application styles
    css = """
    .status-info {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        background-color: #f8f9fa;
    }
    
    .large-text {
        font-size: 18px !important;
        font-weight: 500;
    }
    
    .success-box {
        background-color: #d4edda !important;
        border-color: #c3e6cb !important;
        color: #155724 !important;
    }
    
    .warning-box {
        background-color: #fff3cd !important;
        border-color: #ffeaa7 !important;
        color: #856404 !important;
    }
    
    .error-box {
        background-color: #f8d7da !important;
        border-color: #f5c6cb !important;
        color: #721c24 !important;
    }
    
    .compact-row {
        margin: 4px 0 !important;
    }
    
    .center-content {
        text-align: center;
    }
    """

    # Create Gradio interface
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=css,
        title="🎯 TAPNext Trajectory Generator"
    ) as app:
        
        # Application title
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="color: #2E86AB; margin-bottom: 10px;">🎯 TAPNext Trajectory Generator</h1>
            <p style="color: #666; font-size: 16px; margin: 0;">
                High-precision video trajectory generation and interactive editing platform | Supports batch processing and visualization
            </p>
        </div>
        """)

        # ================== Main Interface Tabs ==================
        with gr.Tabs() as main_tabs:
            
            # ================== Trajectory Generation Tab ==================
            with gr.Tab("🎬 Trajectory Generation", id="generation_tab"):
                
                with gr.Row():
                    # Left side: Model and file selection
                    with gr.Column(scale=1):
                        
                        # Model loading area
                        with gr.Group():
                            gr.HTML('<h3 style="margin-top: 0;">🤖 Model Management</h3>')
                            
                            checkpoint_path = gr.Textbox(
                                label="Model Path",
                                value=config.DEFAULT_CHECKPOINT_PATH,
                                placeholder="Enter model checkpoint file path",
                                info="Path to TAPNext model .npz file"
                            )
                            
                            load_model_btn = gr.Button(
                                "🔄 Load Model",
                                variant="primary",
                                size="lg"
                            )
                            
                            model_status = gr.Markdown(
                                "❌ **Model Status**: Not loaded",
                                elem_classes=["status-info"]
                            )
                        
                        # Video selection area
                        with gr.Group():
                            gr.HTML('<h3>📁 Video Selection</h3>')
                            
                            with gr.Tabs():
                                # Single video upload
                                with gr.Tab("📤 Upload Video"):
                                    video_upload = gr.File(
                                        label="Select video file",
                                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv"],
                                        type="filepath"
                                    )
                                
                                # Folder batch processing
                                with gr.Tab("📂 Folder Processing"):
                                    folder_path = gr.Textbox(
                                        label="Video folder path",
                                        placeholder="/home/user/tapnet/videos",
                                        info="Path to folder containing video files"
                                    )
                                    
                                    scan_folder_btn = gr.Button(
                                        "🔍 Scan Folder",
                                        variant="secondary"
                                    )
                                    
                                    video_list_display = gr.Markdown(
                                        "📁 Please select a folder containing video files",
                                        elem_classes=["status-info"]
                                    )
                                    
                                    video_choice = gr.Dropdown(
                                        label="Select video to process",
                                        choices=[],
                                        interactive=True,
                                        info="Select one video from scanned files to process"
                                    )
                        
                        # Parameter configuration area
                        with gr.Group():
                            gr.HTML('<h3>⚙️ Generation Parameters</h3>')
                            

                            
                            point_method = gr.Radio(
                                choices=["manual"],
                                value="manual",
                                label="Point generation method",
                                info="manual: Manually click to select trajectory points"
                            )
                            
                            enable_visualization = gr.Checkbox(
                                value=True,
                                label="🎨 Generate visualization video",
                                info="Generate visualization video with trajectory annotations"
                            )
                            
                            output_dir = gr.Textbox(
                                label="💾 Base output directory",
                                value=config.DEFAULT_OUTPUT_DIR,
                                placeholder="./outputs",
                                info="Directory for saving trajectory files and visualization videos 💡 Files for each video will be saved in outputs/{video_name}/ subfolder"
                            )
                    
                    # Right side: Preview and operations
                    with gr.Column(scale=2):
                        
                        # Video preview and manual point selection
                        with gr.Group():
                            gr.HTML('<h3 style="margin-top: 0;">🖼️ Video Preview</h3>')
                            
                            # Manual point selection instructions
                            manual_mode_info = gr.Markdown(
                                """
                                **📍 Manual Point Selection Instructions**:
                                - Click on the image below to select trajectory starting points
                                - Selected points will be displayed as red markers
                                - Click "Clear Selection" to reselect
                                - Recommended to select obvious feature points in the image
                                """,
                                visible=False,
                                elem_classes=["status-info", "warning-box"]
                            )
                            
                            # Image preview area
                            image_preview = gr.Image(
                                label="Click image to select trajectory points",
                                type="pil",
                                interactive=True,
                                height=400
                            )
                            
                            # Manual point selection controls
                            with gr.Row():
                                clear_points_btn = gr.Button(
                                    "🗑️ Clear Selection",
                                    variant="secondary",
                                    visible=False
                                )
                                
                                manual_points_info = gr.Markdown(
                                    "**Selected points**: 0",
                                    visible=False
                                )
                        
                        # Semantic information editing area
                        with gr.Group():
                            gr.HTML('<h3>📝 Trajectory Point Semantic Information</h3>')
                            
                            # Current selected point information
                            current_point_info = gr.Markdown(
                                "💡 After clicking on image to select trajectory points, you can add semantic descriptions for that point",
                                elem_classes=["status-info"]
                            )
                            
                            # Semantic information editing
                            with gr.Row():
                                semantic_text_input = gr.Textbox(
                                    label="Semantic description",
                                    placeholder="Enter semantic description for this trajectory point...",
                                    lines=2,
                                    interactive=False
                                )
                            
                            with gr.Row():
                                save_semantic_point_btn = gr.Button(
                                    "💾 Save Current Point Semantic",
                                    variant="secondary",
                                    interactive=False,
                                    size="sm"
                                )
                                clear_semantic_point_btn = gr.Button(
                                    "🗑️ Clear Current Point Semantic",
                                    variant="secondary",
                                    interactive=False,
                                    size="sm"
                                )
                            
                            with gr.Row():
                                delete_current_point_btn = gr.Button(
                                    "❌ Delete Current Point",
                                    variant="secondary",
                                    interactive=False,
                                    size="sm"
                                )
                                export_semantic_generation_btn = gr.Button(
                                    "📄 Export Semantic Information",
                                    variant="secondary", 
                                    interactive=False,
                                    size="sm"
                                )
                            
                            # Semantic information status
                            semantic_status = gr.Markdown(
                                "",
                                elem_classes=["status-info"]
                            )
                            
                            # Selected points list - clickable selection
                            with gr.Group():
                                gr.HTML('<h4>📍 Point List (Click to Select for Editing)</h4>')
                                selected_points_list = gr.Markdown(
                                    "",
                                    elem_classes=["status-info"]
                                )
                                
                                # Hidden point selection component
                                point_selector = gr.Radio(
                                    choices=[],
                                    label="Select point to edit",
                                    visible=False,
                                    interactive=True
                                )
                        
                        # Generation button and status
                        with gr.Group():
                            generate_btn = gr.Button(
                                "🚀 Generate Trajectories",
                                variant="primary",
                                size="lg",
                                elem_classes=["large-text"]
                            )
                            
                            progress_bar = gr.Progress()
                            
                            generation_status = gr.Markdown(
                                "💡 **Please load model and select video first**",
                                elem_classes=["status-info"]
                            )
                        
                        # Results display
                        with gr.Group():
                            gr.HTML('<h3>📊 Generation Results</h3>')
                            
                            # Result files
                            with gr.Row():
                                result_file = gr.File(
                                    label="📁 Trajectory file (.pth)",
                                    visible=False
                                )
                                
                                viz_file = gr.File(
                                    label="🎨 Visualization video (.mp4)",
                                    visible=False
                                )
                            
                            # Preview video playback
                            preview_video = gr.Video(
                                label="🎬 Trajectory Preview",
                                visible=False,
                                height=300
                            )

            # ================== Interactive Editing Tab ==================
            with gr.Tab("✏️ Interactive Editing", id="editing_tab"):
                
                with gr.Row():
                    # Left side: Edit controls
                    with gr.Column(scale=1):
                        
                        # File selection
                        with gr.Group():
                            gr.HTML('<h3 style="margin-top: 0;">📂 File Selection</h3>')
                            
                            edit_video_file = gr.File(
                                label="Select video file",
                                file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv"],
                                type="filepath"
                            )
                            
                            edit_tracks_file = gr.File(
                                label="Select trajectory file",
                                file_types=[".pth"],
                                type="filepath"
                            )
                            
                            auto_fill_btn = gr.Button(
                                "📋 Auto-fill last generated files",
                                variant="primary",
                                size="lg",
                                visible=True  # Ensure button is visible
                            )
                            
                            enter_edit_btn = gr.Button(
                                "✏️ Enter Edit Mode", 
                                variant="secondary",
                                size="lg"
                            )
                            
                            edit_status = gr.Markdown(
                                "📝 **Please select video and trajectory files**",
                                elem_classes=["status-info"]
                            )
                        
                        # Frame navigation
                        with gr.Group():
                            gr.HTML('<h3>🎞️ Frame Navigation</h3>')
                            
                            frame_slider = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                label="Frame position",
                                interactive=False
                            )
                            
                            with gr.Row():
                                prev_frame_btn = gr.Button(
                                    "⬅️ Previous Frame",
                                    size="sm",
                                    interactive=False
                                )
                                
                                next_frame_btn = gr.Button(
                                    "➡️ Next Frame",
                                    size="sm",
                                    interactive=False
                                )
                            
                            frame_info = gr.Markdown(
                                "**Frame info**: Not loaded",
                                elem_classes=["status-info"]
                            )
                        
                        # Trajectory points list
                        with gr.Group():
                            gr.HTML('<h3>📍 Trajectory Points List</h3>')
                            
                            points_list = gr.Dropdown(
                                label="Select trajectory point",
                                choices=[],
                                interactive=False,
                                info="Click on points in the list for quick selection"
                            )
                            
                            points_info = gr.Markdown(
                                "📋 No trajectory point information available",
                                elem_classes=["status-info"]
                            )
                        
                        # Edit operations
                        with gr.Group():
                            gr.HTML('<h3>🛠️ Edit Operations</h3>')
                            
                            with gr.Row():
                                clear_selection_btn = gr.Button(
                                    "🔄 Clear Selection",
                                    size="sm",
                                    interactive=False
                                )
                                
                                toggle_visibility_btn = gr.Button(
                                    "👁️ Toggle Visibility",
                                    size="sm",
                                    interactive=False
                                )
                            
                            # Semantic annotation
                            semantic_text = gr.Textbox(
                                label="Semantic annotation",
                                placeholder="Add semantic description for selected trajectory point...",
                                lines=3,
                                interactive=False
                            )
                            
                            save_semantic_btn = gr.Button(
                                "💾 Save Semantic Information",
                                variant="secondary",
                                size="sm",
                                interactive=False
                            )
                        
                        # Save and export
                        with gr.Group():
                            gr.HTML('<h3>💾 Save & Export</h3>')
                            
                            save_tracks_btn = gr.Button(
                                "💾 Save Edited Trajectories",
                                variant="primary",
                                interactive=False
                            )
                            
                            export_semantic_btn = gr.Button(
                                                                        "📄 Export Semantic Information (JSON)",
                                variant="secondary",
                                interactive=False
                            )
                            
                            visualize_edited_btn = gr.Button(
                                "🎨 Generate Edited Visualization",
                                variant="secondary",
                                interactive=False
                            )
                            

                    
                    # Right side: Edit interface
                    with gr.Column(scale=2):
                        
                        # Edit image
                        with gr.Group():
                            gr.HTML('<h3 style="margin-top: 0;">🖼️ Edit Interface</h3>')
                            
                            edit_instructions = gr.Markdown(
                                """
                                **📝 Edit Instructions**:
                                - 🎯 Click trajectory points to select
                                - 🖱️ Click empty space to move selected point
                                - 👁️ Use "Toggle Visibility" button to control point display
                                - 🏷️ Add semantic annotations for each point
                                """,
                                elem_classes=["status-info"]
                            )
                            
                            edit_image = gr.Image(
                                label="Click to select or move trajectory points",
                                type="pil",
                                interactive=False,
                                height=500
                            )
                            
                            edit_operation_status = gr.Markdown(
                                "💡 **Please enter edit mode first**",
                                elem_classes=["status-info"]
                            )
                            
                            # Visualization preview video
                            edited_visualization_video = gr.Video(
                                label="🎬 Edited Visualization Preview",
                                visible=False,
                                show_download_button=True,
                                interactive=False,
                                height=400
                            )
                        
                        # Edit results
                        with gr.Group():
                            gr.HTML('<h3>📊 Edit Results</h3>')
                            
                            with gr.Row():
                                edited_tracks_file = gr.File(
                                    label="📁 Edited trajectory file",
                                    visible=False
                                )
                                
                                semantic_export_file = gr.File(
                                    label="📄 Semantic information file",
                                    visible=False
                                )
                                
                                edited_viz_file = gr.File(
                                    label="🎨 Edited visualization video",
                                    visible=False
                                )

        # ================== Event Handler Functions ==================
        
        def load_model_handler(checkpoint_path):
            """Load model handler function"""
            message, success = tracker.load_model(checkpoint_path)
            if success:
                return gr.update(value=f"✅ **Model Status**: Loaded\n\n{message}", elem_classes=["status-info", "success-box"])
            else:
                return gr.update(value=f"❌ **Model Status**: Loading failed\n\n{message}", elem_classes=["status-info", "error-box"])
        
        def scan_folder_handler(folder_path):
            """Scan folder handler function"""
            if not folder_path:
                return (
                    gr.update(value="❌ Please provide folder path", elem_classes=["status-info", "error-box"]),
                    gr.update(choices=[], value=None)
                )
            
            success, message, video_files = tracker.scan_folder(folder_path)
            
            if success:
                display_text = format_video_list_display(video_files)
                choices = get_video_choices(video_files)
                return (
                    gr.update(value=display_text, elem_classes=["status-info", "success-box"]),
                    gr.update(choices=choices, value=None if not choices else choices[0])
                )
            else:
                return (
                    gr.update(value=message, elem_classes=["status-info", "error-box"]),
                    gr.update(choices=[], value=None)
                )
        
        def video_selection_handler(choice):
            """Video selection handler function"""
            nonlocal manual_selected_points, points_semantic_info, current_selected_point_index
            
            # Clear previous edit state
            manual_selected_points = []
            points_semantic_info = {}
            current_selected_point_index = None
            
            if not choice:
                return (
                    None,  # image_preview
                    "**Selected points**: 0",  # manual_points_info
                    "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                    "",  # semantic_text_input
                    gr.update(interactive=False),  # semantic_text_input interactive
                    gr.update(interactive=False),  # save_semantic_point_btn
                    gr.update(interactive=False),  # clear_semantic_point_btn
                    gr.update(interactive=False),  # delete_current_point_btn
                    gr.update(interactive=False),  # export_semantic_generation_btn
                    "",  # semantic_status
                    "",  # selected_points_list
                    gr.update(choices=[], visible=False)  # point_selector
                )
            
            video_path = tracker.get_video_path_by_choice(choice)
            if video_path:
                try:
                    from ..utils.video_utils import extract_first_frame
                    frame, original_size = extract_first_frame(video_path)
                    frame_pil = Image.fromarray(frame)
                    return (
                        frame_pil,  # image_preview
                        "**Selected points**: 0",  # manual_points_info
                        "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                        "",  # semantic_text_input
                        gr.update(interactive=False),  # semantic_text_input interactive
                        gr.update(interactive=False),  # save_semantic_point_btn
                        gr.update(interactive=False),  # clear_semantic_point_btn
                        gr.update(interactive=False),  # delete_current_point_btn
                        gr.update(interactive=False),  # export_semantic_generation_btn
                        "",  # semantic_status
                        "",  # selected_points_list
                        gr.update(choices=[], visible=False)  # point_selector
                    )
                except Exception as e:
                    print(f"Failed to preview first frame of video: {e}")
                    return (
                        None,  # image_preview
                        "**Selected points**: 0",  # manual_points_info
                        "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                        "",  # semantic_text_input
                        gr.update(interactive=False),  # semantic_text_input interactive
                        gr.update(interactive=False),  # save_semantic_point_btn
                        gr.update(interactive=False),  # clear_semantic_point_btn
                        gr.update(interactive=False),  # delete_current_point_btn
                        gr.update(interactive=False),  # export_semantic_generation_btn
                        "",  # semantic_status
                        "",  # selected_points_list
                        gr.update(choices=[], visible=False)  # point_selector
                    )
            return (
                None,  # image_preview
                "**Selected points**: 0",  # manual_points_info
                "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                "",  # semantic_text_input
                gr.update(interactive=False),  # semantic_text_input interactive
                gr.update(interactive=False),  # save_semantic_point_btn
                gr.update(interactive=False),  # clear_semantic_point_btn
                gr.update(interactive=False),  # delete_current_point_btn
                gr.update(interactive=False),  # export_semantic_generation_btn
                "",  # semantic_status
                "",  # selected_points_list
                gr.update(choices=[], visible=False)  # point_selector
            )
        
        def point_method_change_handler(method):
            """Point generation method change handler function"""
            if method == "manual":
                return (
                    gr.update(visible=True),  # manual_mode_info
                    gr.update(visible=True),  # clear_points_btn
                    gr.update(visible=True)   # manual_points_info
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        # Handle manual point selection and semantic information
        manual_selected_points = []
        points_semantic_info = {}  # Store semantic information for each point {point_index: semantic_text}
        current_selected_point_index = None  # Current selected point index
        
        def generate_points_list_text():
            """Generate points list text"""
            if not manual_selected_points:
                return ""
            
            lines = ["**📍 Selected trajectory points (click radio button below to select for editing):**"]
            for i, (x, y) in enumerate(manual_selected_points):
                semantic = points_semantic_info.get(i, "")
                status = "✅" if semantic.strip() else "⚪"
                current_marker = "🎯" if i == current_selected_point_index else "  "
                semantic_preview = semantic[:20] + "..." if len(semantic) > 20 else semantic
                lines.append(f"{current_marker}{status} **Point {i}**: ({x}, {y}) - {semantic_preview or '(no description)'}")
            
            return "\n".join(lines)
        
        def generate_point_choices():
            """Generate point selection choices"""
            if not manual_selected_points:
                return []
            
            choices = []
            for i, (x, y) in enumerate(manual_selected_points):
                semantic = points_semantic_info.get(i, "")
                status = "✅" if semantic.strip() else "⚪"
                semantic_preview = semantic[:15] + "..." if len(semantic) > 15 else semantic
                choices.append(f"{status} Point {i}: ({x}, {y}) - {semantic_preview or '(no description)'}")
            
            return choices

        def manual_image_click_handler(evt: gr.SelectData, current_point_method, current_image):
            """Manual image click handler function"""
            nonlocal manual_selected_points
            
            print(f"🖱️ Image click event: method={current_point_method}, current selected points={len(manual_selected_points)}")
            
            # If no image available, return original image
            if current_image is None:
                print("❌ No image available")
                return current_image, "❌ Please select video file first"
            
            if current_point_method != "manual":
                print(f"❌ Not in manual mode: {current_point_method}")
                return current_image, "📍 Currently not in manual selection mode"
            
            # Add clicked point
            click_x, click_y = evt.index[0], evt.index[1]
            manual_selected_points.append((click_x, click_y))
            print(f"✅ Added point: ({click_x}, {click_y}), total points: {len(manual_selected_points)}")
            
            # Debug: Show image and click information
            print(f"🖼️ Image information:")
            print(f"   Image size: {current_image.size}")
            print(f"   Click coordinates: ({click_x}, {click_y})")
            print(f"   Relative position: x={click_x/current_image.size[0]:.3f}, y={click_y/current_image.size[1]:.3f}")
            
            # Draw selected points on image
            try:
                img = current_image.copy()
                draw = ImageDraw.Draw(img)
                
                for i, (x, y) in enumerate(manual_selected_points):
                    # Draw red circle
                    radius = 8
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill="red", outline="white", width=2)
                    # Draw point number
                    draw.text((x+10, y-10), str(i), fill="white")
                
                # Update current selected point
                nonlocal current_selected_point_index
                current_selected_point_index = len(manual_selected_points) - 1
                
                # Generate points list information and update semantic editing interface
                points_list_text = generate_points_list_text()
                point_choices = generate_point_choices()
                point_info_text = f"🎯 **Current selection**: Point {current_selected_point_index}"
                current_semantic = points_semantic_info.get(current_selected_point_index, "")
                
                return (
                    img, 
                    f"**Selected points**: {len(manual_selected_points)}",
                    point_info_text,  # current_point_info
                    current_semantic,  # semantic_text_input
                    gr.update(interactive=True),  # semantic_text_input
                    gr.update(interactive=True),  # save_semantic_point_btn
                    gr.update(interactive=True),  # clear_semantic_point_btn
                    gr.update(interactive=True),  # delete_current_point_btn
                    gr.update(interactive=len(manual_selected_points) > 0),  # export_semantic_generation_btn
                    "",  # semantic_status
                    points_list_text,  # selected_points_list
                    gr.update(choices=point_choices, value=point_choices[current_selected_point_index] if point_choices else None, visible=len(manual_selected_points) > 0)  # point_selector
                )
            except Exception as e:
                print(f"Error drawing point selection: {e}")
                return (
                    current_image, 
                    f"**Selected points**: {len(manual_selected_points)}",
                    "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",
                    "",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "",
                    "",
                    gr.update(choices=[], visible=False)
                )
        
        def clear_points_handler(current_image):
            """Clear selected points"""
            nonlocal manual_selected_points, points_semantic_info, current_selected_point_index
            manual_selected_points = []
            points_semantic_info = {}
            current_selected_point_index = None
            
            # Restore original image
            return (
                current_image, 
                "**Selected points**: 0",
                "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                "",  # semantic_text_input
                gr.update(interactive=False),  # semantic_text_input
                gr.update(interactive=False),  # save_semantic_point_btn
                gr.update(interactive=False),  # clear_semantic_point_btn
                gr.update(interactive=False),  # delete_current_point_btn
                gr.update(interactive=False),  # export_semantic_generation_btn
                "",  # semantic_status
                "",  # selected_points_list
                gr.update(choices=[], visible=False)  # point_selector
            )
        
        # Semantic information handler functions
        def save_semantic_point_handler(semantic_text):
            """Save semantic information for current point"""
            nonlocal points_semantic_info, current_selected_point_index
            
            if current_selected_point_index is None:
                return (
                    "❌ Please select a trajectory point first",
                    generate_points_list_text(),
                    gr.update()
                )
            
            points_semantic_info[current_selected_point_index] = semantic_text.strip()
            
            return (
                f"✅ Saved semantic information for point {current_selected_point_index}",
                generate_points_list_text(),
                gr.update(choices=generate_point_choices(), value=generate_point_choices()[current_selected_point_index] if generate_point_choices() else None)
            )
        
        def clear_semantic_point_handler():
            """Clear semantic information for current point"""
            nonlocal points_semantic_info, current_selected_point_index
            
            if current_selected_point_index is None:
                return (
                    "❌ Please select a trajectory point first",
                    "",
                    generate_points_list_text(),
                    gr.update()
                )
            
            points_semantic_info[current_selected_point_index] = ""
            
            return (
                f"✅ Cleared semantic information for point {current_selected_point_index}",
                "",  # Clear input box
                generate_points_list_text(),
                gr.update(choices=generate_point_choices(), value=generate_point_choices()[current_selected_point_index] if generate_point_choices() else None)
            )
        
        def delete_current_point_handler():
            """Delete currently selected point"""
            nonlocal manual_selected_points, points_semantic_info, current_selected_point_index
            
            if current_selected_point_index is None:
                return (
                    "❌ Please select a trajectory point first",
                    generate_points_list_text(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    ""
                )
            
            # Delete point
            del manual_selected_points[current_selected_point_index]
            
            # Reorganize semantic information (shift indices forward)
            new_semantic_info = {}
            for old_idx, semantic in points_semantic_info.items():
                if old_idx < current_selected_point_index:
                    new_semantic_info[old_idx] = semantic
                elif old_idx > current_selected_point_index:
                    new_semantic_info[old_idx - 1] = semantic
                # Don't copy deleted index
            
            points_semantic_info = new_semantic_info
            
            # Update current selected point
            if len(manual_selected_points) == 0:
                current_selected_point_index = None
                current_point_info_text = "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point"
                current_semantic = ""
                interactive_state = False
                point_choices = []
                visible_state = False
                selected_value = None
            else:
                if current_selected_point_index >= len(manual_selected_points):
                    current_selected_point_index = len(manual_selected_points) - 1
                current_point_info_text = f"🎯 **Current selection**: Point {current_selected_point_index}"
                current_semantic = points_semantic_info.get(current_selected_point_index, "")
                interactive_state = True
                point_choices = generate_point_choices()
                visible_state = True
                selected_value = point_choices[current_selected_point_index] if point_choices else None
            
            return (
                f"✅ Point deleted, {len(manual_selected_points)} points remaining",
                generate_points_list_text(),
                current_point_info_text,
                current_semantic,
                gr.update(interactive=interactive_state),  # semantic_text_input
                gr.update(interactive=interactive_state),  # save_semantic_point_btn
                gr.update(interactive=interactive_state),  # clear_semantic_point_btn
                gr.update(interactive=interactive_state),  # delete_current_point_btn
                gr.update(choices=point_choices, value=selected_value, visible=visible_state)
            )
        
        def point_selector_handler(selected_point):
            """Point selector handler function"""
            nonlocal current_selected_point_index
            
            if not selected_point or not manual_selected_points:
                return (
                    "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",
                    ""
                )
            
            # Extract point index from selected text
            try:
                # Format: "✅ Point 0: (100, 200) - description" or "⚪ Point 1: (150, 250) - (no description)"
                import re
                match = re.search(r'Point (\d+):', selected_point)
                if match:
                    current_selected_point_index = int(match.group(1))
                    current_semantic = points_semantic_info.get(current_selected_point_index, "")
                    point_info_text = f"🎯 **Current selection**: Point {current_selected_point_index}"
                    
                    return (
                        point_info_text,
                        current_semantic,
                        generate_points_list_text()
                    )
            except:
                pass
            
            return (
                "❌ Unable to parse selected point",
                "",
                generate_points_list_text()
            )
        
        def export_semantic_generation_handler():
            """Export semantic information from trajectory generation stage"""
            if not manual_selected_points:
                return "❌ No trajectory points selected"
            
            # Get current video information
            try:
                # Need to get current selected video path and information
                video_path = None
                if hasattr(tracker, 'last_video_path') and tracker.last_video_path:
                    video_path = tracker.last_video_path
                
                if not video_path:
                    return "❌ Unable to get video file information, please generate trajectories first"
                
                # Initialize semantic manager and export
                from ..interactive.semantic_info import SemanticInfoManager
                temp_semantic_manager = SemanticInfoManager()
                temp_semantic_manager.initialize(
                    num_points=len(manual_selected_points),
                    video_path=video_path,
                    output_dir=config.ensure_output_dir()
                )
                
                # Set semantic information
                for point_id, semantic_text in points_semantic_info.items():
                    temp_semantic_manager.set_semantic_info(point_id, semantic_text)
                
                # Export JSON
                file_path, message = temp_semantic_manager.export_semantic_info_to_json()
                
                if file_path:
                    return f"✅ **Semantic information exported successfully**\n\n📁 File saved to: {file_path}"
                else:
                    return f"❌ **Export failed**: {message}"
                    
            except Exception as e:
                return f"❌ **Export failed**: {str(e)}"

        def video_upload_handler(video):
            """Video upload handler function"""
            nonlocal manual_selected_points, points_semantic_info, current_selected_point_index
            
            # Clear previous edit state
            manual_selected_points = []
            points_semantic_info = {}
            current_selected_point_index = None
            
            if video is None:
                return (
                    None,  # image_preview
                    "**Selected points**: 0",  # manual_points_info
                    "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                    "",  # semantic_text_input
                    gr.update(interactive=False),  # semantic_text_input interactive
                    gr.update(interactive=False),  # save_semantic_point_btn
                    gr.update(interactive=False),  # clear_semantic_point_btn
                    gr.update(interactive=False),  # delete_current_point_btn
                    gr.update(interactive=False),  # export_semantic_generation_btn
                    "",  # semantic_status
                    "",  # selected_points_list
                    gr.update(choices=[], visible=False)  # point_selector
                )
            
            try:
                from ..utils.video_utils import extract_first_frame
                frame, original_size = extract_first_frame(video)
                frame_pil = Image.fromarray(frame)
                return (
                    frame_pil,  # image_preview
                    "**Selected points**: 0",  # manual_points_info
                    "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                    "",  # semantic_text_input
                    gr.update(interactive=False),  # semantic_text_input interactive
                    gr.update(interactive=False),  # save_semantic_point_btn
                    gr.update(interactive=False),  # clear_semantic_point_btn
                    gr.update(interactive=False),  # delete_current_point_btn
                    gr.update(interactive=False),  # export_semantic_generation_btn
                    "",  # semantic_status
                    "",  # selected_points_list
                    gr.update(choices=[], visible=False)  # point_selector
                )
            except Exception as e:
                print(f"Failed to process uploaded video: {e}")
                return (
                    None,  # image_preview
                    "**Selected points**: 0",  # manual_points_info
                    "💡 After clicking image to select trajectory points, you can add semantic descriptions for that point",  # current_point_info
                    "",  # semantic_text_input
                    gr.update(interactive=False),  # semantic_text_input interactive
                    gr.update(interactive=False),  # save_semantic_point_btn
                    gr.update(interactive=False),  # clear_semantic_point_btn
                    gr.update(interactive=False),  # delete_current_point_btn
                    gr.update(interactive=False),  # export_semantic_generation_btn
                    "",  # semantic_status
                    "",  # selected_points_list
                    gr.update(choices=[], visible=False)  # point_selector
                )
        
        def generate_tracks_handler(video, selected_video_choice, point_method, enable_viz, output_dir, current_image):
            """Generate tracks handler function"""
            nonlocal manual_selected_points
            
            print(f"🚀 Starting trajectory generation...")
            print(f"  - point_method: {point_method}")
            print(f"  - manual_selected_points length: {len(manual_selected_points)}")
            print(f"  - manual_selected_points content: {manual_selected_points}")
            
            if not tracker.model_loaded:
                return (
                    gr.update(value="❌ **Please load model first**", elem_classes=["status-info", "error-box"]),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            # Determine video path
            video_path = None
            filename = None
            
            if video:  # Uploaded video
                video_path = video
                filename = os.path.basename(video)
            elif selected_video_choice:  # Video selected from folder
                video_path = tracker.get_video_path_by_choice(selected_video_choice)
                filename = selected_video_choice.split(' ', 1)[-1] if ' ' in selected_video_choice else selected_video_choice
            
            if not video_path:
                return (
                    gr.update(value="❌ **Please select video file**", elem_classes=["status-info", "error-box"]),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            # Prepare manual points (if needed)
            manual_points = None
            print(f"🎯 Preparing manual point processing...")
            print(f"  - point_method == 'manual': {point_method == 'manual'}")
            print(f"  - manual_selected_points non-empty: {bool(manual_selected_points)}")
            print(f"  - manual_selected_points: {manual_selected_points}")
            
            if point_method == "manual" and manual_selected_points:
                # Following original logic: direct mapping to 256x256 coordinate system
                if current_image:
                    img_width, img_height = current_image.size
                    
                    # Direct mapping to 256x256 coordinate system (following original logic)
                    scale_x = 256.0 / img_width
                    scale_y = 256.0 / img_height
                    
                    manual_points_256 = []
                    for x, y in manual_selected_points:
                        mapped_x = x * scale_x
                        mapped_y = y * scale_y
                        manual_points_256.append((mapped_x, mapped_y))
                    
                    manual_points = manual_points_256
                    
                    print(f"✅ Manual points detected: {len(manual_selected_points)} points")
                    print(f"   UI Image size: {img_width}x{img_height}")
                    print(f"   Raw UI points: {manual_selected_points}")
                    print(f"   Mapped 256x256 points: {manual_points}")
                    
                    # Debug: Show detailed conversion for each point (following original)
                    print(f"   📐 Coordinate conversion details (original method):")
                    print(f"   Scale factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
                    for i, ((raw_x, raw_y), (mapped_x, mapped_y)) in enumerate(zip(manual_selected_points, manual_points)):
                        print(f"     Point{i}: UI({raw_x},{raw_y}) -> 256coords({mapped_x:.1f},{mapped_y:.1f})")
                else:
                    print("❌ No image available for manual point conversion")
            elif point_method == "manual":
                print(f"❌ Manual mode selected but no points available")
                print(f"   manual_selected_points is empty: {manual_selected_points}")
            else:
                print(f"📍 Using {point_method} method, not manual")
            
            try:
                # Generate trajectories (in manual mode, num_points determined by manual_points length)
                result_path, viz_path, message, _ = tracker.track_video(
                    video_path=video_path,
                    num_points=len(manual_points) if manual_points else 0,  # Determined by manual click count
                    point_method=point_method,
                    output_dir=output_dir,
                    enable_visualization=enable_viz,
                    manual_points=manual_points,
                    filename=filename
                )
                
                if result_path:
                    # Generation successful
                    return (
                        gr.update(value=message, elem_classes=["status-info", "success-box"]),
                        gr.update(value=result_path, visible=True),
                        gr.update(value=viz_path, visible=bool(viz_path)),
                        gr.update(value=viz_path, visible=bool(viz_path))  # Use same video file
                    )
                else:
                    # Generation failed
                    return (
                        gr.update(value=message, elem_classes=["status-info", "error-box"]),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                    
            except Exception as e:
                error_msg = f"❌ **Generation failed**: {str(e)}"
                return (
                    gr.update(value=error_msg, elem_classes=["status-info", "error-box"]),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        def output_dir_change_handler(new_output_dir):
            """Output directory change handler function"""
            try:
                os.makedirs(new_output_dir, exist_ok=True)
                from ..utils.file_utils import set_output_directory
                set_output_directory(new_output_dir)
                return gr.update()
            except Exception as e:
                print(f"Failed to set output directory: {e}")
                return gr.update()
        
        # ================== Edit Mode Event Handlers ==================
        
        def auto_fill_last_files():
            """Auto-fill last generated files"""
            video_path, tracks_path = tracker.get_last_generated_files()
            
            if video_path and tracks_path and os.path.exists(video_path) and os.path.exists(tracks_path):
                return (
                    gr.update(value=video_path),
                    gr.update(value=tracks_path),
                    gr.update(value=f"✅ **Auto-filled last generated files**\n\n📁 **Video file**: `{os.path.basename(video_path)}`\n📊 **Trajectory file**: `{os.path.basename(tracks_path)}`\n\n💡 **Tip**: These files are saved in the output directory and can be edited directly", elem_classes=["status-info", "success-box"])
                )
            else:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(value="❌ **No last generated files found**\n\n💡 **Tip**: Please generate trajectory files first in the \"Trajectory Generation\" tab", elem_classes=["status-info", "warning-box"])
                )
        
        def enter_edit_mode_handler(video_file, tracks_file):
            """Enter edit mode handler function"""
            if not video_file or not tracks_file:
                return (
                    gr.update(value="❌ **Please select video and trajectory files**", elem_classes=["status-info", "error-box"]),
                    gr.update(),
                    gr.update(),
                    gr.update(),  # semantic_text
                    gr.update(),  # points_list
                    gr.update(),  # points_info
                    gr.update(value="💡 **Please enter edit mode first**", elem_classes=["status-info"]),
                    gr.update(),  # prev_frame_btn
                    gr.update(),  # next_frame_btn
                    gr.update(),  # clear_selection_btn
                    gr.update(),  # toggle_visibility_btn
                    gr.update(),  # save_semantic_btn
                    gr.update(),  # save_tracks_btn
                    gr.update(),  # export_semantic_btn
                    gr.update()   # visualize_edited_btn
                )
            
            # Enter edit mode
            success, message, frame_data = edit_manager.initialize_edit_mode(video_file, tracks_file)
            
            if success and frame_data:
                frame_pil, _, _ = frame_data
                
                # 🔄 Try to load existing semantic information JSON file
                semantic_json_path = None
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                potential_json_path = os.path.join(
                    config.ensure_output_dir(), 
                    video_name, 
                    f"{video_name}_semantic_info.json"
                )
                
                if os.path.exists(potential_json_path):
                    semantic_json_path = potential_json_path
                    print(f"🔍 Found existing semantic information file: {semantic_json_path}")
                
                # Initialize semantic manager - using video-specific folder
                semantic_manager.initialize(
                    num_points=len(edit_manager.point_info),
                    video_path=video_file,  # Video file path
                    output_dir=config.ensure_output_dir(),  # semantic_info.py will automatically handle subdirectories
                    tracks_file_path=tracks_file   # .pth trajectory file path
                )
                
                # 🔄 If semantic information file is found, load it
                if semantic_json_path:
                    try:
                        import json
                        with open(semantic_json_path, 'r', encoding='utf-8') as f:
                            semantic_data = json.load(f)
                        
                        # Load semantic information
                        if 'semantic_info' in semantic_data:
                            for point_id_str, point_info in semantic_data['semantic_info'].items():
                                point_id = int(point_id_str)
                                if point_id < len(edit_manager.point_info):
                                    semantic_desc = point_info.get('semantic_description', '')
                                    semantic_manager.set_semantic_info(point_id, semantic_desc)
                        
                        print(f"✅ Loaded semantic information: {len(semantic_data.get('semantic_info', {}))} points")
                    except Exception as e:
                        print(f"⚠️ Failed to load semantic information: {e}")
                
                # Create trajectory point list options
                point_choices = [f"Point {i} - {'Visible' if point['visible'] else 'Hidden'}" for i, point in enumerate(edit_manager.point_info)]
                
                return (
                    gr.update(value=message, elem_classes=["status-info", "success-box"]),
                    gr.update(value=frame_pil, interactive=True),  # Enable image interaction
                    gr.update(maximum=edit_manager.frame_count-1, interactive=True),
                    gr.update(value="", interactive=True),  # semantic_text
                    gr.update(choices=point_choices, interactive=True),  # points_list - Fill trajectory point list
                    gr.update(value=f"📋 **Found {len(edit_manager.point_info)} trajectory points**"),  # points_info
                    gr.update(value="✅ **Edit mode activated**\n\n💡 Click trajectory points to select and edit", elem_classes=["status-info", "success-box"]),
                    gr.update(interactive=True),  # prev_frame_btn
                    gr.update(interactive=True),  # next_frame_btn
                    gr.update(interactive=True),  # clear_selection_btn
                    gr.update(interactive=True),  # toggle_visibility_btn
                    gr.update(interactive=True),  # save_semantic_btn
                    gr.update(interactive=True),  # save_tracks_btn
                    gr.update(interactive=True),  # export_semantic_btn
                    gr.update(interactive=True)   # visualize_edited_btn
                )
            else:
                return (
                    gr.update(value=message, elem_classes=["status-info", "error-box"]),
                    gr.update(),
                    gr.update(),
                    gr.update(),  # semantic_text
                    gr.update(),  # points_list
                    gr.update(),  # points_info
                    gr.update(value="❌ **Failed to start edit mode**", elem_classes=["status-info", "error-box"]),
                    gr.update(),  # prev_frame_btn
                    gr.update(),  # next_frame_btn
                    gr.update(),  # clear_selection_btn
                    gr.update(),  # toggle_visibility_btn
                    gr.update(),  # save_semantic_btn
                    gr.update(),  # save_tracks_btn
                    gr.update(),  # export_semantic_btn
                    gr.update()   # visualize_edited_btn
                )
        
        # ================== Bind Events ==================
        
        # Model loading
        load_model_btn.click(
            fn=load_model_handler,
            inputs=[checkpoint_path],
            outputs=[model_status]
        )
        
        # Folder scanning
        scan_folder_btn.click(
            fn=scan_folder_handler,
            inputs=[folder_path],
            outputs=[video_list_display, video_choice]
        )
        
        # Video selection
        video_choice.change(
            fn=video_selection_handler,
            inputs=[video_choice],
            outputs=[
                image_preview,
                manual_points_info,
                current_point_info,
                semantic_text_input,
                semantic_text_input,  # for interactive update
                save_semantic_point_btn,
                clear_semantic_point_btn,
                delete_current_point_btn,
                export_semantic_generation_btn,
                semantic_status,
                selected_points_list,
                point_selector
            ]
        )
        
        # Video upload
        video_upload.change(
            fn=video_upload_handler,
            inputs=[video_upload],
            outputs=[
                image_preview,
                manual_points_info,
                current_point_info,
                semantic_text_input,
                semantic_text_input,  # for interactive update
                save_semantic_point_btn,
                clear_semantic_point_btn,
                delete_current_point_btn,
                export_semantic_generation_btn,
                semantic_status,
                selected_points_list,
                point_selector
            ]
        )
        
        # Point generation method change
        point_method.change(
            fn=point_method_change_handler,
            inputs=[point_method],
            outputs=[manual_mode_info, clear_points_btn, manual_points_info]
        )
        
        # Manual point selection
        image_preview.select(
            fn=manual_image_click_handler,
            inputs=[point_method, image_preview],
            outputs=[
                image_preview, 
                manual_points_info,
                current_point_info,
                semantic_text_input,
                semantic_text_input,  # for interactive update
                save_semantic_point_btn,
                clear_semantic_point_btn,
                delete_current_point_btn,
                export_semantic_generation_btn,
                semantic_status,
                selected_points_list,
                point_selector
            ]
        )
        
        # Clear selected points
        clear_points_btn.click(
            fn=clear_points_handler,
            inputs=[image_preview],
            outputs=[
                image_preview, 
                manual_points_info,
                current_point_info,
                semantic_text_input,
                semantic_text_input,  # for interactive update
                save_semantic_point_btn,
                clear_semantic_point_btn,
                delete_current_point_btn,
                export_semantic_generation_btn,
                semantic_status,
                selected_points_list,
                point_selector
            ]
        )
        
        # Semantic information save events
        save_semantic_point_btn.click(
            fn=save_semantic_point_handler,
            inputs=[semantic_text_input],
            outputs=[semantic_status, selected_points_list, point_selector]
        )
        
        # Clear current point semantic information events
        clear_semantic_point_btn.click(
            fn=clear_semantic_point_handler,
            outputs=[semantic_status, semantic_text_input, selected_points_list, point_selector]
        )
        
        # Delete current point events
        delete_current_point_btn.click(
            fn=delete_current_point_handler,
            outputs=[
                semantic_status, 
                selected_points_list, 
                current_point_info,
                semantic_text_input,
                semantic_text_input,  # for interactive update
                save_semantic_point_btn,
                clear_semantic_point_btn,
                delete_current_point_btn,
                point_selector
            ]
        )
        
        # Point selector events
        point_selector.change(
            fn=point_selector_handler,
            inputs=[point_selector],
            outputs=[current_point_info, semantic_text_input, selected_points_list]
        )
        
        # Semantic information export events
        export_semantic_generation_btn.click(
            fn=export_semantic_generation_handler,
            outputs=[semantic_status]
        )
        
        # Generate trajectories
        generate_btn.click(
            fn=generate_tracks_handler,
            inputs=[video_upload, video_choice, point_method, enable_visualization, output_dir, image_preview],
            outputs=[generation_status, result_file, viz_file, preview_video]
        )
        
        # Output directory change
        output_dir.change(
            fn=output_dir_change_handler,
            inputs=[output_dir],
            outputs=[]
        )
        
        # Edit mode
        auto_fill_btn.click(
            fn=auto_fill_last_files,
            outputs=[edit_video_file, edit_tracks_file, edit_status]
        )
        
        enter_edit_btn.click(
            fn=enter_edit_mode_handler,
            inputs=[edit_video_file, edit_tracks_file],
            outputs=[
                edit_status, edit_image, frame_slider, semantic_text, points_list, points_info, edit_operation_status,
                prev_frame_btn, next_frame_btn, clear_selection_btn, toggle_visibility_btn, 
                save_semantic_btn, save_tracks_btn, export_semantic_btn, visualize_edited_btn
            ]
        )
        
        # Frame navigation events
        def frame_slider_change_handler(frame_idx):
            """Frame slider change handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), gr.update(value="❌ **Please enter edit mode first**"), gr.update()
            
            frame_pil, error, point_info = edit_manager.extract_frame_with_tracks(
                edit_manager.video_path, edit_manager.tracks_data, 
                frame_idx, edit_manager.estimated_original_size
            )
            
            if frame_pil is not None:
                edit_manager.current_frame_index = frame_idx
                edit_manager.point_info = point_info
                frame_info = f"**Frame info**: Frame {frame_idx+1} / {edit_manager.frame_count}"
                
                # Update trajectory point list (reflecting current frame visibility status)
                point_choices = [f"Point {i} - {'Visible' if point['visible'] else 'Hidden'}" for i, point in enumerate(point_info)]
                
                return (
                    gr.update(value=frame_pil), 
                    gr.update(value=frame_info),
                    gr.update(choices=point_choices)  # Update trajectory point list
                )
            else:
                return gr.update(), gr.update(value=f"❌ **Frame extraction failed**: {error}"), gr.update()
        
        def prev_frame_handler(current_frame):
            """Previous frame handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), gr.update(), gr.update()
            new_frame = max(0, current_frame - 1)
            frame_result = frame_slider_change_handler(new_frame)
            return frame_result[0], gr.update(value=new_frame), frame_result[2]
        
        def next_frame_handler(current_frame):
            """Next frame handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), gr.update(), gr.update()
            new_frame = min(edit_manager.frame_count - 1, current_frame + 1)
            frame_result = frame_slider_change_handler(new_frame)
            return frame_result[0], gr.update(value=new_frame), frame_result[2]
        
        # Image click events
        def edit_image_click_handler(evt: gr.SelectData):
            """Edit image click handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), gr.update(value="❌ **Please enter edit mode first**")
            
            click_x, click_y = evt.index[0], evt.index[1]
            
            # Find nearest point
            nearest_point = edit_manager.find_nearest_point(click_x, click_y)
            
            if nearest_point is not None:
                # Select point
                edit_manager.selected_point = nearest_point
                
                # Redraw current frame
                frame_pil, _, _ = edit_manager.extract_frame_with_tracks(
                    edit_manager.video_path, edit_manager.tracks_data,
                    edit_manager.current_frame_index, edit_manager.estimated_original_size,
                    selected_point=nearest_point
                )
                
                # Get semantic information
                semantic_info = semantic_manager.get_semantic_info(nearest_point)
                
                status_msg = f"✅ **Selected point {nearest_point}**\n\n💡 You can move the point or toggle visibility"
                return (
                    gr.update(value=frame_pil),
                    gr.update(value=semantic_info),
                    gr.update(value=status_msg, elem_classes=["status-info", "success-box"])
                )
            else:
                # If there's a selected point, move it
                if hasattr(edit_manager, 'selected_point') and edit_manager.selected_point is not None:
                    success, message = edit_manager.update_point_coordinates(
                        edit_manager.selected_point, click_x, click_y, edit_manager.current_frame_index
                    )
                    
                    if success:
                        # Redraw current frame
                        frame_pil, _, _ = edit_manager.extract_frame_with_tracks(
                            edit_manager.video_path, edit_manager.tracks_data,
                            edit_manager.current_frame_index, edit_manager.estimated_original_size,
                            selected_point=edit_manager.selected_point
                        )
                        
                        status_msg = f"✅ **{message}**"
                        return (
                            gr.update(value=frame_pil),
                            gr.update(),
                            gr.update(value=status_msg, elem_classes=["status-info", "success-box"])
                        )
                    else:
                        return (
                            gr.update(),
                            gr.update(),
                            gr.update(value=f"❌ **{message}**", elem_classes=["status-info", "error-box"])
                        )
                else:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(value="💡 **Please select a trajectory point first**", elem_classes=["status-info"])
                    )
        
        # Edit operation events
        def clear_selection_handler():
            """Clear selection handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), gr.update(value="❌ **Please enter edit mode first**")
            
            edit_manager.selected_point = None
            
            # Redraw current frame
            frame_pil, _, _ = edit_manager.extract_frame_with_tracks(
                edit_manager.video_path, edit_manager.tracks_data,
                edit_manager.current_frame_index, edit_manager.estimated_original_size
            )
            
            return (
                gr.update(value=frame_pil),
                gr.update(value="🔄 **Selection cleared**", elem_classes=["status-info"])
            )
        
        def toggle_visibility_handler():
            """Toggle visibility handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), gr.update(value="❌ **Please enter edit mode first**"), gr.update()
            
            if not hasattr(edit_manager, 'selected_point') or edit_manager.selected_point is None:
                return gr.update(), gr.update(value="❌ **Please select a trajectory point first**", elem_classes=["status-info", "error-box"]), gr.update()
            
            success, message = edit_manager.toggle_point_visibility(
                edit_manager.selected_point, edit_manager.current_frame_index
            )
            
            if success:
                # Redraw current frame
                frame_pil, _, point_info = edit_manager.extract_frame_with_tracks(
                    edit_manager.video_path, edit_manager.tracks_data,
                    edit_manager.current_frame_index, edit_manager.estimated_original_size,
                    selected_point=edit_manager.selected_point
                )
                
                # Update point information
                edit_manager.point_info = point_info
                
                # Update trajectory point list (reflecting new visibility status)
                point_choices = [f"Point {i} - {'Visible' if point['visible'] else 'Hidden'}" for i, point in enumerate(point_info)]
                
                return (
                    gr.update(value=frame_pil),
                    gr.update(value=f"✅ **{message}**", elem_classes=["status-info", "success-box"]),
                    gr.update(choices=point_choices)
                )
            else:
                return (
                    gr.update(),
                    gr.update(value=f"❌ **{message}**", elem_classes=["status-info", "error-box"]),
                    gr.update()
                )
        
        # Save operation events
        def save_semantic_handler(semantic_text):
            """Save semantic information handling"""
            if not hasattr(edit_manager, 'selected_point') or edit_manager.selected_point is None:
                return "❌ **Please select a trajectory point first**"
            
            success, message = semantic_manager.save_semantic_info(edit_manager.selected_point, semantic_text)
            if success:
                return f"✅ **{message}**"
            else:
                return f"❌ **{message}**"
        
        def save_tracks_handler():
            """Save tracks handling"""
            if edit_manager.tracks_data is None:
                return gr.update(), "❌ **Please enter edit mode first**"
            
            file_path, message = edit_manager.save_modified_tracks()
            if file_path:
                return gr.update(value=file_path, visible=True), message
            else:
                return gr.update(), message
        
        # Bind interactive editing events
        frame_slider.change(
            fn=frame_slider_change_handler,
            inputs=[frame_slider],
            outputs=[edit_image, frame_info, points_list]
        )
        
        prev_frame_btn.click(
            fn=prev_frame_handler,
            inputs=[frame_slider],
            outputs=[edit_image, frame_slider, points_list]
        )
        
        next_frame_btn.click(
            fn=next_frame_handler,
            inputs=[frame_slider],
            outputs=[edit_image, frame_slider, points_list]
        )
        
        edit_image.select(
            fn=edit_image_click_handler,
            outputs=[edit_image, semantic_text, edit_operation_status]
        )
        
        clear_selection_btn.click(
            fn=clear_selection_handler,
            outputs=[edit_image, edit_operation_status]
        )
        
        toggle_visibility_btn.click(
            fn=toggle_visibility_handler,
            outputs=[edit_image, edit_operation_status, points_list]
        )
        
        save_semantic_btn.click(
            fn=save_semantic_handler,
            inputs=[semantic_text],
            outputs=[points_info]
        )
        
        save_tracks_btn.click(
            fn=save_tracks_handler,
            outputs=[edited_tracks_file, edit_operation_status]
        )
        
        # Export semantic information events
        def export_semantic_handler():
            """Export semantic information handling function - export to JSON format"""
            try:
                # 📄 Use JSON format export, more convenient for program reading
                file_path, message = semantic_manager.export_semantic_info_to_json()
                if file_path:
                    return gr.update(value=message, elem_classes=["status-info", "success-box"])
                else:
                    return gr.update(value=f"❌ **Export failed**: {message}", elem_classes=["status-info", "error-box"])
            except Exception as e:
                return gr.update(value=f"❌ **Export failed**: {str(e)}", elem_classes=["status-info", "error-box"])
        
        export_semantic_btn.click(
            fn=export_semantic_handler,
            outputs=[edit_operation_status]
        )
        
        # Generate edited visualization events
        def visualize_edited_handler():
            """Generate edited visualization handling function"""
            try:
                if edit_manager.tracks_data is None or edit_manager.video_path is None:
                    return gr.update(value="❌ **Please enter edit mode first**", elem_classes=["status-info", "error-box"])
                
                # Import necessary functions
                from ..utils.visualization import visualize_tracks, save_visualization_video
                import cv2
                import torch
                import numpy as np
                import os
                from pathlib import Path
                
                print("🎨 Starting to generate edited visualization...")
                
                # Separate trajectory coordinates and visibility data from tracks_data
                tracks, visible = torch.split(edit_manager.tracks_data, [2, 1], dim=-1)
                print(f"📊 Trajectory data shape: tracks={tracks.shape}, visible={visible.shape}")
                print(f"📊 Trajectory coordinate range: X=[{tracks[:,:,0].min():.1f}, {tracks[:,:,0].max():.1f}], Y=[{tracks[:,:,1].min():.1f}, {tracks[:,:,1].max():.1f}]")
                
                # Get video size information
                cap = cv2.VideoCapture(edit_manager.video_path)
                original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 🔧 Key: Map 256x256 coordinates to original video size
                print(f"🔧 Coordinate mapping: 256x256 -> {original_width}x{original_height}")
                scale_x = original_width / 256.0
                scale_y = original_height / 256.0
                print(f"🔧 Scale factors: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
                
                # Map coordinates to original size
                tracks_scaled = tracks.clone()
                tracks_scaled[:, :, 0] = tracks[:, :, 0] * scale_x  # X coordinates
                tracks_scaled[:, :, 1] = tracks[:, :, 1] * scale_y  # Y coordinates
                
                print(f"🔧 Mapped coordinate range: X=[{tracks_scaled[:,:,0].min():.1f}, {tracks_scaled[:,:,0].max():.1f}], Y=[{tracks_scaled[:,:,1].min():.1f}, {tracks_scaled[:,:,1].max():.1f}]")
                
                # Load video frames
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                cap.release()
                
                frames = np.array(frames)
                print(f"📹 Loaded {len(frames)} video frames, size: {frames.shape}")
                
                # Adjust data format to match visualization function expectations
                # Current: tracks_scaled[frames, points, 2], visible[frames, points, 1]
                # Needed: tracks[points, frames, 1, 2], visible[points, frames, 1]
                
                # 1. Transpose dimensions: [frames, points, 2] -> [points, frames, 2]
                tracks_transposed = tracks_scaled.permute(1, 0, 2)  # [points, frames, 2]
                visible_transposed = visible.permute(1, 0, 2)  # [points, frames, 1]
                
                # 2. Add batch dimension: [points, frames, 2] -> [points, frames, 1, 2]
                tracks_for_viz = tracks_transposed.unsqueeze(2)  # [points, frames, 1, 2]
                visible_for_viz = visible_transposed  # [points, frames, 1] - keep this dimension
                
                print(f"🔄 Visualization data shape: tracks_for_viz={tracks_for_viz.shape}, visible_for_viz={visible_for_viz.shape}")
                
                # 🎨 Use fixed color mapping from edit manager to ensure color consistency
                fixed_colors = edit_manager.point_colors if edit_manager.point_colors else None
                print(f"🎨 Edited visualization using fixed colors: {'Yes' if fixed_colors else 'No'}")
                
                # Generate visualization frames
                viz_frames = visualize_tracks(
                    frames, tracks_for_viz.numpy(), visible_for_viz.numpy(),
                    tracks_leave_trace=config.TRACKS_LEAVE_TRACE,
                    fixed_colors=fixed_colors
                )
                
                # Generate output filename - save to video-specific folder
                video_name = Path(edit_manager.video_path).stem
                video_output_dir = os.path.join(config.ensure_output_dir(), video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                output_path = os.path.join(video_output_dir, f"{video_name}_edited_visualization.mp4")
                
                # Save visualization video
                save_visualization_video(viz_frames, output_path, fps=config.DEFAULT_FPS)
                
                return (
                    gr.update(value=f"✅ **Visualization generated successfully**\n\n📁 File saved to: {output_path}", elem_classes=["status-info", "success-box"]),
                    gr.update(value=output_path, visible=True)  # Display video preview
                )
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                return (
                    gr.update(value=f"❌ **Visualization generation failed**: {str(e)}", elem_classes=["status-info", "error-box"]),
                    gr.update()  # Don't update video component
                )
        
        visualize_edited_btn.click(
            fn=visualize_edited_handler,
            outputs=[edit_operation_status, edited_visualization_video]
        )
        
        # Trajectory point list selection events
        def points_list_change_handler(selected_point):
            """Trajectory point list selection handling"""
            if edit_manager.tracks_data is None or not selected_point:
                return gr.update(), gr.update(), gr.update(), gr.update()
            
            # Parse selected point ID (format: "Point 0 - Visible")
            try:
                point_id = int(selected_point.split()[1])
            except:
                return gr.update(), gr.update(), gr.update(), gr.update()
            
            # Set selected point
            edit_manager.selected_point = point_id
            
            # Redraw current frame, highlighting selected point
            frame_pil, _, _ = edit_manager.extract_frame_with_tracks(
                edit_manager.video_path, edit_manager.tracks_data,
                edit_manager.current_frame_index, edit_manager.estimated_original_size,
                selected_point=point_id
            )
            
            # Get semantic information for this point
            semantic_info = semantic_manager.get_semantic_info(point_id)
            
            # Display detailed information for the point
            if point_id < len(edit_manager.point_info):
                point = edit_manager.point_info[point_id]
                point_detail = f"""📍 **Trajectory Point {point_id} Details**

🔍 **Coordinates**: ({point['x']:.1f}, {point['y']:.1f})  
👁️ **Visibility**: {'Visible' if point['visible'] else 'Hidden'}  
📐 **Boundary Status**: {'Within bounds' if point['in_bounds'] else 'Out of bounds'}  
🏷️ **Semantic Annotation**: {semantic_info if semantic_info else 'None'}
"""
            else:
                point_detail = f"📍 **Trajectory Point {point_id}** - Loading information..."
            
            status_msg = f"✅ **Selected point {point_id} from list**\n\n💡 You can move the point in the image or toggle visibility"
            
            return (
                gr.update(value=frame_pil),
                gr.update(value=semantic_info),
                gr.update(value=point_detail),
                gr.update(value=status_msg, elem_classes=["status-info", "success-box"])
            )
        
        points_list.change(
            fn=points_list_change_handler,
            inputs=[points_list],
            outputs=[edit_image, semantic_text, points_info, edit_operation_status]
        )

    return app 