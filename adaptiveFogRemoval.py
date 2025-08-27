import cv2
import numpy as np
import os
import time

# --- Set OpenCV to use all available CPU cores ---
try:
    num_threads = os.cpu_count()
    cv2.setNumThreads(num_threads)
    print(f"✅ OpenCV is set to use {cv2.getNumThreads()} threads.")
except Exception as e:
    print(f"⚠️ Could not set OpenCV threads: {e}")


# --- Parameter Presets for Different Haze Levels ---
# The user can select one of these levels in the final application.
PRESETS = {
    'low': {
        'omega': 0.75,             # Less aggressive haze removal
        't0': 0.1,
        'clahe_clip_limit': 1.5,   # Less contrast enhancement
        'guided_filter_radius': 40 # Slightly smaller radius for more detail
    },
    'medium': {
        'omega': 0.9,              # Balanced haze removal
        't0': 0.1,
        'clahe_clip_limit': 2.0,   # Standard contrast enhancement
        'guided_filter_radius': 50
    },
    'high': {
        'omega': 0.95,             # Aggressive haze removal
        't0': 0.2,                 # Higher t0 for stability in dense fog
        'clahe_clip_limit': 3.0,   # More contrast to recover washed-out scenes
        'guided_filter_radius': 60 # Larger radius for smoother transmission
    }
}

# --- General Parameters (Usually don't need to change per level) ---
PROCESSING_WIDTH = 480
PATCH_SIZE = 9
ALPHA = 0.4
GUIDED_FILTER_EPS = 0.001
CLAHE_GRID_SIZE = (8, 8)


def get_dark_channel(frame, kernel):
    """Calculates the dark channel of an image using a pre-computed kernel."""
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_airlight(frame, dark_channel):
    """Estimates the atmospheric airlight using a more robust median."""
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = int(num_pixels * 0.001)
    flat_dark = dark_channel.flatten()
    flat_frame = frame.reshape(num_pixels, 3)
    indices = np.argsort(flat_dark)[-num_brightest:]
    brightest_pixels = flat_frame[indices]
    airlight = np.median(brightest_pixels, axis=0)
    return airlight

def get_transmission_map(frame, airlight, kernel, omega):
    """Estimates the transmission map."""
    airlight_safe = np.maximum(airlight.astype(np.float32), 1)
    normalized_frame = frame.astype(np.float32) / airlight_safe
    dark_channel_norm = get_dark_channel(normalized_frame, kernel)
    transmission = 1 - omega * dark_channel_norm
    return transmission

def refine_transmission_map(frame, transmission_map, radius, eps):
    """Refines the transmission map using the fast Guided Filter."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transmission_map_float = transmission_map.astype(np.float32)
    refined_t = cv2.ximgproc.guidedFilter(
        guide=gray_frame,
        src=transmission_map_float,
        radius=radius,
        eps=eps
    )
    return refined_t

def simple_white_balance(img):
    """Corrects color cast using the Gray World assumption."""
    img_float = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    if avg_b == 0 or avg_g == 0 or avg_r == 0: return img
    scale_b, scale_g, scale_r = avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r
    return np.clip(img_float, 0, 255).astype(np.uint8)

def restore_image(frame, airlight, transmission_map, t0):
    """Restores the haze-free image using the atmospheric scattering model."""
    t_clipped = np.maximum(transmission_map, t0)
    t_reshaped = cv2.cvtColor(t_clipped, cv2.COLOR_GRAY2BGR)
    frame_float = frame.astype(np.float32)
    airlight_float = airlight.astype(np.float32)
    restored = (frame_float - airlight_float) / t_reshaped + airlight_float
    return np.clip(restored, 0, 255).astype(np.uint8)

def apply_clahe(frame, clahe):
    """Applies CLAHE for final contrast enhancement using a pre-created object."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def temporal_smoothing(current_frame, previous_frame, alpha):
    """Applies an IIR filter for temporal smoothing between frames to reduce flicker."""
    if previous_frame is None:
        return current_frame
    return cv2.addWeighted(current_frame, alpha, previous_frame, 1 - alpha, 0)

def process_video(input_path, output_path, haze_level='medium'):
    """Main function to process the video with selectable haze level presets."""
    if haze_level not in PRESETS:
        print(f"❌ Error: Invalid haze level '{haze_level}'. Choose from {list(PRESETS.keys())}")
        return
    
    # Load parameters from the selected preset
    params = PRESETS[haze_level]
    print(f"✅ Using '{haze_level}' preset with parameters: {params}")

    if not os.path.exists(input_path):
        print(f"❌ Error: Input video not found at '{input_path}'")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {input_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    processing_height = int(PROCESSING_WIDTH * (original_height / original_width))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (PATCH_SIZE, PATCH_SIZE))
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'], tileGridSize=CLAHE_GRID_SIZE)

    prev_frame_smoothed = None
    frame_count = 0
    start_time = time.time()

    print(f"🚀 Starting video processing for '{input_path}'...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        small_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height), interpolation=cv2.INTER_AREA)
        dark_channel = get_dark_channel(small_frame, kernel)
        airlight = get_airlight(small_frame, dark_channel)
        
        # Use parameters from the 'params' dictionary
        transmission_map = get_transmission_map(small_frame, airlight, kernel, params['omega'])
        refined_t_small = refine_transmission_map(small_frame, transmission_map, params['guided_filter_radius'], GUIDED_FILTER_EPS)
        
        full_size_t = cv2.resize(refined_t_small, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        restored_frame = restore_image(frame, airlight, full_size_t, params['t0'])
        balanced_frame = simple_white_balance(restored_frame)
        enhanced_frame = apply_clahe(balanced_frame, clahe)
        smoothed_frame = temporal_smoothing(enhanced_frame, prev_frame_smoothed, ALPHA)

        prev_frame_smoothed = smoothed_frame
        out.write(smoothed_frame)

        elapsed_time = time.time() - start_time
        fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        print(f"\r⚙️  Processing frame {frame_count}/{total_frames} | FPS: {fps_current:.2f} | ETA: {eta_str}", end="")

    print(f"\n\n🎉 Finished processing! Video saved to '{output_path}'")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video_name = 'hazy_input.mp4'
    output_video_name = 'dehazed_output_high_fog.mp4'
    
    # --- CHOOSE YOUR FOG LEVEL HERE ---
    # Options: 'low', 'medium', 'high'
    selected_level = 'high'
    
    process_video(input_video_name, output_video_name, haze_level=selected_level)