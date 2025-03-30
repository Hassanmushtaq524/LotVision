import cv2
import numpy as np
import time
import os
import platform
import argparse
import glob
from paddleocr import PaddleOCR
from tqdm import tqdm  # Import tqdm for progress bars
import logging
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests  # Add requests library for API calls

# Configure logging to suppress debug messages
logging.basicConfig(level=logging.ERROR)  # Only show ERROR level messages or higher

# Redirect paddle debug output
os.environ["GLOG_v"] = "0"  # Suppress Paddle verbose logs
os.environ["FLAGS_logtostderr"] = "0"  # Disable logging to stderr

# Set default folder path for videos - remove the extra single quotes
DEFAULT_FOLDER_PATH = "/Users/wasifkarim/Documents/Working Project/Final/test"

# Set default video paths as fallback
DEFAULT_VIDEO_PATHS = [
    "videos/traffic1.mp4",
    "videos/traffic2.mp4"
    # Add more default videos as needed
]

# Define valid video extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# Enhanced parameters for license plate detection
MIN_PLATE_RATIO = 2.0  # Minimum w/h ratio of a license plate
MAX_PLATE_RATIO = 5.5  # Maximum w/h ratio of a license plate
MIN_PLATE_AREA = 500   # Minimum area of a license plate
MAX_PLATE_AREA = 15000 # Maximum area of a license plate
PLATE_PADDING = 10     # Pixels to add around detected plate for OCR

# Optimization parameters
DOWNSAMPLE_FACTOR = 0.5  # Downsample large frames to speed up processing
MAX_PLATE_CANDIDATES = 30  # Max number of contours to consider as plate candidates
MAX_PENDING_FUTURES = 50   # Maximum number of pending OCR tasks
CONTOUR_APPROXIMATION_FACTOR = 0.018  # Control how much we simplify contours

# Validation patterns for license plates (add specific patterns for your region if needed)
# This is a generic alphanumeric pattern - customize for your specific needs
LICENSE_PLATE_PATTERNS = [
    r'^[A-Z0-9]{5,8}$',         # Generic 5-8 character alphanumeric pattern
    r'^[A-Z]{1,3}[0-9]{1,5}$',  # Letters followed by numbers
    r'^[0-9]{1,5}[A-Z]{1,3}$'   # Numbers followed by letters
]

def zoom_image(image, zoom_factor=0):
    """
    Zoom into the center of an image by the specified factor.
    zoom_factor: 0.4 means zoom in by 40% (i.e., make center objects 40% larger)
    """
    # Calculate dimensions
    h, w = image.shape[:2]
    
    # Calculate the crop size (smaller values = more zoom)
    crop_factor = 1.0 / (1.0 + zoom_factor)
    
    # Calculate new dimensions
    new_h = int(h * crop_factor)
    new_w = int(w * crop_factor)
    
    # Calculate crop position (center crop)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    
    # Crop the center of the image
    cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
    
    # Resize back to the original dimensions
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed

def clean_plate_text(text):
    """
    Enhanced cleaning of license plate text with validation
    """
    if not text:
        return ""
    
    # Remove common OCR errors and standardize formatting
    cleaned_text = text.replace('-', '').replace(' ', '').replace('.', '').replace('_', '')
    cleaned_text = cleaned_text.upper()
    
    # Remove any non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', cleaned_text)
    
    # Ensure minimum length (most plates have at least 5-6 characters)
    if len(cleaned_text) < 4:
        return ""
    
    return cleaned_text

def validate_plate_text(text):
    """
    Validate if the cleaned text matches expected license plate patterns
    Returns a score indicating how likely this is to be a valid plate (0-1)
    """
    if not text or len(text) < 4:
        return 0.0
    
    # Check against known patterns
    for pattern in LICENSE_PLATE_PATTERNS:
        if re.match(pattern, text):
            return 1.0  # Perfect match to a known pattern
    
    # Partial validation based on string composition
    alphanumeric_ratio = sum(c.isalnum() for c in text) / max(1, len(text))
    length_score = min(1.0, max(0, len(text) - 3) / 5)  # Length between 4-8 chars scores 0.2-1.0
    
    # Combined score
    return 0.5 * alphanumeric_ratio + 0.5 * length_score

def preprocess_plate_roi(roi):
    """
    Apply preprocessing to enhance the license plate region for better OCR
    """
    if roi is None or roi.size == 0:
        return None
    
    # Keep a copy of the original for cases where processing makes things worse
    original_roi = roi.copy()
    
    # Resize to a standard height for consistency while maintaining aspect ratio
    target_height = 60
    aspect_ratio = roi.shape[1] / roi.shape[0]
    target_width = int(target_height * aspect_ratio)
    roi = cv2.resize(roi, (target_width, target_height))
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Try to invert the result if it's a dark background with light text
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    
    # Create a color image from the threshold result for OCR
    processed_roi = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Return both original and processed ROIs for OCR to try both
    return (roi, processed_roi)

def get_video_files_from_folder(folder_path):
    """
    Get a list of all video files in the specified folder
    """
    # Check if path contains unnecessary quotes and remove them
    if folder_path.startswith("'") and folder_path.endswith("'"):
        folder_path = folder_path[1:-1]
        print(f"Notice: Removing quotes from path. Using: {folder_path}")
        
    if not os.path.isdir(folder_path):
        print(f"Warning: '{folder_path}' is not a valid directory or cannot be accessed")
        print(f"Current working directory is: {os.getcwd()}")
        return []
    
    video_files = []
    
    # Walk through all files in the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a video extension
            _, ext = os.path.splitext(file)
            if ext.lower() in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video files found in {folder_path}")
        # List files to help troubleshoot
        print("Files in directory:")
        try:
            for f in os.listdir(folder_path):
                print(f"  - {f}")
        except Exception as e:
            print(f"Error listing directory contents: {e}")
    else:
        print(f"Found {len(video_files)} video files in {folder_path}")
        for video in video_files:
            print(f"  - {os.path.basename(video)}")
    
    return video_files

# Function to process a batch of potential license plates
def process_plates_batch(rois, ocr, frame_count, source, timestamp):
    """
    Process a batch of ROIs with OCR to improve throughput
    """
    results = []
    
    for roi_data in rois:
        x1, y1, x2, y2, roi, processed_roi = roi_data
        
        # Try both the original and processed ROI with OCR
        best_text = ""
        best_confidence = 0
        
        for roi_img in [roi, processed_roi]:
            ocr_results = ocr.ocr(roi_img, cls=False)
            
            # Check for valid results
            has_valid_results = False
            if ocr_results is not None and len(ocr_results) > 0:
                if ocr_results[0] is not None and len(ocr_results[0]) > 0:
                    has_valid_results = True
            
            if has_valid_results:
                # Find best text by confidence
                for line in ocr_results[0]:
                    if line[1][1] > best_confidence:
                        best_text = line[1][0]
                        best_confidence = line[1][1]
        
        # Clean and validate the text
        if best_text:
            cleaned_text = clean_plate_text(best_text)
            validation_score = validate_plate_text(cleaned_text)
            
            # Only consider results with good validation and confidence
            combined_score = best_confidence * validation_score
            if combined_score > 0.5 and best_confidence > 0.7:
                plate_info = {
                    "raw_text": best_text,
                    "text": cleaned_text,
                    "confidence": best_confidence,
                    "validation": validation_score,
                    "combined_score": combined_score,
                    "frame": frame_count,
                    "timestamp": timestamp,
                    "source": source,
                    "roi": None,  # Don't store the image
                    "coordinates": (x1, y1, x2, y2)  # Store coordinates for tracking
                }
                results.append(plate_info)
    
    return results

def resize_if_needed(frame, max_width=1280):
    """
    Resizes frame if it's too large for faster processing
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def process_video(source, ocr=None, frame_sample_rate=15, headless=False, camera_delay=0.1, zoom_camera=0.4, use_cuda=False, use_opencl=False, batch_size=12, threads=None, smart_skip=True):
    """
    Process a single video source and return detected license plates
    """
    # Create OCR instance if not provided (for parallel processing)
    if ocr is None:
        use_gpu = False
        try:
            import paddle
            use_gpu = paddle.device.is_compiled_with_cuda()
        except Exception:
            pass
            
        # Initialize PaddleOCR with fallback mechanisms
        ocr = None
        
        # Try with optimized settings first
        try:
            ocr_kwargs = {
                'lang': 'en',             
                'use_gpu': use_gpu,       
                'show_log': False,        
                'use_angle_cls': False,   
                'det_limit_side_len': 960,
                'det_db_thresh': 0.3,     
                'det_db_box_thresh': 0.5, 
                'rec_batch_num': batch_size,
                # Simplified settings for compatibility
                'use_mp': True,           
                'total_process_num': threads if threads else min(os.cpu_count(), 4),
            }
            ocr = PaddleOCR(**ocr_kwargs)
            print("Subprocess: PaddleOCR initialized with optimized settings")
        except Exception as e:
            print(f"Subprocess: Error initializing PaddleOCR with optimized settings: {e}")
            print("Subprocess: Trying with simplified settings...")
            
            try:
                # Try with simpler settings
                ocr_kwargs = {
                    'lang': 'en',
                    'use_gpu': use_gpu,
                    'show_log': False,
                    'use_angle_cls': False,
                    'det_limit_side_len': 960,
                    'det_db_thresh': 0.3,
                    'det_db_box_thresh': 0.5,
                    'rec_batch_num': batch_size
                }
                ocr = PaddleOCR(**ocr_kwargs)
                print("Subprocess: PaddleOCR initialized with simplified settings")
            except Exception as e:
                print(f"Subprocess: Error initializing PaddleOCR with simplified settings: {e}")
                print("Subprocess: Trying with basic settings...")
                
                try:
                    # Try with minimal settings
                    ocr_kwargs = {
                        'lang': 'en',
                        'use_gpu': False,  # Force CPU mode
                        'show_log': False
                    }
                    ocr = PaddleOCR(**ocr_kwargs)
                    print("Subprocess: PaddleOCR initialized with basic settings")
                except Exception as e:
                    print(f"Subprocess: Failed to initialize PaddleOCR: {e}")
                    print("Subprocess: Cannot continue without OCR engine")
                    return []
                    
        if ocr is None:
            print("Subprocess: Failed to initialize PaddleOCR")
            return []
    
    # Check if source is a video file
    is_video_file = isinstance(source, str) and os.path.isfile(source)
    
    # Set up the GPU pipeline if available
    if use_cuda:
        # Create CUDA Stream for faster processing
        stream = cv2.cuda_Stream()
    elif use_opencl:
        cv2.ocl.setUseOpenCL(True)
        
    if is_video_file:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file {source}")
            return []
    else:
        # Camera initialization code (simplified to reduce output)
        is_macos = platform.system() == 'Darwin'
        camera_index = source if isinstance(source, int) else 0
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if is_macos:
                cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                test_success, _ = cap.read()
                if test_success:
                    cap.set(3, 640)  # width
                    cap.set(4, 480)  # height
                    break
                else:
                    cap.release()
                    time.sleep(2)
            else:
                camera_index = (camera_index + 1) % 3
                time.sleep(2)
        
        if not cap or not cap.isOpened():
            print("ERROR: Could not access any camera.")
            return []
        
        # Warmup the camera - some cameras need a few frames to stabilize
        for _ in range(10):
            cap.read()
    
    count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    frame_count = 0
    
    # Create a list to store all detected license plates
    detected_plates = []
    
    # Initialize detection counter at the start of the function
    detection_count = 0
    
    # Get video details if processing a file
    if is_video_file:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frames to process based on sample rate
        process_every_n_frames = max(1, int(fps / frame_sample_rate))
        
        # Create progress bar for video processing - the only output we want to see
        if headless:
            pbar = tqdm(total=total_frames, desc="Processing video", unit="frames", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    # For camera mode - track current zoom level
    current_zoom = zoom_camera if not is_video_file else 0
    
    # Prepare for batch processing
    roi_batch = []
    
    # Set up multi-threading - create a thread pool for parallel OCR
    max_workers = threads if threads is not None else min(os.cpu_count() + 4, 16)  # More workers for I/O bound tasks
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    # For plate tracking (to reduce duplicate detections)
    previously_detected_regions = {}
    skip_frames_after_detection = 10  # Skip processing this region for N frames after a detection
    
    # Keep track of pending futures
    pending_futures = []
    
    # Create histograms for faster contour filtering
    hist_w = np.zeros(100)
    hist_h = np.zeros(100)
    hist_area = np.zeros(100)
    hist_ratio = np.zeros(100)
    
    # Dictionary to track processing times for optimization
    timings = {
        'read': 0,
        'preprocess': 0,
        'contour': 0,
        'roi_extract': 0,
        'total': 0
    }
    last_timing_report = time.time()
    
    # For smart frame skipping
    if smart_skip and is_video_file:
        # Variables for smart frame skipping
        prev_frame = None
        skip_counter = 0
        motion_threshold = 2000  # Threshold for motion detection
        base_skip = process_every_n_frames
        adaptive_skip = base_skip
        
        print("Smart frame skipping enabled")
    
    while True:
        start_time = time.time()
        
        success, frame = cap.read()
        timings['read'] += time.time() - start_time
        
        if not success:
            if is_video_file:
                break
                
            consecutive_failures += 1
            
            if consecutive_failures >= max_consecutive_failures:
                cap.release()
                time.sleep(2)
                
                if is_macos:
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
                else:
                    cap = cv2.VideoCapture(camera_index)
                
                if not cap.isOpened():
                    break
                consecutive_failures = 0
            
            time.sleep(0.5)
            continue
        
        frame_count += 1
        consecutive_failures = 0  # Reset the counter when successful
        
        # Update progress bar for video files in headless mode
        if is_video_file and headless:
            pbar.update(1)
        
        # Smart frame skipping for video files
        if smart_skip and is_video_file:
            # Determine whether to skip this frame based on motion
            process_this_frame = False
            
            if frame_count % max(1, adaptive_skip) == 0:
                process_this_frame = True
                
                # If we have a previous frame, check for motion
                if prev_frame is not None:
                    # Resize for faster processing during motion detection
                    small_frame = cv2.resize(frame, (320, 240))
                    small_prev = cv2.resize(prev_frame, (320, 240))
                    
                    # Calculate frame difference
                    frame_diff = cv2.absdiff(small_frame, small_prev)
                    motion_score = np.sum(frame_diff) / (small_frame.size * 255)
                    
                    # Adjust skip rate based on motion
                    if motion_score > 0.05:  # Significant motion
                        adaptive_skip = max(1, base_skip // 2)  # Process more frames
                    elif motion_score > 0.02:  # Moderate motion
                        adaptive_skip = base_skip
                    else:  # Little to no motion
                        adaptive_skip = base_skip * 2  # Skip more frames
            else:
                process_this_frame = False
                
            # Store current frame for next comparison
            prev_frame = frame.copy()
            
            # Skip this frame if not processing
            if not process_this_frame:
                continue
        # Standard frame skipping for video files
        elif is_video_file and (frame_count % process_every_n_frames != 0):
            continue
        
        # Apply zoom for camera input (not video files)
        if not is_video_file and current_zoom > 0:
            frame = zoom_image(frame, current_zoom)
            
        # Add intentional delay for camera mode to slow down processing
        if not is_video_file:
            time.sleep(camera_delay)
            
        # Resize large frames for faster processing
        t0 = time.time()
        frame = resize_if_needed(frame)  # Use default max_width from wrapper
        
        # Process frame with GPU if available
        if use_cuda:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur instead of bilateral filter (much faster)
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
            
            # Edge detection on GPU
            gpu_edges = cv2.cuda.createCannyEdgeDetector(30, 200).detect(gpu_blurred)
            
            # Download results for contour detection (contour detection not available on CUDA)
            edged = gpu_edges.download()
        elif use_opencl:
            # OpenCL path - using UMat for GPU acceleration
            gpu_frame = cv2.UMat(frame)
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur instead of bilateral filter (much faster)
            gpu_blurred = cv2.GaussianBlur(gpu_gray, (5, 5), 0)
            
            # Edge detection
            edged = cv2.Canny(gpu_blurred, 30, 200)
            
            # Convert back to CPU for contour detection
            edged = edged.get()
        else:
            # Original CPU processing path
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur instead of bilateral filter (much faster)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edged = cv2.Canny(gray, 30, 200)
            
        timings['preprocess'] += time.time() - t0
        
        t0 = time.time()
        # Find contours (must be done on CPU)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first) and take only the top candidates
        # This is faster than processing all contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_PLATE_CANDIDATES]
        
        timings['contour'] += time.time() - t0
        
        # Clear the batch for this frame
        roi_batch = []
        
        # Process each contour
        t0 = time.time()
        for contour in contours:
            # Skip very small contours early
            area = cv2.contourArea(contour)
            if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
                continue
                
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, CONTOUR_APPROXIMATION_FACTOR * peri, True)
            
            # Look for contours with 4 corners (rectangular shape)
            if len(approx) >= 4 and len(approx) <= 6:  # More lenient: 4-6 corners
                x, y, w, h = cv2.boundingRect(approx)
                
                # Skip small or large regions early
                if w * h < MIN_PLATE_AREA or w * h > MAX_PLATE_AREA:
                    continue
                
                aspect_ratio = float(w)/h
                
                # Filter based on aspect ratio (typical license plate aspect ratio)
                if MIN_PLATE_RATIO <= aspect_ratio <= MAX_PLATE_RATIO:
                    # Check if this region overlaps with a recently detected plate
                    # This prevents processing the same plate multiple times
                    region_key = f"{x//10}-{y//10}-{w//10}-{h//10}"  # Quantize to reduce noise
                    
                    # Skip if we've seen this region recently
                    if region_key in previously_detected_regions:
                        last_frame, _ = previously_detected_regions[region_key]
                        if frame_count - last_frame < skip_frames_after_detection:
                            continue
                    
                    # Add padding
                    padding = PLATE_PADDING
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    # Extract ROI
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Pre-process the ROI to improve OCR
                        processed_rois = preprocess_plate_roi(roi)
                        
                        if processed_rois:
                            original_roi, processed_roi = processed_rois
                            roi_batch.append((x1, y1, x2, y2, original_roi, processed_roi))
                            
                            # Draw the contour in real-time visualization
                            if not headless:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        timings['roi_extract'] += time.time() - t0
        
        # Process the batch of ROIs if we have enough
        current_timestamp = time.time()
        
        if roi_batch:
            # Process in batches using the thread pool for better performance
            future = executor.submit(
                process_plates_batch, roi_batch, ocr, frame_count, source, current_timestamp
            )
            # Add to pending futures
            pending_futures.append(future)
        
        # Check for completed futures without blocking
        # Get a list of futures that are done
        done_futures = []
        for future in pending_futures:
            if future.done():
                done_futures.append(future)
        
        # Process results from completed futures
        for future in done_futures:
            try:
                batch_results = future.result()  # This won't block since we know it's done
                if batch_results:
                    # Update detection count
                    detection_count += len(batch_results)
                    detected_plates.extend(batch_results)
                    
                    # Log detection (but not too frequently)
                    if detection_count % 10 == 0:
                        print(f"Detected {detection_count} license plates so far...")
                    
                    # Mark detected regions to avoid reprocessing
                    for plate in batch_results:
                        x1, y1, x2, y2 = plate["coordinates"]
                        region_key = f"{x1//10}-{y1//10}-{(x2-x1)//10}-{(y2-y1)//10}"
                        previously_detected_regions[region_key] = (frame_count, plate["text"])
                        
                        # Draw on the frame for visualization
                        if not headless:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, plate["text"], (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error processing batch: {e}")
        
        # Remove processed futures from the pending list
        for future in done_futures:
            pending_futures.remove(future)
            
        # Limit the number of pending futures to prevent memory issues
        if len(pending_futures) > MAX_PENDING_FUTURES:
            # Wait for some futures to complete if we have too many pending
            # This ensures we don't accumulate an unbounded number of futures
            try:
                for done_future in as_completed(pending_futures[:max_workers], timeout=0.1):
                    try:
                        batch_results = done_future.result()
                        if batch_results:
                            detection_count += len(batch_results)
                            detected_plates.extend(batch_results)
                    except Exception as e:
                        print(f"Error processing batch during cleanup: {e}")
                    pending_futures.remove(done_future)
            except TimeoutError:
                # It's ok if we timeout here, we just continue processing
                pass
        
        # Display frames in interactive mode (now default)
        if not headless:
            # Add zoom level indicator on camera mode
            if not is_video_file:
                zoom_text = f"Zoom: {current_zoom*100:.0f}%"
                cv2.putText(frame, zoom_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection info to the frame
            if detected_plates:
                info_text = f"Detected: {detection_count} plates"
                cv2.putText(frame, info_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame info
            source_name = os.path.basename(source) if isinstance(source, str) else f"Camera {source}"
            frame_info = f"Source: {source_name} | Frame: {frame_count}"
            cv2.putText(frame, frame_info, (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow("License Plate Detection", frame)
            
            if not is_video_file or frame_count % 30 == 0:  # Don't show edges for every frame in video mode
                cv2.imshow("Edges", edged)  # Show edge detection for debugging
            
            # Handle key presses
            waitkey_time = 1 if not is_video_file else max(1, int(1000/frame_sample_rate))  # Control playback speed
            key = cv2.waitKey(waitkey_time) & 0xFF
            
            if key == ord('s') and 'roi' in locals() and roi.size > 0:
                # Make sure the directory exists
                save_dir = "Model01/plates"
                if not is_video_file:
                    save_dir = os.path.join(save_dir, "camera")
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"scaned_img_{count}.jpg")
                cv2.imwrite(save_path, roi)
                
                # Show save confirmation
                cv2.rectangle(frame, (0,200), (640,300), (0,255,0), cv2.FILLED)
                cv2.putText(frame, "Plate Saved", (150, 265), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("License Plate Detection", frame)
                cv2.waitKey(500)
                count += 1
            elif key == ord('+') or key == ord('='):  # Increase zoom for camera
                if not is_video_file:
                    current_zoom = min(1.0, current_zoom + 0.1)  # Max zoom 100%
                    print(f"Zoom level increased to {current_zoom*100:.0f}%")
            elif key == ord('-') or key == ord('_'):  # Decrease zoom for camera
                if not is_video_file:
                    current_zoom = max(0, current_zoom - 0.1)
                    print(f"Zoom level decreased to {current_zoom*100:.0f}%")
            elif key == ord(' '):  # Space to pause/resume video
                if is_video_file:
                    print("Video paused - press space to continue")
                    while True:
                        if cv2.waitKey(100) & 0xFF == ord(' '):
                            print("Resuming video")
                            break
            elif key == ord('q'):
                break
        else:
            # For headless mode, we'll remove the sleep to increase throughput
            # time.sleep(0.001)
            
            # For live camera in headless mode, add a way to break the loop (e.g., with Ctrl+C)
            if not is_video_file and frame_count % 100 == 0:
                print(f"Processing ongoing... (frame {frame_count}) - Press Ctrl+C to stop")
                
        # Log processing performance metrics periodically
        timings['total'] += time.time() - start_time
        current_time = time.time()
        if current_time - last_timing_report > 10.0:  # Every 10 seconds
            frames_since_last = max(1, frame_count - (last_timing_report_frame if 'last_timing_report_frame' in locals() else 0))
            fps = frames_since_last / (current_time - last_timing_report)
            if not headless:
                print(f"\nPerformance: {fps:.1f} FPS, Read: {timings['read']/frames_since_last*1000:.1f}ms, "
                      f"Preprocess: {timings['preprocess']/frames_since_last*1000:.1f}ms, "
                      f"Contour: {timings['contour']/frames_since_last*1000:.1f}ms, "
                      f"ROI: {timings['roi_extract']/frames_since_last*1000:.1f}ms")
            
            # Reset timings
            for k in timings:
                timings[k] = 0
            last_timing_report = current_time
            last_timing_report_frame = frame_count
    
    # Close the progress bar if it exists
    if is_video_file and headless and 'pbar' in locals():
        pbar.close()
    
    # Wait for any remaining futures to complete before shutting down
    if pending_futures:
        print(f"Waiting for {len(pending_futures)} pending OCR tasks to complete...")
        for future in as_completed(pending_futures):
            try:
                batch_results = future.result()
                if batch_results:
                    detected_plates.extend(batch_results)
            except Exception as e:
                print(f"Error in final batch processing: {e}")
    
    # Cleanup
    if use_cuda and 'stream' in locals():
        stream.waitForCompletion()
    
    # Shutdown the thread pool
    executor.shutdown(wait=True)
    
    if cap:
        cap.release()
    
    if not headless:
        cv2.destroyAllWindows()
    
    # Return the detected plates
    return detected_plates

def generate_summary(detected_plates, source=None):
    """
    Generate and display a summary of the detected license plates using a set
    Only includes detections with high confidence level
    Also sends each unique plate to the API endpoint
    """
    if not detected_plates:
        return

    # Filter plates if a specific source is provided
    if source:
        plates_to_summarize = [p for p in detected_plates if p["source"] == source]
    else:
        plates_to_summarize = detected_plates

    if not plates_to_summarize:
        return

    # Use a set to store unique license plates
    unique_plates = set()
    
    # Process each plate detection
    for plate in plates_to_summarize:
        text = plate["text"]
        confidence = plate["confidence"]
        validation_score = plate.get("validation", 0.75)
        combined_score = plate.get("combined_score", confidence * validation_score)
        
        # Only include plates with good combined score
        if combined_score > 0.93:  # High confidence threshold
            unique_plates.add(text)

    # Display results
    if unique_plates:
        print("\nUnique License Plates Detected:")
        print("------------------------------")
        for plate in sorted(unique_plates):  # Sort for consistent display
            print(f"- {plate}")
            
            # Send each plate to the API endpoint
            try:
                api_url = "https://lotvision.onrender.com/current_cars/"
                payload = {
                    "car_plate_num": plate,
                    "lot_id": "B"
                }
                response = requests.post(api_url, json=payload)
                
                # Print API response information
                if response.status_code == 200 or response.status_code == 201:
                    print(f"  ✓ Successfully sent to API (Status: {response.status_code})")
                else:
                    print(f"  ✗ Failed to send to API (Status: {response.status_code}): {response.text}")
            except Exception as e:
                print(f"  ✗ Error sending to API: {str(e)}")
                
        print(f"\nTotal unique plates detected: {len(unique_plates)}")
    else:
        print("\nNo high-confidence license plate detections")

def main(source=None, frame_sample_rate=10, headless=False, camera_delay=0.1, zoom_camera=0.4, batch_size=12, threads=None, smart_skip=True):
    """
    Main function to process a single video source
    """
    # Default to camera if no source is provided
    if source is None:
        source = 0  # Default to camera 0
    
    # Initialize PaddleOCR silently with optimized settings
    use_gpu = False
    use_opencl = False
    opencv_cuda = False
    
    try:
        # Check OpenCV OpenCL availability for AMD GPUs
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            use_opencl = cv2.ocl.useOpenCL()
            
        # Check for CUDA in case of NVIDIA GPU
        import paddle
        use_gpu = paddle.device.is_compiled_with_cuda()
    except Exception:
        pass
    
    # Check OpenCV CUDA availability
    try:
        cv_build_info = cv2.getBuildInformation()
        if "NVIDIA CUDA" in cv_build_info and "Yes" in cv_build_info[cv_build_info.find("NVIDIA CUDA"):cv_build_info.find("\n", cv_build_info.find("NVIDIA CUDA"))]:
            opencv_cuda = True
            # Create a CUDA-enabled OpenCV device
            cuda_test = cv2.cuda_GpuMat()
    except Exception:
        opencv_cuda = False
    
    # Print GPU acceleration status
    print(f"GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"OpenCL Acceleration: {'Enabled' if use_opencl else 'Disabled'}")
    print(f"OpenCV CUDA: {'Enabled' if opencv_cuda else 'Disabled'}")
    
    # Initialize PaddleOCR with progressively simpler settings in case of failures
    ocr = None
    
    # Try with optimized settings first
    try:
        ocr_kwargs = {
            'lang': 'en',
            'use_gpu': use_gpu,
            'show_log': False,
            'use_angle_cls': False,
            'det_limit_side_len': 960,
            'det_db_thresh': 0.3,
            'det_db_box_thresh': 0.5,
            'rec_batch_num': batch_size,
            'use_mp': True,
            'total_process_num': threads if threads else min(os.cpu_count(), 4),
        }
        ocr = PaddleOCR(**ocr_kwargs)
        print("PaddleOCR initialized with optimized settings")
    except Exception as e:
        print(f"Error initializing PaddleOCR with optimized settings: {e}")
        # Simplified fallback logic for clarity
        try:
            ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)
            print("PaddleOCR initialized with basic settings")
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            return []
    
    # Process video and get detected plates
    detected_plates = process_video(
        source=source,
        ocr=ocr,
        frame_sample_rate=frame_sample_rate,
        headless=headless,
        camera_delay=camera_delay,
        zoom_camera=zoom_camera,
        use_cuda=opencv_cuda,
        use_opencl=use_opencl,
        batch_size=batch_size,
        threads=threads,
        smart_skip=smart_skip
    )
    
    # Generate summary for this source
    generate_summary(detected_plates, source=source)
    
    return detected_plates

if __name__ == "__main__":
    # Simplified command line argument parsing
    parser = argparse.ArgumentParser(description='License Plate Detection from Camera or Video')
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to video file (default: use default video path)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index to use if no video file is specified (default: 0)')
    parser.add_argument('--fps', type=int, default=10, 
                        help='Frames per second to process when using video file (default: 10)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode without displaying video (default: show video)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay in seconds between camera frames (default: 0.1)')
    parser.add_argument('--zoom', type=float, default=0.4,
                        help='Camera zoom level (0.0-1.0, default: 0.4 = 40%%)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration even if available')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Number of ROIs to process in a batch (default: 12)')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of worker threads (default: Auto)')
    parser.add_argument('--max-width', type=int, default=960,
                        help='Max frame width for processing (default: 960)')
    
    args = parser.parse_args()
    
    # Improved source selection logic to prioritize DEFAULT_FOLDER_PATH
    source = None
    if args.video:
        # User explicitly specified a video file
        if os.path.isfile(args.video):
            source = args.video
            print(f"Processing video file: {args.video}")
        else:
            print(f"Video file not found: {args.video}")
            print("Checking default video path...")
            if os.path.isfile(DEFAULT_FOLDER_PATH):
                source = DEFAULT_FOLDER_PATH
                print(f"Using default video path: {DEFAULT_FOLDER_PATH}")
            else:
                print(f"Default video not found. Falling back to camera")
                source = args.camera
    else:
        # No video explicitly specified, try using the default path
        if os.path.isfile(DEFAULT_FOLDER_PATH):
            source = DEFAULT_FOLDER_PATH
            print(f"Processing default video file: {DEFAULT_FOLDER_PATH}")
        else:
            # If default path is not a file, check if it's a directory
            if os.path.isdir(DEFAULT_FOLDER_PATH):
                # Get first video from directory
                video_files = get_video_files_from_folder(DEFAULT_FOLDER_PATH)
                if video_files:
                    source = video_files[0]
                    print(f"Using first video from default folder: {source}")
                else:
                    print("No videos found in default folder. Using camera.")
                    source = args.camera
            else:
                # Try default video paths as last resort
                existing_defaults = [path for path in DEFAULT_VIDEO_PATHS if os.path.isfile(path)]
                if existing_defaults:
                    source = existing_defaults[0]
                    print(f"Using default video: {source}")
                else:
                    print("No video files found. Using camera.")
                    source = args.camera
    
    # Always display video processing in real-time unless headless is specified
    headless = args.headless
    
    # Print configuration
    print("\nProcessing Configuration:")
    source_type = "Camera" if isinstance(source, int) else "Video file"
    source_name = f"{source}" if isinstance(source, int) else os.path.basename(source)
    print(f"- Source: {source_type} ({source_name})")
    print(f"- Processing mode: {'Headless' if headless else 'Interactive (showing video)'}")
    print(f"- Frame sampling rate: {args.fps} fps")
    print(f"- Maximum processing width: {args.max_width}px")
    print(f"- Batch size: {args.batch_size} ROIs per batch")
    
    # Update global parameter
    MAX_PLATE_CANDIDATES = 20
    
    # Override resize_if_needed for this run
    original_resize_if_needed = resize_if_needed
    def resize_if_needed_wrapper(frame, max_width=None):
        if max_width is None:
            max_width = args.max_width
        return original_resize_if_needed(frame, max_width)
    resize_if_needed = resize_if_needed_wrapper
    
    # Start processing
    main(
        source=source,
        frame_sample_rate=args.fps,
        headless=headless,
        camera_delay=args.delay,
        zoom_camera=args.zoom,
        batch_size=args.batch_size,
        threads=args.threads,
        smart_skip=True
    )
