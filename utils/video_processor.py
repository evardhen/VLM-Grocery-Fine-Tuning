import cv2

class VideoFrameExtractor:
    def __init__(self, video_path: str, extraction_fps: float, encode: bool = True):
        """
        Initialize the extractor with the video path, desired extraction FPS, and an option to encode frames.

        Parameters:
            video_path (str): Path to the video file.
            extraction_fps (float): Number of frames per second to extract.
            encode (bool): If True, encode frames as JPEG bytes. If False, return raw numpy arrays.
                           Defaults to True.
        """
        self.video_path = video_path
        self.extraction_fps = extraction_fps
        self.encode = encode

    def extract_frames(self):
        """
        Extract frames from the video at the specified FPS.

        Returns:
            List[Union[bytes, numpy.ndarray]]: A list of extracted frames.
                If encode is True, each frame is JPEG-encoded and returned as bytes.
                Otherwise, frames are returned as numpy arrays.

        Raises:
            IOError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        
        # Get the video's native frames per second
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the interval between frames to extract
        interval = max(int(round(video_fps / self.extraction_fps)), 1)
        
        output_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                if self.encode:
                    # Encode the frame to JPEG format, then convert to bytes
                    success, encoded = cv2.imencode('.jpg', frame)
                    if success:
                        output_frames.append(encoded.tobytes())
                    else:
                        print(f"Warning: Could not encode frame at index {frame_idx} from {self.video_path}")
                else:
                    output_frames.append(frame)
            frame_idx += 1
        
        cap.release()
        return output_frames
