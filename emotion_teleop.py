import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import cv2
import numpy as np
import pickle
import os
import urllib.request
from collections import deque
from ament_index_python.packages import get_package_share_directory

# MediaPipe 1.0+ Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class EmotionTeleopNode(Node):
    def __init__(self):
        super().__init__('emotion_teleop_node')
        
        # Publisher using TwistStamped for ROS 2 Jazzy / Gazebo compatibility
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Resolve ESN Model Path
        model_filename = 'teers_model.pkl'
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        candidate_paths = [
            os.path.join(script_dir, model_filename),
            os.path.join(os.path.expanduser('~'), 'teers_test', model_filename),
            os.path.join(os.path.expanduser('~'), 'teers_test', 'teers_model .pkl'),
        ]
        
        try:
            share_dir = get_package_share_directory('teers_robot_control')
            candidate_paths.append(os.path.join(share_dir, model_filename))
        except Exception:
            pass

        model_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            self.get_logger().error(f"Could not locate '{model_filename}'")
            raise SystemExit

        # Load ESN Model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.W_in = model['W_in']
            self.W_res = model['W_res']
            self.theta = model['theta']
            self.get_logger().info(f"TEERS ESN Model successfully loaded from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model file at {model_path}: {e}")
            raise SystemExit

        # Emotion Mappings
        self.EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise'}
        
        # ESN Buffers
        self.reservoir_size = self.W_res.shape[0]
        self.state = np.zeros((self.reservoir_size, 1))
        self.LEAKING_RATE = 0.35
        self.feature_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=15)
        self.smoothed_probs = None
        self.ALPHA = 0.12

        # Initialize MediaPipe Tasks FaceLandmarker
        task_model_path = os.path.join(script_dir, 'face_landmarker.task')
        if not os.path.exists(task_model_path):
            self.get_logger().info("Downloading MediaPipe FaceLandmarker task bundle...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, task_model_path)

        base_options = mp_python.BaseOptions(model_asset_path=task_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Video Capture
        self.cap = cv2.VideoCapture(0)
        
        # ROS 2 Loop Timer (30 Hz)
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)

    def softmax(self, x, temp=1.5):
        e_x = np.exp((x - np.max(x)) / temp)
        return e_x / e_x.sum()

    def get_pt(self, landmarks, idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=np.float32)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        # Solid black canvas to completely hide face/background for recording privacy
        display_frame = np.zeros_like(frame)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        
        detected_emotion = "Neutral / No Face"

        if detection_result.face_landmarks:
            lm = detection_result.face_landmarks[0]

            # Render key face mesh landmarks directly onto black privacy background
            for point in lm:
                px = int(point.x * w)
                py = int(point.y * h)
                cv2.circle(display_frame, (px, py), 1, (0, 255, 0), -1)

            # Draw facial outline contours (Lips & Eyes)
            eye_lip_indices = [33, 133, 159, 145, 263, 362, 386, 374, 61, 291, 13, 14, 70, 300]
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_lip_indices]
            for pt in pts:
                cv2.circle(display_frame, pt, 2, (255, 255, 255), -1)

            # Feature Extraction
            left_eye = self.get_pt(lm, 33)
            right_eye = self.get_pt(lm, 263)
            mouth_l = self.get_pt(lm, 61)
            mouth_r = self.get_pt(lm, 291)
            lip_t = self.get_pt(lm, 13)
            lip_b = self.get_pt(lm, 14)
            eyebrow_l = self.get_pt(lm, 70)
            eye_t = self.get_pt(lm, 159)

            inter_eye_dist = np.linalg.norm(left_eye - right_eye)
            if inter_eye_dist == 0:
                inter_eye_dist = 1e-6

            f1 = np.linalg.norm(lip_t - lip_b) / (np.linalg.norm(mouth_l - mouth_r) + 1e-6)
            corners_y = (mouth_l[1] + mouth_r[1]) / 2.0
            mouth_center_y = (lip_t[1] + lip_b[1]) / 2.0
            f2 = (mouth_center_y - corners_y) / inter_eye_dist
            f3 = np.linalg.norm(eyebrow_l - eye_t) / inter_eye_dist
            f4 = np.linalg.norm(mouth_l - mouth_r) / inter_eye_dist

            raw_feats = np.array([f1, f2, f3, f4], dtype=np.float32)
            self.feature_history.append(raw_feats)

            if len(self.feature_history) > 10:
                hist_arr = np.array(self.feature_history)
                mean_f = np.mean(hist_arr, axis=0)
                std_f = np.std(hist_arr, axis=0) + 1e-6
                norm_feats = (raw_feats - mean_f) / std_f
            else:
                norm_feats = raw_feats

            u = np.append(norm_feats, 1.0).astype(np.float32).reshape(-1, 1)
            state_update = np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, self.state))
            self.state = (1.0 - self.LEAKING_RATE) * self.state + self.LEAKING_RATE * state_update

            raw_output = np.dot(self.theta, self.state).flatten()[:6]
            current_probs = self.softmax(raw_output, temp=1.5)

            if self.smoothed_probs is None:
                self.smoothed_probs = current_probs
            else:
                self.smoothed_probs = (self.ALPHA * current_probs) + ((1.0 - self.ALPHA) * self.smoothed_probs)

            frame_pred = int(np.argmax(self.smoothed_probs))
            self.prediction_history.append(frame_pred)
            
            stable_idx = max(set(self.prediction_history), key=self.prediction_history.count)
            detected_emotion = self.EMOTIONS.get(stable_idx, "Neutral")

            if detected_emotion == 'Happy':
                twist_msg.twist.linear.x = 0.2
            elif detected_emotion == 'Angry':
                twist_msg.twist.angular.z = 0.5
            elif detected_emotion == 'Surprise':
                twist_msg.twist.linear.x = -0.15
            elif detected_emotion == 'Fear':
                twist_msg.twist.angular.z = -0.5
        else:
            self.state = np.zeros((self.reservoir_size, 1))
            self.smoothed_probs = None
            self.prediction_history.clear()

        self.cmd_vel_pub.publish(twist_msg)
        cv2.putText(display_frame, f"ROS 2 Teleop: {detected_emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('TEERS ROS 2 Emotion Control', display_frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EmotionTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
