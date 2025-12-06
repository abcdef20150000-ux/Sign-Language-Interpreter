import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def get_finger_states(landmarks):
    # Fingertip indices for index, middle, ring, pinky
    tips = [8, 12, 16, 20]
    # DIP joint indices (one level below fingertip)
    dips = [6, 10, 14, 18]
    # [(x_coordinate , y_coordinate ), ( , ), ( , )]
    # Determine if fingers are open by comparing y-coordinates
    # A finger is open if its tip is higher (smaller y) than its DIP joint
    fingers = [landmarks[t][1] < landmarks[d][1] for t, d in zip(tips, dips)]

    # Determine if the thumb is open by comparing x-coordinates
    # Thumb is open when its tip is farther to the right than its previous joint
    thumb_open = landmarks[4][0] > landmarks[3][0]

    return fingers, thumb_open


def detect_phrase(landmarks):
    # Get the states of the four fingers and the thumb
    fingers, thumb_open = get_finger_states(landmarks)

    # Phrase 1: Hello -> all fingers open including thumb
    if fingers == [True, True, True, True] and thumb_open:
        return "Hello"

    # Phrase 2: I Love You -> thumb + index + pinky open
    if fingers == [True, False, False, True] and thumb_open:
        return "I Love You"

    # Phrase 3: Stop -> all fingers closed including thumb
    if fingers == [False, False, False, False] and not thumb_open:
        return "Stop"

    # Phrase 4: Yes -> only thumb open while all other fingers closed
    if fingers == [False, False, False, False] and thumb_open:
        return "Yes"

    # Phrase 5: No -> thumb pointing downward
    if landmarks[4][1] > landmarks[3][1]:
        return "No"

    # Phrase 6: Thank You -> hand raised (index tip higher than its base)
    if landmarks[8][1] < landmarks[5][1]:
        return "Thank You"

    return ""


# Main function for camera processing
def main():
    # Open webcam (default camera index 0)
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands model
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Get frame dimensions
            h, w, c = frame.shape

            # Convert BGR (OpenCV format) to RGB (MediaPipe format)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            results = hands.process(rgb)

            text_output = ""

            # Check if any hands were detected
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    # Convert normalized coordinates to pixel values
                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.append((int(lm.x * w), int(lm.y * h)))

                    # Detect the phrase based on hand posture
                    text_output = detect_phrase(landmarks)

            # Display detected phrase on the screen
            if text_output != "":
                cv2.putText(frame, text_output, (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 3)

            # Show the result frame
            cv2.imshow("Sign Phrases Recognition", frame)

            # Quit program by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()