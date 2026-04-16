import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

import rclpy
from std_msgs.msg import Int32MultiArray


#Fonction utilitaire
def draw_manual(image, landmarks):
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in points:
        cv2.circle(image, pt, 4, (0, 255, 0), -1)

#Def des constantes
val_fist = 0
val_palm = 0
val_obj = 0
dernier_chiffre = 0
current_holding_gesture = "None"


gestures_full = [] #Utile au débug
gestures_chosen = []#Utile au débug

#Gestion de l'anti rebond (0.5sec à 30FPS)
thresh_choice = 15
hold_frames = 0



#Init communication ROS
rclpy.init()
node = rclpy.create_node('gesture_publisher_flat')
publisher = node.create_publisher(Int32MultiArray ,'gestures', 10)

#Paramétrage du modèle de reconnaissance
model_path = "./my_model/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,  
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
recognizer = GestureRecognizer.create_from_options(options)


#Lancement du pipeline de reconnaissance de geste
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


try:
    while len(gestures_chosen) < 6: #On a 3 chiffres et 3 actions associées --> 6 valeurs
  
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.flip(frame, 1) # Enlève l'effet mirroir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        recognition_result = recognizer.recognize(mp_image)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            nom_geste = top_gesture.category_name
            score = top_gesture.score

            node.get_logger().info(f"{nom_geste}") #DÉBUG
            print(nom_geste) #DÉBUG
            gestures_full.append(nom_geste)

            #Anti-Rebond (détection d'un geste : maintenir)
            if nom_geste == current_holding_gesture:
                hold_frames += 1
            else:
                current_holding_gesture = nom_geste
                hold_frames = 1

            if hold_frames == thresh_choice and nom_geste != "None":
                nb_gestures = len(gestures_chosen)

                if nb_gestures % 2 == 0 and nom_geste in ["1", "2", "3"]:
                    gestures_chosen.append(nom_geste)
                    dernier_chiffre = int(nom_geste)

                elif nb_gestures % 2 == 1 and nom_geste in ["Fist", "palm","O"]:
                    if nom_geste == "Fist":
                        val_fist = dernier_chiffre
                    elif nom_geste == "palm":
                        val_palm = dernier_chiffre
                    elif nom_geste == "O":
                        val_obj = dernier_chiffre
                    gestures_chosen.append(nom_geste)
                else:
                    print("Mauvais Geste")

            texte = f"Geste: {nom_geste} ({score:.2f})"
            cv2.putText(
                frame, texte, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            if recognition_result.hand_landmarks:
                draw_manual(frame, recognition_result.hand_landmarks[0])

        cv2.imshow("Custom AI Gesture - RealSense", frame)
        print(gestures_chosen)
        print(current_holding_gesture)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    #Envoi d'un multiarray en ROS, toujours dans l'ordre objet, départ arrivée
    #On peut faire les gestes dans l'ordre désiré, l'envoi ros sera toujours bien formaté
    print([val_obj,val_fist,val_palm])
    msg = Int32MultiArray()
    msg.data = [val_obj,val_fist,val_palm] 
    publisher.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)

finally:
    print(gestures_chosen)
    pipeline.stop()
    cv2.destroyAllWindows()
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
