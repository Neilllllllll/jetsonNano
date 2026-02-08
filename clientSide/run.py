import cv2
import mss
import numpy as np
import socket
import struct

def stream_screen(ip_jetson, port=5005):
    # 1. Configuration du Socket UDP
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_address = (ip_jetson, port)

    # 2. Configuration de la capture d'écran
    sct = mss.mss()
    # On définit la zone à capturer (tout l'écran ou une portion)
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    print(f"Streaming vers {ip_jetson}:{port}...")

    try:
        while True:
            # A. Capture l'écran
            img = np.array(sct.grab(monitor))

            # B. Prétraitement
            # On convertit de BGRA (standard mss) vers BGR (standard OpenCV)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # Optionnel : Redimensionner pour économiser de la bande passante
            frame = cv2.resize(frame, (640, 480))

            # C. Compression JPEG (Crucial pour le réseau)
            # '90' est la qualité du JPEG (0-100)
            result, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            if result:
                data = encoded_img.tobytes()
                print(f"Taille du paquet envoyé : {len(data)} octets")
                # D. Envoi de la taille puis de l'image
                # Si l'image est trop grosse (> 65KB), l'UDP peut échouer. 
                # Pour de la HD, il faudrait fragmenter ou utiliser TCP/RTSP.
                if len(data) < 65000:
                    client_socket.sendto(data, dest_address)

    except KeyboardInterrupt:
        print("Arrêt du stream.")
    finally:
        client_socket.close()

# Remplacer par 192.168.55.1 si connecté en USB
stream_screen("192.168.1.17")