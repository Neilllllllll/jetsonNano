import torch
import torchvision

def check_torchvision_cuda():
    print("\n--- Vérification de la compatibilité CUDA de Torchvision ---")
    if torch.cuda.is_available():
        try:
            # Test de chargement d'un modèle pré-entraîné de Torchvision
            model = torchvision.models.resnet18(pretrained=True).cuda()
            print("Torchvision est compatible avec CUDA et le modèle a été chargé sur le GPU.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle sur le GPU : {e}")
        # Libérer la mémoire GPU
        finally:            
            del model
            torch.cuda.empty_cache()
    else:
        print("CUDA n'est pas disponible, donc la compatibilité CUDA de Torchvision ne peut pas être vérifiée.")

def check_torch_cuda():
    print("\n--- Vérification de la compatibilité CUDA de PyTorch ---")
    if torch.cuda.is_available():
        print("CUDA est disponible. PyTorch peut utiliser le GPU.")
        print(f"Nombre de GPU disponibles : {torch.cuda.device_count()}")
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA n'est pas disponible. PyTorch utilisera le CPU.")

def check_opencv():
    try:
        import cv2
        print(f"Version d'OpenCV : {cv2.__version__}")
    except ImportError:
        print("OpenCV n'est pas installé. Veuillez l'installer pour utiliser les fonctionnalités de traitement d'image.")

def check_yolov5():
    # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

    # Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
    img = "https://ultralytics.com/images/zidane.jpg"  # Example image

    # Perform inference (handles batching, resizing, normalization automatically)
    results = model(img)

    # Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
    results.print()  # Print results to console
    results.show()  # Display results in a window
    results.save()  # Save results to runs/detect/exp

# Cette fonction effectue un warmup en chargeant un modèle YOLOv5 et en effectuant une inférence sur une image d'exemple.
def warmup():
    check_torchvision_cuda()
    check_torch_cuda()
    check_opencv()
    #check_yolov5()