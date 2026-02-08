import torch
import torchvision

# Cette fonction vérifie les versions de PyTorch et Torchvision, ainsi que la disponibilité de CUDA.
def check_pytorch_torchvision_cuda():
    print("--- Vérification du système ---")
    print(f"Version de PyTorch : {torch.__version__}")
    print(f"Version de Torchvision : {torchvision.__version__}")
    print(f"Version de CUDA interne à PyTorch : {torch.version.cuda}")
    print(f"Version de cuDNN : {torch.backends.cudnn.version()}")

    print("\n--- Test CUDA ---")
    cuda_disponible = torch.cuda.is_available()
    print(f"CUDA est disponible : {cuda_disponible}")

    # Si CUDA est disponible, affichez des informations sur le GPU et effectuez un test de calcul.
    if cuda_disponible:
        print(f"Nombre de GPU : {torch.cuda.device_count()}")
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
        
        # Test de calcul sur le GPU
        x = torch.rand(3, 3).cuda()
        print("\nTest de calcul réussi : Un tenseur a été créé sur le GPU.")
        print(f"Emplacement du tenseur : {x.device}")
    else:
        print("\nATTENTION : CUDA n'est pas détecté par PyTorch.")
        print("Vérifiez que vous avez bien installé la version 'NVIDIA-PyTorch' spécifique aux Jetson.")

# Cette fonction effectue un warmup en chargeant un modèle YOLOv5 et en effectuant une inférence sur une image d'exemple.
def warmup():
    check_pytorch_torchvision_cuda()
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