from PIL import Image
import torch
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor)

# funciones utiles 

def compute_gradient(func, inp, **kwargs):
    """ Calcula el gradiente con respecto a 'inp'
    Parámetros
    ----------
    func : callable
        Toma 'inp' y 'kwargs' y retorna un único elemento tensor.

    inp : torch.Tensor
        El tensor lo queremos para obtener los gradientes. necesita ser un nodo de hoja.

    **kwargs : dict
        argumentos adicionales en 'func'

    Returns
    -------
    grad : torch.tensor
        Tensor de la misma forma como 'imp' que es la reprecentación de un gradiente
    """
    # el tensor es un nodo de hoja 
    # torch calcula el gradiente con respecto a este tensor
    inp.requires_grad = True
    # ejecutamos inp en func
    # reprecenta una red reunoral seguido de algún criterio de perdida
    loss = func(inp, **kwargs)
    # aqui le decimos a torch que calcule los gradientes
    loss.backward()
    # deshacemos lo que hicimos en el primer paso de la fucunción
    inp.requires_grad = True
    # devuelve el tensor de gradiente real
    return inp.grad.data

def read_image(path):
    """ Carga imagenes del almacenmiento y lo convierte para torch.Tensor

    Parámetros
    ----------
    path : str
        Ruta de la imagen.

    Returns
    -------
    tensor : torch.Tensor
        Lote de una sola muestra nuestras imagenes (lista para ser usada con redes preentrenadas). Laa forma es '(1, 3, 224, 224)'.
    """
    img = Image.open(path)

    # contine los pasos
    # 1. Cambiamos el tamaño de nuestra imagen para que sea un cuadrado
    # 2. Recortamos un cuadro en el centro
    # 3. Convertimos la imagen en tensor
    # 4. Finalmente la normalizamos
    transform = Compose([Resize(256),
                         CenterCrop(224),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])
    # agregamos una dimensión adicional que reprecenta el lote
    tensor_ = transform(img)
    tensor = tensor_.unsqueeze(0)

    return tensor

def to_array(tensor):
    """Convert torch.Tensor ton np.ndarray

    Parameters
    ----------
    tensor : torch.Tensor 
        Tensor of shape '(1, 3, *, *)' rerecenting one sample batch of images.

    Returns
    -------
    arr : np.ndarray
        Array of shape '(*, *, 3)' representing an image that can be plotted directly.
    """
    # Eliminamos la dimención del lote
    tensor_ = tensor.squeeze()
    # Esta transformación deshace la normalización de 'read_image'
    unnormalize_transform = Compose([Normalize(mean=[0, 0, 0], 
                                               std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    Normalize(mean=[-0.485, -0.456, -0.406],
                                               std=[1, 1, 1])])
    # Ejecutamos la transformación
    arr_ = unnormalize_transform(tensor_)
    # Permutamos las dimenciones pata que los canales sean la última dimensión
    arr = arr_.permute(1, 2, 0).detach().numpy()

    return arr

def scale_grad(grad):
    """Scale gradient tensor

    Paramenters
    -----------
    grad : torch.Tensor
        Gradient of shape '(1, 3, *, *)'

    Returns
    -------
    grad_arr : nparray
        Array of shape '(*, *, 1)'.    
    """
    # Calculamos los valores absolutos, luego tomamos un promedio sobre la dimensión del canal y finalmente colocamos la dimensión del canal hasta el final.
    grad_arr = torch.abs(grad).mean(dim=1).detach().permute(1, 2, 0)
    # Aquí dividimos todos los elementos del tensor por un número que está muy cerca del máximo (elegí el cuantil 98)
    grad_arr /= grad_arr.quantile(0.98)
    # Aquí nos aseguramos que todos los elementos esten en el rango (0, 1)
    grad_arr = torch.clamp(grad_arr, 0, 1)

    # Finalmente lo convertimos en una matriz numpy
    return grad_arr.numpy()