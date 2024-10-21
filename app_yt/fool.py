import matplotlib.pyplot as plt
import numpy as np
import torch
# cargar un modelo previamente entrenado
import torchvision.models as models

from utils import compute_gradient, read_image, to_array

def func(inp, net=None, target=None):
    """calcular la probabilidad negativa del registro

    Parameters
    ----------
    inp : torch.Tensor
        Input image (single image batch).

    net : torch.nn.Module
        Imagenet ground truth label id.

    Returns
    -------
    loss : torch.Tensor
        Loss for the 'inp' image.
    """
    # hacemos un paso hacia a adelante
    out = net(inp)
    # calculamos las perdidas
    loss = torch.nn.functional.nll_loss(out, target=torch.LongTensor([target]))

    print(f"Loss: {loss.item()}")
    return loss

def attack(tensor, net, eps=1e-3, n_iter=50):
    """Run the Fast Sign Gradient Method (FSGM) attack.

    Parameters
    ----------
    tensor : torch.Tensor
        the input image of shape '(1, 3, 224, 224)'

    net : torch.nn.Module
        Classifier network.
    
    eps : float
        Determines how much we modify the image in single iteration.

    n_iter : int
        Number of iterations.

    Returns
    -------
    new_tensor : torch.Tensor
        New image that is a modificaciton of the input image that "fools" the classifier.
    """
    # clonamos el tensor de entrada
    new_tensor = tensor.detach().clone()
    # comprobamos cual es la predicción original en la entrada
    orig_prediction = net(tensor).argmax()
    print(f"Original prediction: {orig_prediction.item()}")
    # establecemos los gradientes en cero
    for i in range(n_iter):
        net.zero_grad()
        # obtnemos el gradiente con respecto a la entrada
        grad = compute_gradient(
            func, new_tensor, net=net, target=orig_prediction.item()
        )
        # esta linea reprecenta el corazon de nuestro ataque
        # Calculamos el signo del degradado y lo multiplicamos por un numero pequeño y lo agregamos a la imagen actual
        # new_tensor = torch.clamp(new_tensor + eps * grad.sign(), -2, 2)
        new_tensor = new_tensor.detach() + eps * grad.sign()  # Detach para crear un nuevo nodo de hoja
        new_tensor = torch.clamp(new_tensor, -2, 2)

        # tomamos un tensor y dejamos que la red lo clasifique
        new_prediction = net (new_tensor).argmax()
        # si la nueva predicción es diferente a la predicción original entonces logramos engañarla
        if orig_prediction != new_prediction:
            print(f"We fooled the network after {i} iterations!")
            print(f"New prediction: {new_prediction.item()}")
            break
        # devolvemos una nueva imagen, la prdicción original y la nueva predición
    return new_tensor, orig_prediction.item(), new_prediction.item()

if __name__ == "__main__":
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.eval()

    tensor = read_image("img.jpg")

    # Aquí ejecutamos nuestro ataque
    new_tensor, orig_prediction, new_prediction = attack(
        tensor, net, eps=1e-3, n_iter=100
    )

    # visualizamos como se veía la imagen antes y despues del ataque
    _, (ax_orig, ax_new, ax_diff) = plt.subplots(1, 3)

    # convertimos los tensores en matrices para visualizarlos
    # despues creamos una matriz para poder ver las diferencias con matplotlib
    arr = to_array(tensor)
    new_arr = to_array(new_tensor)
    diff_arr = np.abs(arr - new_arr).mean(axis=-1)
    diff_arr = diff_arr / diff_arr.max()

    ax_orig.imshow(arr)
    ax_new.imshow(new_arr)
    ax_diff.imshow(diff_arr, cmap="gray")

    ax_orig.axis("off")
    ax_new.axis("off")
    ax_diff.axis("off")

    ax_orig.set_title(f"Original: {orig_prediction}")
    ax_new.set_title(F"Modified: {new_prediction}")
    ax_diff.set_title("Difference")
    # finalmente formateamos un gráfico y luego lo guradamos en una imagen 
    plt.savefig("res_1.png")