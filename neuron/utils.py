
from neuron.tensor import Tensor
    
def print_dependency(tensor: Tensor, tab=0) -> None:

    print(tensor.grad)

    for dep in tensor.depends_on:

        print(f"{'-' * tab} {dep.tensor.shape} {dep.gradfn_name}")
        print(dep.tensor.grad)

        print_dependency(dep.tensor, tab=2)


