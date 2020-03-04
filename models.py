
from torch import nn




def build_model(tag):
    constructors = {

        'mlp': (
            lambda: VanillaMLP(
                input_size=2,
                hidden_size=50,
                output_size=3,
            )
        )
    }

    return constructors[tag]()


class VanillaMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        hidden_size = 50
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.network = nn.Sequential(
           nn.Linear(input_size, hidden_size),
          
#           nn.Sigmoid(),
           nn.ReLU(),
           nn.Linear(hidden_size, output_size)
           
        )
        nn.init.xavier_uniform_(self.network[0].weight)

    def forward(self, x):
        return { 'logits': self.network(x), 'gain': 0.0}

