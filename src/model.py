import torch

class ChessTranformerModel(torch.nn.Module):

    def __init__(self, input_dim, model_dim, bin_dim, n_heads, num_layers, dim_feedforward=None, dropout=None):
        super(ChessTranformerModel, self).__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.bin_dim = bin_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else 2 * embedding_dim
        self.dropout = dropout if dropout is not None else 0.1
        

        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            self.model_dim,
            self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.transformer_encoder_layer,
            self.num_layers
        )

        self.project_to_model_dim = torch.nn.Linear(self.input_dim, self.model_dim)
        self.projection_to_bin_dim = torch.nn.Linear(self.model_dim, self.bin_dim)
        self.mateness_head = torch.nn.Linear(self.model_dim, 1)

    def forward(self, board_representation):
        x = self.project_to_model_dim(board_representation)
        
        x = self.transformer_encoder(x)
        
        # Average pooling over the sequence dimension
        x = torch.mean(x, 1)

        win_prob_logits = self.projection_to_bin_dim(x)
        mateness_logit = self.mateness_head(x)

        return win_prob_logits, mateness_logit