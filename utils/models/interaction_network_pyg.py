import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

# class InteractionNetwork(MessagePassing):
#     def __init__(self, aggr='add', flow='source_to_target', hidden_size=40):
#         super(InteractionNetwork, self).__init__(aggr=aggr,
#                                                  flow=flow)
#         self.R1 = RelationalModel(10, 4, hidden_size)
#         self.O = ObjectModel(7, 3, hidden_size)
#         self.R2 = RelationalModel(10, 1, hidden_size)
#         self.n_neurons = hidden_size

#     def forward(self, data):
#         x = data.x
#         edge_index, edge_attr = data.edge_index, data.edge_attr
#         print(f"edge_index: {edge_index.shape}")
#         print(f"edge_attr: {edge_attr.shape}")
#         # print(f"edge_attr: {edge_attr}")
#         x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr)
#         print(f"x_tilde: {x_tilde.shape}")

#         if self.flow=='source_to_target':
#             r = edge_index[1]
#             s = edge_index[0]
#         else:
#             r = edge_index[0]
#             s = edge_index[1]

#         m2 = torch.cat([x_tilde[r],
#                         x_tilde[s],
#                         self.E], dim=1)
#         return torch.sigmoid(self.R2(m2))

#     def message(self, x_i, x_j, edge_attr):
#         # x_i --> incoming
#         # x_j --> outgoing        
#         print(f"x_i: {x_i.shape}")
#         print(f"x_j: {x_j.shape}")
#         # print(f"edge_attr: {edge_attr}")
#         m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
#         self.E = self.R1(m1)
#         return self.E

#     def update(self, aggr_out, x):
#         c = torch.cat([x, aggr_out], dim=1)
#         return self.O(c) 

"""
Hyeon-Seo code begin
"""
class ResidualBlock(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size

    def forward(self, h_prev, h_after):
        assert(h_prev.shape == h_after.shape)
        return h_prev+h_after

class NodeEncoder(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, output_size):
        super(NodeEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.encoder(x)


class EdgeEncoder(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, output_size):
        super(EdgeEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.encoder(x)

"""
Hyeon-seo code end
"""

class InteractionNetwork(MessagePassing):
    def __init__(self, aggr='add', flow='source_to_target', hidden_size=40):
        super(InteractionNetwork, self).__init__(aggr=aggr,
                                                 flow=flow)
        self.n_neurons = hidden_size
        # self.R1 = RelationalModel(10, 4, hidden_size)
        # self.O = ObjectModel(7, 3, hidden_size)
        # self.R2 = RelationalModel(10, 1, hidden_size)
        # self.res_block = ResidualBlock(3)
        self.out_channels = 5#128
        self.R1 = RelationalModel(3*self.out_channels, self.out_channels, hidden_size)
        self.O = ObjectModel(self.out_channels, self.out_channels, hidden_size)
        self.R2 = RelationalModel(3*self.out_channels, 1, hidden_size)
        self.res_block = ResidualBlock(self.out_channels)
        # self.node_encoder = NodeEncoder(3, self.out_channels)
        # self.edge_encoder = EdgeEncoder(4, self.out_channels)
        self.node_encoder = nn.Linear(3, self.out_channels)
        self.edge_encoder = nn.Linear(4, self.out_channels)

    def forward(self, data):
        x = data.x
        x = self.node_encoder(x)
        # print(f"Node Encoder output max: {torch.max(x)}")
        # print(f"Node Encoder output abs means: {torch.mean(torch.abs(x))}")
        print(f"Node Encoder output mean: {torch.mean(x)}. std: {torch.std(x)}")
        edge_index, edge_attr = data.edge_index, data.edge_attr
        edge_attr = self.edge_encoder(edge_attr)
        # print(f"Edge Encoder output max: {torch.max(edge_attr)}")
        # print(f"Edge Encoder output abs means: {torch.mean(torch.abs(edge_attr))}")
        print(f"Edge Encoder output mean: {torch.mean(edge_attr)}. std:{torch.std(edge_attr)}")
        # print(f"Edge Encoder output: {edge_attr}")
        # print(f"edge_index: {edge_index.shape}")
        # print(f"edge_attr: {edge_attr.shape}")
        # print(f"edge_attr: {edge_attr}")
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # print(f"x_tilde: {x_tilde.shape}")

        if self.flow=='source_to_target':
            r = edge_index[1]
            s = edge_index[0]
        else:
            r = edge_index[0]
            s = edge_index[1]

        m2 = torch.cat([x_tilde[r],
                        x_tilde[s],
                        self.E], dim=1)
        # x_j =  x_tilde[s]
        # m2 = self.E + x_j
        # print("forwarding")
        output = self.R2(m2)
        # print(f"R2 output max: {torch.max(output)}")
        print(f"R2 output mean: {torch.mean(output)}, std: {torch.std(output)}")
        return torch.sigmoid(output)

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing        
        # print(f"x_i: {x_i.shape}")
        # print(f"x_j: {x_j.shape}")
        # print(f"edge_attr: {edge_attr}")
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        # m1 = x_j + edge_attr
        self.E = self.R1(m1)
        # print(f"R1 output max: {torch.max(self.E)}")
        print(f"R1 output mean: {torch.mean(self.E)}, std: {torch.std(self.E)}")
        # print("message passing")
        return self.E

    def update(self, aggr_out, x):
        # c = torch.cat([x, aggr_out], dim=1)
        c = x + aggr_out
        # c = x
        residual = c
        output = self.O(c) 
        print(f"O output mean: {torch.mean(output)}, std: {torch.std(output)}")
        # return output
        return self.res_block(residual, output) 

    # def update(self, aggr_out, x):
        #aggr_out is the output of the aggregate()
        # c = torch.cat([x, aggr_out], dim=1)
        # c=x
        # print("node update")
        # residual =c 
        # c = self.O(c) 
        # result = self.res_block(residual,c) 
        # print(residual+ c ==result)
        # return result



# class IN_block(MessagePassing):
#     def __init__(self, index: int, aggr='add', flow='source_to_target', hidden_size=40):
#         super(IN_block, self).__init__(aggr=aggr, flow=flow)
#         exec(f"self.R_{index} = RelationalModel(6, 4, hidden_size)")
#         exec(f"self.O_{index} = ObjectModel(7, 3, hidden_size)")
#         self.n_neurons = hidden_size
#     def forward(self, data):
#         x = data.x
#         edge_index, edge_attr = data.edge_index, data.edge_attr
#         x_tilde = self.propagate(edge_index, x=x)
        
        
#         return m2

#     def message(self, x_i, x_j):
#         # x_i --> incoming
#         # x_j --> outgoing        
#         m1 = torch.cat([x_i, x_j], dim=1)
#         self.E = self.R1(m1)
#         return self.E

#     def update(self, aggr_out, x):
#         c = torch.cat([x, aggr_out], dim=1)
#         return self.O(c) 

# class IN_block(nn.Module)::
#     def __init__(self, size: int, hidden_size=40):
#         self.graph_blocks_ = nn.ModuleList()
#         for index in range(size):
#             layer = IN_block(index)
#             self.graph_blocks_.append(layer)
#         self.final_layer_ = RelationalModel(10, 1, hidden_size)
#     def forward(self, data):
#         for layer in self.graph_blocks_:

            