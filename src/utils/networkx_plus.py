import networkx as nx


def drop_weights(G):
    """
    Drop the weights from a networkx weighted graph.
    """
    for node, edges in nx.to_dict_of_dicts(G).items():
        for edge, attrs in edges.items():
            attrs.pop('weight', None)
    return G
