import numba as nb

@nb.jit(nopython=True,cache=True,nogil=True)
def add_flow(data,parent,flow):
    updated_flows = []
    while parent >= 0:
        data[parent,22] = flow + data[parent,22]
        updated_flows.append(parent)
        parent = int(data[parent,17].item())
    return updated_flows
