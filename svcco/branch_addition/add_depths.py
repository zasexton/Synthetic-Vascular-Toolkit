import numba as nb

@nb.jit(nopython=True,cache=True,nogil=True)
def add_depths(data, edge):
    search = [int(edge)]
    while len(search) > 0:
        for idx in search:
            if data[idx, 15].item() >= 0 or data[idx, 16].item() >= 0:
                left = int(data[idx, 15].item())
                right = int(data[idx, 16].item())
                data[left, 26] += 1
                data[right, 26] += 1
                search.remove(idx)
                search.append(left)
                search.append(right)
            else:
                search.remove(idx)