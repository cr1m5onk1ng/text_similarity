import torch
from torch import nn
import torch_scatter
import torch_sparse

def unpack_sparse_tensor(sparse_tensor):
    pieces = (sparse_tensor._indices(), sparse_tensor._values(), torch.LongTensor(list(sparse_tensor.shape)))
    return pieces

def repack_sparse_tensor(*pieces):
    ii, vv, size = pieces
    if isinstance(size, torch.Size):
        ...
    else:
        size = torch.Size(size.cpu().tolist())
    return torch.sparse.FloatTensor(ii, vv, size).coalesce()


class StructuredLogits(nn.Module):
    def __init__(self, adjacency=None, train_adjecency=False, renormalize=False):
        super(StructuredLogits, self).__init__()
        self.adjacency = adjacency
        self.train_adjacency = train_adjecency
        self.renormalize = renormalize

        if adjacency is not None:

            ii, vv, size = unpack_sparse_tensor(adjacency)
            ii = torch.nn.Parameter(ii, requires_grad=False)
            vv = torch.nn.Parameter(vv, requires_grad=adjacency_trainable)
            size = torch.nn.Parameter(size, requires_grad=False)

            self.adjacency_pars = torch.nn.ParameterList([ii, vv, size])
            self._coalesce(self.adjacency_pars)

            if not self.renormalize:
                self._initialize_to_1_over_n(self.adjacency_pars)

            self.self_loops = None

        else:
            self.adjacency_pars = None

    def forward(self, logits):
        if self.adjacency_pars is not None:
            logits_old = logits
            neighbors = self._spmm(logits, self.adjacency_pars)
            if self.renormalize:
                neighbor_sum = self._get_row_sum(self.adjacency_pars)
                neighbors = neighbors / neighbor_sum.view(1, 1, -1)
            logits = neighbors + logits_old
        return logits

    def _coalesce(self, params):
        ii, vv, size = params
        coalesced_ii, coalesced_vv = torch_sparse.coalesce(ii, vv, *size, op='max')
        ii.data = coalesced_ii
        vv.data = coalesced_vv
        return params

    def _get_row_sum(self, params):
        ii, vv, size = params
        row_sum = torch_scatter.scatter_add(vv, ii[0], dim_size=size[1])
        return row_sum

    def _get_col_sum(self, params):
        ii, vv, size = params
        col_sum = torch_scatter.scatter_add(vv, ii[1], dim_size=size[0])
        return col_sum

    def _initialize_to_1_over_n(self, params, sum='none'):
        if sum == 'none':
            return params
        ii, vv, size = params
        vv.data[:] = 1
        if sum == 'row':
            row_sum = self._get_row_sum(params)
            vv.data[:] = 1 / row_sum[ii[0]]
        elif sum == 'col':
            col_sum = self._get_col_sum(params)
            vv.data[:] = 1 / col_sum[ii[1]]
        return params

    def _spmm(self, inp, params):
        ii, vv, size = params
        old_inp_size = inp.size()
        inp_flat_T = inp.view(-1, inp.size(-1)).t() 
        out_flat = torch_sparse.spmm(
            ii, vv,
            m=size[0], n=size[1],
            matrix=inp_flat_T
        ).t()
        out = out_flat.view(*old_inp_size)
        return out

    @property
    def device(self):
        return next(self.parameters()).device






