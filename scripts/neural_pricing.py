"""One scalar price function shared by price-only and differential pilots.

Physical input/price units wrap fixed, fit-only normalization. No batch coupling,
dropout, clipping or separately predicted Greek outputs. Vega labels are disabled.
"""
import math
import torch
from torch import nn

FEATURES = ('S', 'K', 'T', 'V0', 'R', 'Q', 'IsCall')
ACTIVE_GREEKS = ('Delta', 'Gamma', 'Theta')


class PriceNetwork(nn.Module):
    def __init__(self, scalers, seed=42):
        super().__init__()
        means = [scalers['mean'][k] for k in FEATURES]
        scales = [scalers['scale'][k] for k in FEATURES]
        pm, ps = scalers['mean']['OptionMid'], scalers['scale']['OptionMid']
        if not all(math.isfinite(v) for v in means+scales+[pm, ps]) or min(scales+[ps]) <= 0:
            raise ValueError('Normalization must be finite with positive scales')
        for name, values in [('input_mean', means), ('input_scale', scales),
                             ('price_mean', pm), ('price_scale', ps)]:
            self.register_buffer(name, torch.tensor(values, dtype=torch.float64))
        # Isolate initialization from the caller's minibatch/random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers = []
            for i, o in [(7,128),(128,64),(64,32),(32,1)]:
                layer = nn.Linear(i,o,dtype=torch.float64)
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
                if o != 1:
                    layers.append(nn.Softplus())
            self.layers = nn.Sequential(*layers)

    def forward(self, raw):
        if raw.ndim != 2 or raw.shape[1] != len(FEATURES) or len(raw) == 0:
            raise ValueError('Require a nonempty N x 7 input tensor in physical units')
        if not torch.isfinite(raw).all():
            raise ValueError('Nonfinite pricing input')
        if (raw[:,:3] <= 0).any() or (raw[:,3] < 0).any():
            raise ValueError('Require S/K/T>0 and V0>=0')
        if not ((raw[:,6] == 0) | (raw[:,6] == 1)).all():
            raise ValueError('IsCall must be zero or one')
        normalized = (raw-self.input_mean)/self.input_scale
        return self.price_mean + self.price_scale*self.layers(normalized).squeeze(-1)


def price_and_greeks(model, raw, *, training_graph=False):
    """Model must act independently on each row. S/K/T/V0/R/Q/type held fixed
    except for the differentiated coordinate. Theta is calendar-time theta.
    InitialVolSensitivity is dP/d(sqrt(V0)), never vendor BSM Vega.
    """
    with torch.enable_grad():
        x = raw.detach().clone().requires_grad_(True)
        price = model(x)
        first = torch.autograd.grad(price.sum(), x, create_graph=True)[0]
        delta = first[:,0]
        second = None
        if delta.requires_grad:
            second = torch.autograd.grad(delta.sum(), x, create_graph=training_graph,
                                          retain_graph=True, allow_unused=True)[0]
        gamma = x[:,0]*0 if second is None else second[:,0]
        result = dict(Price=price, Delta=delta, Gamma=gamma, Theta=-first[:,2],
                      InitialVolSensitivity=2*torch.sqrt(x[:,3])*first[:,3])
    return result if training_graph else {k:v.detach() for k,v in result.items()}


def pilot_loss(predictions, targets, masks, scalers, *, differential, weights=None):
    """Physical-unit residuals transformed with frozen training-only scales.
    Index masked labels BEFORE arithmetic so missing-label NaNs never backpropagate.
    """
    weights = dict(price=1.,Delta=.5,Gamma=.1,Theta=.2,Vega=0.) if weights is None else weights
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Loss weights must be finite and nonnegative')
    if weights.get('Vega',0) != 0 or ('Vega' in masks and masks['Vega'].any()):
        raise ValueError('Vendor Vega cannot supervise initial-volatility sensitivity')
    price = predictions['Price']
    if price.ndim != 1 or targets['Price'].shape != price.shape or len(price)==0:
        raise ValueError('Price predictions and targets must be matching nonempty vectors')
    if not torch.isfinite(price).all() or not torch.isfinite(targets['Price']).all():
        raise ValueError('All price rows require finite prediction and target')
    scale = scalers['scale']
    components = {'price': ((price-targets['Price'])/scale['OptionMid']).square().mean()}
    count = {'price': len(price)}
    total = weights['price']*components['price']
    if differential:
        factors = dict(Delta=scale['S']/scale['OptionMid'],
                       Gamma=scale['S']**2/scale['OptionMid'],
                       Theta=-scale['T']/scale['OptionMid'])
        for key in ACTIVE_GREEKS:
            mask = masks[key]
            if mask.dtype != torch.bool or mask.shape != price.shape:
                raise ValueError('Greek masks must be boolean vectors matching the price rows')
            if predictions[key].shape != price.shape or targets[key].shape != price.shape:
                raise ValueError('Greek vectors must match price rows')
            if not torch.isfinite(targets[key][mask]).all() or not torch.isfinite(predictions[key][mask]).all():
                raise ValueError('Enabled Greek rows must be finite')
            rms = scalers['derivative_loss_rms'][key]
            if not math.isfinite(rms) or rms <= 0:
                raise ValueError('Greek loss RMS must be finite and positive')
            count[key] = int(mask.sum())
            components[key] = (((predictions[key][mask]-targets[key][mask])*factors[key]/rms).square().mean()
                               if mask.any() else components['price']*0)
            total = total + weights[key]*components[key]
    return total, components, count
