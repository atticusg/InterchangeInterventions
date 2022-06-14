import torch
import copy

class InverseLinearLayer(torch.nn.Module):
    """The inverse of a given `LinearLayer` module."""
    def __init__(self, lin_layer):
        super().__init__()
        self.lin_layer = lin_layer

    def forward(self, x):
        output = torch.matmul(x, self.lin_layer.weight.T)
        return output

class LinearLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""
    def __init__(self, n, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(n,n).to(device), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)

class LayeredIntervenableModel(torch.nn.Module):
    def __init__(self, device=None, debug=False, use_wrapper=False):
        """

        Base class for all the PyTorch-based models.

        Parameters
        ----------
        model_layers: list of `torch.nn.Module`s
            `torch.nn.Sequential(model_layers)` is your model. The
            input of the module `model_layers[j]` must be a tensor with
            shape (-1, model_layer_dims[j]). The output must be
            a tensor of shape (-1, model_layer_dims[j+1]).

        model_layer_dims: list of ints
            The input and output dimensions of each layer, see model_layers
            definition above.

        device: string
            e.g. "cuda" or "cpu"

        Attributes
        -------
        labeled_layers: list of dictionaries
            The elements of this list are dictionaries with values
            "model", "disentangle", and "reentangle" and keys that
            are of type `torch.nn.Module`

        unlabeled_layer: list of `torch.nn.Module`s
            because it is a concatentation of the dictionary values
            of each element of `labeled_layers`. As such, this list is
            three times the length of `labeled_layers`
        """
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.debug = debug
        self.combiner = torch.nn.Sequential
        self.use_wrapper = use_wrapper

    def build_graph(self, model_layers, model_layer_dims):
        self.analysis_model = torch.nn.ModuleList()
        self.normal_model = torch.nn.ModuleList()
        self.labeled_layers = []
        # Initialize the orthogonal transformations used for disentanglement
        for index, model_layer in enumerate(model_layers[:-1]):
            self.normal_model.append(model_layer)
            self.analysis_model.extend([model_layer])
            if not self.debug:
                lin_layer = LinearLayer(model_layer_dims[index+1],
                                        self.device)
                lin_layer = torch.nn.utils.parametrizations.orthogonal(lin_layer)
                inverse_lin_layer = InverseLinearLayer(lin_layer)
                self.analysis_model.extend([lin_layer, inverse_lin_layer])
                self.labeled_layers.append({"disentangle":lin_layer, "reentangle":inverse_lin_layer, "model":model_layer})
            else:
                self.labeled_layers.append({"model":model_layer})



        self.labeled_layers.append({"model":model_layers[-1]})
        self.normal_model.extend([model_layers[-1]])
        self.analysis_model.extend([model_layers[-1]])

        self.normal_model = self.combiner(*self.normal_model)
        self.analysis_model = self.combiner(*self.analysis_model)

        self.set_analysis_mode(False)


    def set_analysis_mode(self, mode):
        self.analysis = mode
        if self.analysis:
            self.unfreeze_disentangling_parameters()
            self.freeze_model_parameters()
        else:
            self.freeze_disentangling_parameters()
            self.unfreeze_model_parameters()

    def set_device(self, device):
        self.bert.to(device)
        self.analysis_model.to(device)
        self.normal_model.to(device)

    def freeze_disentangling_parameters(self):
        """Freezes the orthogonal transformations used for disentangling"""
        for layer in self.labeled_layers:
            if "disentangle" in layer:
                layer["disentangle"].parametrizations.weight.original.requires_grad = False
                layer["disentangle"].parametrizations.weight.original.grad = None

    def freeze_model_parameters(self):
        """Freezes the model weights (for analysis purposes)"""
        for layer in self.labeled_layers:
            for param in layer["model"].parameters():
                param.requires_grad = False
                param.grad = None

    def unfreeze_disentangling_parameters(self):
        """Unfreezes the orthogonal transformations used for disentangling"""
        for layer in self.labeled_layers:
            if "disentangle" in layer:
                layer["disentangle"].parametrizations.weight.original.requires_grad = True

    def unfreeze_model_parameters(self):
        """Unfreezes the model weights (for training purposes)"""
        for layer in self.labeled_layers:
            for param in layer["model"].parameters():
                param.requires_grad = True


    def forward(self, X):
        """Computes a forward pass with input `X`."""
        if self.analysis:
            return self.analysis_model(X)
        else:
            return self.normal_model(X)

    def iit_forward(self,
            base,
            sources,
            intervention_ids,
            intervention_ids_to_coords):
        """
        Computes a multi-source interchange interventions with base input `base`
        and source inputs in the list `sources`. `intervention_ids` contain
        integers that indicate where to perform interventions and
        `intervention_ids_to_coords` is the dictionary used to translate these
        integers to coordinates denoting the layer, start index, and end index.
        """
        #unstack sources
        sources = [sources[:,j,:].squeeze(1).type(torch.FloatTensor).to(self.device)
           for j in range(sources.shape[1])]
        #translate intervention_ids to coordinates
        gets =  intervention_ids_to_coords[int(intervention_ids.flatten()[0])]
        sets = copy.deepcopy(gets)
        self.activation = dict()

        #retrieve the value of interventions by feeding in the source inputs
        for i, get in enumerate(gets):
            handlers = self._gets_sets(gets =[get],sets = None)
            source_logits = self.forward(sources[i])
            for handler in handlers:
                handler.remove()
            sets[i]["intervention"] =\
                self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']


        handlers = self._gets_sets(gets = None, sets = sets)
        counterfactual_logits = self.forward(base)
        for handler in handlers:
            handler.remove()
        return counterfactual_logits

    def intervention_wrapper(self, output, set):
        raise "Not Imlemented Error"

    def retrieval_wrapper(self, output, set):
        raise "Not Imlemented Error"

    def intervention(self, output, set):
        return torch.cat([output[:,:set["start"]], set["intervention"],
                            output[:,set["end"]:]],
                            dim = 1)

    def retrieval(self, output, get):
        return output[:,get["start"]: get["end"] ]

    def make_hook(self, gets, sets, layer, use_wrapper=False):
        """
        Returns a function that both retrieves the output values of a module it
        is registered too according to the coordinates in `gets` and also fixes
        the output of a module given the intervention values and coordinates in
        the elements of `sets`.
        """
        def hook(model, input, output):
            layer_gets, layer_sets = [], []
            if gets is not None:
                for get in gets:
                    if layer == get["layer"]:
                        layer_gets.append(get)
            for get in layer_gets:
                if use_wrapper:
                    self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']\
                        = self.retrieval_wrapper(output, get)
                else:
                    self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']\
                        = self.retrieval(output, get)

            if sets is not None:
                for set in sets:
                    if layer == set["layer"]:
                        layer_sets.append(set)
            for set in layer_sets:
                if use_wrapper:
                    output = self.intervention_wrapper(output, set)
                else:
                    output = self.intervention(output, set)
            return output
        return hook

    def _gets_sets(self,gets=None, sets = None):
        """
        Register hooks at the defined by `gets` and `sets` to the
        disentanglement module of each layer.
        """
        handlers = []
        for layer_num, layer in enumerate(self.labeled_layers):
            if self.analysis:
                hook = self.make_hook(gets,sets, layer_num)
                if "disentangle" in layer:
                    handler = layer["disentangle"].register_forward_hook(hook)
            else:
                hook = self.make_hook(gets,sets, layer_num, use_wrapper=self.use_wrapper)
                handler = layer["model"].register_forward_hook(hook)
            handlers.append(handler)
        return handlers

    def retrieve_activations(self, input, get, sets):
        """
        Returns the values at the coordinate in `get` for model input `input`
        when the interventions in `sets` are performed.
        """
        input = input.type(torch.FloatTensor).to(self.device)
        self.activation = dict()
        get_val = {get} if get is not None else None
        set_val = {set} if sets is not None else None
        handlers = self._gets_sets(gets=get_val, sets=set_val)
        logits = self.model(input)
        for handler in handlers:
            handler.remove()
        return self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']
