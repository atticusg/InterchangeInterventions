from typing import Optional, Tuple, Union
import torch
import copy
from layered_intervenable_model import LayeredIntervenableModel, LinearLayer, InverseLinearLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class SequentialLayers(torch.nn.Module):
    def __init__(self, layers, target_dims=None):
        super().__init__()
        self.layers = layers
        self.target_dims = target_dims

    def forward(
        self,
        hidden_states,
        layer_num=0,
        eot_indices: Optional[torch.Tensor] = None,     # receive info about inputs, to be used in intervention
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        args = (
            hidden_states,
            layer_num,
            eot_indices,
            attention_mask,
            causal_attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )
        args = self.layers[0](*args)
        count = 0

        prefix = None
        suffix = None
        
        first_linear = True
        first_inverse = True

        st = self.target_dims["start"]
        en = self.target_dims["end"]

        for layer in self.layers[1:]:
            if isinstance(layer, LinearLayer):
                output = args[0]
                sz = copy.deepcopy(output.shape)
                # output = torch.reshape(output, (original_shape[0], -1))
                rest = args[1:]
                if self.target_dims is None:
                    args = layer(output)
                else:
                    # target = output[:,self.target_dims["start"]:self.target_dims["end"]]
                    # prefix = output[:,:self.target_dims["start"]]
                    # suffix = output[:,self.target_dims["end"]:]

                    # get intervention coordinates over eot token
                    target = output[
                        torch.arange(output.size(0)), eot_indices, st:en
                    ]
                    args = layer(target)
            elif isinstance(layer, InverseLinearLayer):
                intervention = layer(args)
                reps = torch.stack([
                    torch.cat((
                        # everything before intervention for the i-th input in the batch
                        output.flatten()[i*sz[1]*sz[2]:i*sz[1]*sz[2] + j*sz[2] + st],
                        # intervention for i-th input
                        intervention[i],
                        # everything after intervention for teh i-th input in the batch
                        output.flatten()[i*sz[1]*sz[2] + j*sz[2] + en:(i+1)*sz[1]*sz[2]]
                    ))
                    for i, j in zip(torch.arange(output.size(0)), eot_indices)
                ]).view(sz)

                # args = (torch.cat([prefix, layer(args), suffix], 1).reshape(original_shape), *rest)
                args = (reps, *rest)
            else:
                args = layer(*args)
        return args
    
    ############## NOTE: is the function below for when we are intervening on multiple variables?? ###################

    # def forward(self,
    #             hidden_states,
    #             layer_num=0,
    #             attention_mask=None,
    #             head_mask=None,
    #             encoder_hidden_states=None,
    #             encoder_attention_mask=None,
    #             past_key_values=None,
    #             use_cache=None,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #             return_dict=True):

    #     args = (hidden_states,
    #             layer_num,
    #             attention_mask,
    #             head_mask,
    #             encoder_hidden_states,
    #             encoder_attention_mask,
    #             past_key_values,
    #             use_cache,
    #             output_attentions,
    #             output_hidden_states,
    #             return_dict)
    #     args = self.layers[0](*args)
    #     count = 0

    #     prefix = None
    #     suffix = None
        
    #     first_linear = True
    #     first_inverse = True
    #     for layer in self.layers[1:]:
    #         if isinstance(layer, LinearLayer):
    #             if first_linear:
    #                 output = args[0]
    #                 original_shape = copy.deepcopy(output.shape)
    #                 output = torch.reshape(output, (original_shape[0], -1))
    #                 rest = args[1:]
    #                 if self.target_dims is None:
    #                     args = layer(output)
    #                 else:
    #                     target = output[:,self.target_dims["start"]:self.target_dims["end"]]
    #                     prefix = output[:,:self.target_dims["start"]]
    #                     suffix = output[:,self.target_dims["end"]:]
    #                     args = layer(target)
    #                 first_linear = False
    #             else:
    #                 args = layer(target)
    #         elif isinstance(layer, InverseLinearLayer):
    #             if first_inverse:
    #                 args = layer(args)
    #                 first_inverse = False
    #             else:
    #                 args = (torch.cat([prefix, layer(args), suffix], 1).reshape(original_shape), *rest)
    #         else:
    #             args = layer(*args)
    #     return args


class LIMClipEncoderLayer(torch.nn.Module):
    def __init__(self, layer, final_layer_num):
        super().__init__()
        self.layer = layer
        self.final_layer_num = final_layer_num

    def forward(
        self,
        hidden_states,
        layer_num=0,
        eot_indices: Optional[torch.Tensor] = None,     # pass down eot indices for intervention location
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        
        # run our single layer
        layer_outputs = self.layer(
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # if it's the final layer, then return whatever encoder should output
        if layer_num == self.final_layer_num:
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states, 
                        encoder_states, 
                        all_attentions
                    ]
                    if v is not None
                )
            return BaseModelOutput(
                        last_hidden_state=hidden_states,
                        hidden_states=hidden_states,
                        attentions=all_attentions
                    )
        
        # otherwise, update hidden states and layer num, and pass down all arguments
        return (
            hidden_states,
            layer_num + 1,
            eot_indices,
            attention_mask,
            causal_attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )


class LIMClipTextModel(LayeredIntervenableModel):
    def __init__(
        self,
        clip,
        max_length=20,
        device=None,
        use_wrapper=True,
        debug=False,
        target_dims=None,
        target_layers=None,
        static_search=False,
        nested_disentangle_inplace=False,
        # set to True in order to learn intervention vector
        learn_intervention_vector=False 
    ):
        super().__init__(
            debug=debug,
            use_wrapper=use_wrapper,
            target_dims=target_dims,
            target_layers=target_layers,
            device=device
        )

        self.n_classes = None  # marker for CLIP loss (ver scrappy code here...)
        
        self.combiner = SequentialLayers
        self.num_layers = len(clip.text_model.encoder.layers)
        self.hidden_dim = clip.text_model.embeddings.token_embedding.embedding_dim
        self.model_dim = self.hidden_dim*max_length
        if self.target_dims is not None:
            self.model_dim = self.target_dims["end"]-self.target_dims["start"]
        
        self.embeddings = clip.text_model.embeddings
        self.model_layers = torch.nn.ModuleList()
        for layer in clip.text_model.encoder.layers:
            self.model_layers.append(LIMClipEncoderLayer(layer, self.num_layers - 1))
        
        self.final_layer_norm = clip.text_model.final_layer_norm

        self.text_projection = clip.text_projection

        # logit scale for similarity score
        self.logit_scale = clip.logit_scale

        # learn intervention during training if its dimension is supplied
        self.learn_intervention_vector = learn_intervention_vector
        if self.learn_intervention_vector:
            self.intervention_vector = torch.nn.Parameter(torch.randn(self.model_dim))
        
        self.build_graph(self.model_layers, self.model_dim, static_search, nested_disentangle_inplace)

    def freeze_model_parameters(self):
        """Freezes the model weights (for analysis purposes)"""
        for param in self.embeddings.parameters():
            param.requires_grad = False
            param.grad = None
        for param in self.final_layer_norm.parameters():
            param.requires_grad = False
            param.grad = None
        for param in self.text_projection.parameters():
            param.requires_grad = False
            param.grad = None
        self.logit_scale.requires_grad = False
        self.logit_scale.grad = None
        super().freeze_model_parameters()
        
    def unfreeze_model_parameters(self):
        """Unfreezes the model weights (for training purposes)"""
        for param in self.embeddings.parameters():
            param.requires_grad = True
        for param in self.final_layer_norm.parameters():
            param.requires_grad = True
        for param in self.text_projection.parameters():
            param.requires_grad = True
        self.logit_scale.requires_grad = True
        super().unfreeze_model_parameters()
    
    def get_text_encodings(self, inputs):
        X, attention_mask = inputs
        input_shape = X.size()

        eot_indices = X.to(torch.int).argmax(dim=-1)

        hidden_states = self.embeddings(X)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        if self.analysis:
            last_hidden_state = self.analysis_model(
                hidden_states,
                eot_indices=eot_indices,
                attention_mask=attention_mask, 
                causal_attention_mask=causal_attention_mask
            )[0]
        else:
            last_hidden_state = self.normal_model(
                hidden_states, 
                eot_indices=eot_indices,
                attention_mask=attention_mask, 
                causal_attention_mask=causal_attention_mask
            )[0]

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), eot_indices
        ]

        text_embeds = self.text_projection(pooled_output)
        return text_embeds

    def forward(self, inputs):
        """
        Computes a forward pass with input `X`.
        
        Input is a tuple of:
         - X : torch.Tensor, token id's of text input
         - attention_mask : torch.Tensor, attention mask of text input
         - image_embeds : torch.Tensor, pre-computed CLIP encoding of image to describe
        """
        X, attention_mask, image_embeds = inputs
        input_shape = X.size()

        eot_indices = X.to(torch.int).argmax(dim=-1)

        hidden_states = self.embeddings(X)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        if self.analysis:
            last_hidden_state = self.analysis_model(
                hidden_states,
                eot_indices=eot_indices,
                attention_mask=attention_mask, 
                causal_attention_mask=causal_attention_mask
            )[0]
        else:
            last_hidden_state = self.normal_model(
                hidden_states, 
                eot_indices=eot_indices,
                attention_mask=attention_mask, 
                causal_attention_mask=causal_attention_mask
            )[0]

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), eot_indices
        ]

        text_embeds = self.text_projection(pooled_output)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.squeeze(1).t()) * logit_scale

        return logits_per_text

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def iit_forward(self,
            base_pair,
            sources_pair,
            intervention_ids,
            intervention_ids_to_coords):
        """
        Computes a multi-source interchange interventions with base input `base`
        and source inputs in the list `sources`. `intervention_ids` contain
        integers that indicate where to perform interventions and
        `intervention_ids_to_coords` is the dictionary used to translate these
        integers to coordinates denoting the layer, start index, and end index.
        """
        base_input, base_mask, base_image_embeds = base_pair
        sources_input, sources_mask, source_image_embeds = sources_pair


        #unstack sources
        if len(sources_mask.shape) == 2:
            sources_mask = torch.unsqueeze(sources_mask, dim=1)
        if len(sources_input.shape) == 2:
            sources_input = torch.unsqueeze(sources_input, dim=1)
        if len(source_image_embeds.shape) == 2:
            source_image_embeds = torch.unsqueeze(source_image_embeds, dim=1)

        sources_mask = [sources_mask[:,j,:].squeeze(1).type(torch.FloatTensor).long().to(sources_mask.device)
           for j in range(sources_mask.shape[1])]
        sources_input = [sources_input[:,j,:].squeeze(1).type(torch.FloatTensor).long().to(sources_input.device)
           for j in range(sources_input.shape[1])]
        source_image_embeds = [source_image_embeds[:,j,:].squeeze(1).to(source_image_embeds.device)
           for j in range(source_image_embeds.shape[1])]
        #translate intervention_ids to coordinates
        gets =  intervention_ids_to_coords[int(intervention_ids.flatten()[0])]
        sets = copy.deepcopy(gets)
        self.activation = dict()

        #retrieve the value of interventions by feeding in the source inputs
        for i, get in enumerate(gets):
            handlers = self._gets_sets(gets =[get],sets = None)
            source_logits = self.forward((sources_input[i], sources_mask[i], source_image_embeds[i]))
            for handler in handlers:
                handler.remove()

            if self.learn_intervention_vector:
                # NOTE: by learning interventino, we are completely ignoring the "clean" run
                # repeat intervention to match batch size (1st dimension of input)
                sets[i]['intervention'] = self.intervention_vector.repeat((base_input.size(0), 1))
            else:
                sets[i]["intervention"] =\
                    self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']


        handlers = self._gets_sets(gets = None, sets = sets)
        counterfactual_logits = self.forward((base_input, base_mask, base_image_embeds))
        for handler in handlers:
            handler.remove()
        return counterfactual_logits

    def intervention_wrapper(self, output, set):
        sz = copy.deepcopy(output[0].shape)
        st = set['start']
        en = set['end']

        eot_indices = output[2]

        # admittedly, this is not very pretty code
        # one day i will sit down and learn how to use indices in pytorch properly
        # but for now, this will (hopefully) do
        reps = torch.stack([
            torch.cat((
                # everything before intervention for the i-th input in the batch
                output[0].flatten()[i*sz[1]*sz[2]:i*sz[1]*sz[2] + j*sz[2] + st],
                # intervention for i-th input
                set['intervention'][i],
                # everything after intervention for teh i-th input in the batch
                output[0].flatten()[i*sz[1]*sz[2] + j*sz[2] + en:(i+1)*sz[1]*sz[2]]
            ))
            for i, j in zip(torch.arange(output[0].size(0)), eot_indices)
        ]).view(sz)

        return tuple([reps] + list(output[1:]))

    def retrieval_wrapper(self, output, get):
        eot_indices = output[2]
        reps = output[0][
            torch.arange(output[0].size(0)), eot_indices, get["start"]: get["end"]
        ]
        return reps