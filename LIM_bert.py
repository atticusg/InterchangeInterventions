import torch
import copy
from layered_intervenable_model import LayeredIntervenableModel, LinearLayer, InverseLinearLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class SequentialLayers(torch.nn.Module):
    def __init__(self, layers, target_dims=None):
        super().__init__()
        self.layers = layers
        self.target_dims = target_dims

    def forward(self,
                hidden_states,
                layer_num=0,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):

        args = (hidden_states,
                layer_num,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict)
        args = self.layers[0](*args)
        count = 0

        prefix = None
        suffix = None
        
        first_linear = True
        first_inverse = True
        for layer in self.layers[1:]:
            if isinstance(layer, LinearLayer):
                output = args[0]
                original_shape = copy.deepcopy(output.shape)
                output = torch.reshape(output, (original_shape[0], -1))
                rest = args[1:]
                if self.target_dims is None:
                    args = layer(output)
                else:
                    target = output[:,self.target_dims["start"]:self.target_dims["end"]]
                    prefix = output[:,:self.target_dims["start"]]
                    suffix = output[:,self.target_dims["end"]:]
                    args = layer(target)
            elif isinstance(layer, InverseLinearLayer):
                args = (torch.cat([prefix, layer(args), suffix], 1).reshape(original_shape), *rest)
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



class LIMBertLayer(torch.nn.Module):
    def __init__(self, layer, final_layer_num):
        super().__init__()
        self.layer = layer
        self.final_layer_num = final_layer_num

    def forward(self,
                hidden_states,
                layer_num=0,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None

        next_decoder_cache = () if use_cache else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[layer_num] if head_mask is not None else None
        past_key_value = past_key_values[layer_num] if past_key_values is not None else None

        layer_outputs = self.layer(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if layer_num == self.final_layer_num:
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_decoder_cache,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                        last_hidden_state=hidden_states,
                        past_key_values=next_decoder_cache,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attentions,
                        cross_attentions=all_cross_attentions,
                    )

        return hidden_states, \
                layer_num + 1, \
                attention_mask, \
                head_mask, \
                encoder_hidden_states, \
                encoder_attention_mask, \
                past_key_values, \
                use_cache, \
                output_attentions, \
                output_hidden_states, \
                return_dict


class LIMBERTClassifier(LayeredIntervenableModel):
    def __init__(self,
                n_classes,
                bert,
                max_length=20,
                device=None,
                use_wrapper=True,
                debug=False,
                target_dims=None,
                target_layers=None,
                static_search=False,
                nested_disentangle_inplace=False,
                learned_intervention_size=None   # set to None in order to not learn an intervention
         ):
        super().__init__(
            debug=debug,
            use_wrapper=use_wrapper,
            target_dims=target_dims,
            target_layers=target_layers,
            device=device
        )
        
        self.combiner = SequentialLayers
        self.n_classes = n_classes
        self.num_layers = 12
        self.hidden_dim = bert.embeddings.word_embeddings.embedding_dim
        self.model_dim = self.hidden_dim*max_length
        if self.target_dims is not None:
            self.model_dim = self.target_dims["end"]-self.target_dims["start"]
        
        self.embeddings = bert.embeddings
        self.model_layers = torch.nn.ModuleList()
        for layer in bert.encoder.layer:
            self.model_layers.append(LIMBertLayer(layer, len(bert.encoder.layer)-1))
        self.get_extended_attention_mask = bert.get_extended_attention_mask
        self.classifier_layer = torch.nn.Linear(self.hidden_dim, self.n_classes)
        self.pooler = bert.pooler
        
        self.build_graph(self.model_layers, self.model_dim, static_search, nested_disentangle_inplace)

        # NOTE: LEARN INTERVENTION BASED ON INPUT INTERVENTION SIZE
        self.learned_intervention_size = learned_intervention_size
        self.set_learn_intervention_vector(self.learned_intervention_size is not None)

    # NOTE: AMIR'S ADDITION
    def set_learn_intervention_vector(self, learn_intervention_vector):
        self.learn_intervention_vector = learn_intervention_vector
        if self.learn_intervention_vector:
            self.intervention_vector = torch.nn.Parameter(torch.randn(self.learned_intervention_size))
            self.freeze_model_parameters()
            self.freeze_disentangling_parameters()
        else:
            # restore to original configuration
            self.intervention_vector = None
            self.set_analysis_mode(self.analysis)

    def freeze_model_parameters(self):
        """Freezes the model weights (for analysis purposes)"""
        for param in self.embeddings.parameters():
            param.requires_grad = False
            param.grad = None
        for param in self.classifier_layer.parameters():
            param.requires_grad = False
            param.grad = None
        for param in self.pooler.parameters():
            param.requires_grad = False
            param.grad = None
        super().freeze_model_parameters()
        
    def unfreeze_model_parameters(self):
        """Unfreezes the model weights (for training purposes)"""
        for param in self.embeddings.parameters():
            param.requires_grad = True
        for param in self.classifier_layer.parameters():
            param.requires_grad = True
        for param in self.pooler.parameters():
            param.requires_grad = True
        super().freeze_model_parameters()
        
    def forward(self, pair, output_hidden_states=False):
        """Computes a forward pass with input `X`."""
        X, mask = pair
        input_shape = X.size()
        extended_attention_mask = self.get_extended_attention_mask(mask, input_shape)
        X = self.embeddings(X)
        if self.analysis:
            output = self.analysis_model(X, attention_mask=extended_attention_mask)[0]
        else:
            output = self.normal_model(X, attention_mask=extended_attention_mask)[0]
        pooler_output = self.pooler(output)
        if output_hidden_states:
            return pooler_output
        return self.classifier_layer(pooler_output)

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
        base_input, base_mask = base_pair
        sources_input, sources_mask = sources_pair


        #unstack sources
        if len(sources_mask.shape) == 2:
            sources_mask = torch.unsqueeze(sources_mask, dim=1)
        if len(sources_input.shape) == 2:
            sources_input = torch.unsqueeze(sources_input, dim=1)

        sources_mask = [sources_mask[:,j,:].squeeze(1).type(torch.FloatTensor).long().to(sources_mask.device)
           for j in range(sources_mask.shape[1])]
        sources_input = [sources_input[:,j,:].squeeze(1).type(torch.FloatTensor).long().to(sources_input.device)
           for j in range(sources_input.shape[1])]
        #translate intervention_ids to coordinates
        gets =  intervention_ids_to_coords[int(intervention_ids.flatten()[0])]
        sets = copy.deepcopy(gets)
        self.activation = dict()

        #retrieve the value of interventions by feeding in the source inputs
        for i, get in enumerate(gets):
            handlers = self._gets_sets(gets =[get],sets = None)
            source_logits = self.forward((sources_input[i], sources_mask[i]))
            for handler in handlers:
                handler.remove()
            sets[i]["intervention"] =\
                self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']


        handlers = self._gets_sets(gets = None, sets = sets)
        counterfactual_logits = self.forward((base_input, base_mask))
        for handler in handlers:
            handler.remove()
        return counterfactual_logits

    def intervention_wrapper(self,output, set):
        original_shape = copy.deepcopy(output[0].shape)
        reps = output[0].reshape((original_shape[0], -1))
        reps = torch.cat([reps[:,:set["start"]], set["intervention"],
                            reps[:,set["end"]:]],
                            dim = 1)
        return tuple([reps.reshape(original_shape)]\
                                + [ _ for _ in output[1:]])

    def forward_with_intervention(self, pair, intervention_ids_to_coords):
        input_size = pair[0].size()

        # NOTE: FOR NOW, ASSUME ONLY A SINGLE INTERVENTION AT 0
        sets = copy.deepcopy(intervention_ids_to_coords[0])
        for i, get in enumerate(sets):
            # NOTE: by learning interventinon, we are completely ignoring the "clean" run
            # repeat intervention to match batch size (1st dimension of input)
            sets[i]['intervention'] = self.intervention_vector.repeat((input_size[0], 1))
        
        handlers = self._gets_sets(gets=None, sets=sets)
        counterfactual_logits = self.forward(pair)
        for handler in handlers:
            handler.remove()
        return counterfactual_logits

    def retrieval_wrapper(self, output, get):
        reps = output[0].reshape((output[0].shape[0], -1))
        return reps[:,get["start"]: get["end"] ]


