import torch
import copy
from layered_intervenable_model import LayeredIntervenableModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class SequentialLayers(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = args

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
        for layer in self.layers[1:]:
            args = layer(*args)
        return args



class LIMBertLayer(torch.nn.Module):
    def __init__(self, bert,layer, final_layer_num):
        super().__init__()
        self.bert = bert
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
        all_cross_attentions = () if output_attentions and self.bert.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[layer_num] if head_mask is not None else None
        past_key_value = past_key_values[layer_num] if past_key_values is not None else None
        #PUSHTHIS TO GIT
        if getattr(self.bert.config, "gradient_checkpointing", False) and self.bert.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                                        create_custom_forward(layer_module),
                                        hidden_states,
                                        attention_mask,
                                        layer_head_mask,
                                        encoder_hidden_states,
                                        encoder_attention_mask,
                                        )
        else:
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
                if self.bert.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

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
                debug=False):
        super().__init__(debug=debug)
        self.combiner = SequentialLayers
        self.n_classes = n_classes
        self.bert = bert
        self.bert.train()
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        n = self.hidden_dim*max_length

        self.model_dims = [n]
        self.model_layers = torch.nn.ModuleList()
        for layer in self.bert.encoder.layer:
            self.model_layers.append(LIMBertLayer(bert, layer, len(self.bert.encoder.layer)-1))
            self.model_dims.append(n)
        self.classifier_layer = torch.nn.Linear(self.hidden_dim, self.n_classes)

        def unwrap(X):
            output = X[0]
            original_shape = copy.deepcopy(output.shape())
            output = torch.reshape(output, (original_shape[0], -1))
            rest = X[1:]
            return output, (rest, original_shape)

        def rewrap(output, stow):
            rest, original_shape = stow
            return (X[0].reshape(original_shape), *rest)


        self.build_graph(self.model_layers, self.model_dims, unwrap, rewrap)


    def forward(self, pair):
        """Computes a forward pass with input `X`."""
        X, mask = pair
        X = torch.squeeze(X).long()
        mask = torch.squeeze(mask)
        if self.analysis:
            self.bert.encoder = self.analysis_model
        else:
            self.bert.encoder =  self.normal_model
        output = self.bert(X, mask).pooler_output
        output = self.classifier_layer(output)
        return output

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
        sources_mask = [sources_mask[:,j,:].squeeze(1).type(torch.FloatTensor).to(self.device)
           for j in range(sources_mask.shape[1])]
        sources_input = [sources_input[:,j,:].squeeze(1).type(torch.FloatTensor).to(self.device)
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

    def intervention_wrapper(self,output):
        original_shape = copy.deepcopy(output[0].shape)
        reps = output[0].reshape((original_shape[0], -1))
        reps = torch.cat([reps[:,:set["start"]], set["intervention"],
                            reps[:,set["end"]:]],
                            dim = 1)
        return reps.reshape(original_shape), *output[1:]
